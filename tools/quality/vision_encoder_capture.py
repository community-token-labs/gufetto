#!/usr/bin/env python3
"""Private, stdlib-only prepare/run driver for the independent CPU encoder.

Freeze the printed packet.json SHA256 externally BEFORE run. No downloading,
resizing, production imports, cached-result reuse, or kernel qualification.
Failed runs retain private logs/partial outputs and cannot be resumed in place.
Source identities are canonical JSON {basename: SHA256} hashed as UTF-8, with
sorted keys, compact separators and a trailing newline (see encoded()).
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import time
import zlib


LIMIT = 64 * 1024 * 1024
MAX_AREA = 16_777_216
PRODUCTION = (907542944, "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355")
DEVELOPMENT = (907542784, "19f74eef0db4408ca568fabcb26dd2584f797e6b2f9218c48ac5362791f21941")
STAGES = ["patch", *[f"layer{i}" for i in range(27)], "embedding"]
SOURCE_FILES = tuple(Path(__file__).resolve().with_name(name) for name in (
    "vision_encoder_capture.py", "vision_ref_math.hpp", "vision_ref_math.cpp",
    "vision_encoder_reference.cpp"))
ARITHMETIC_PLAN = dict(projections="f32", qk="f32", softmax="f64", pv="f64",
                       output="f32", query_block=128, full_keys=True)
MIN_AVAILABLE, MIN_DISK, MAX_RSS = 6 * 1024**3, 80_000_000_000, 12 * 1024**3
GUARDRAILS = dict(min_available_bytes=MIN_AVAILABLE, min_disk_free_bytes=MIN_DISK,
                  max_parent_child_rss_bytes=MAX_RSS, poll_seconds=0.5, nice=19,
                  threads=4, runtime_limit_seconds=None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def bounded(path, limit=LIMIT):
    with Path(path).open("rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "input must be a regular file")
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, f"input exceeds {limit} byte budget")
    return raw


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    raw = bounded(path)
    def reject(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject), sha(raw)


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "input must be a regular file")
        while data := stream.read(1024 * 1024):
            digest.update(data)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "input changed while hashing")
    return dict(path=str(path), size=after.st_size, sha256=digest.hexdigest())


def source_identity(paths=None):
    files = {p.name: identity(p) for p in map(Path, SOURCE_FILES if paths is None else paths)}
    return files, sha(encoded({name: item["sha256"] for name, item in files.items()}))


def private(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def save(path, data):
    with private(path) as output:
        output.write(data)


def decode_png(path):
    """Only frozen RGB8/noninterlaced/filter-0 PNG; reject all other formats."""
    raw = bounded(path)
    require(raw[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG signature")
    position, chunks, compressed = 8, [], bytearray()
    width = height = 0
    while position < len(raw):
        require(position + 12 <= len(raw), "truncated PNG chunk")
        size, kind = struct.unpack_from(">I4s", raw, position)
        end = position + 12 + size
        require(end <= len(raw), "truncated PNG payload")
        payload = raw[position + 8:end - 4]
        require(zlib.crc32(kind + payload) & 0xffffffff ==
                struct.unpack_from(">I", raw, end - 4)[0], "PNG CRC mismatch")
        if kind == b"IHDR":
            require(not chunks and size == 13, "invalid PNG IHDR order/size")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            require((depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0),
                    "unsupported PNG format")
            require(width > 0 and height > 0 and width % 32 == height % 32 == 0
                    and width * height <= MAX_AREA, "PNG dimensions require multiples of 32 within area budget")
        elif kind == b"IDAT":
            require(chunks and chunks[-1] in (b"IHDR", b"IDAT"), "invalid PNG IDAT order")
            compressed.extend(payload)
        elif kind == b"IEND":
            require(chunks and chunks[-1] == b"IDAT" and size == 0 and end == len(raw),
                    "invalid PNG IEND/trailing bytes")
        else:
            raise ValueError("unsupported PNG chunk")
        chunks.append(kind)
        position = end
    require(chunks and chunks[-1] == b"IEND", "missing PNG IEND")
    expected = height * (width * 3 + 1)
    decoder = zlib.decompressobj()
    scanlines = decoder.decompress(compressed, expected + 1)
    require(len(scanlines) == expected and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail, "PNG zlib length/truncation/trailing data")
    stride = width * 3 + 1
    require(all(scanlines[i] == 0 for i in range(0, expected, stride)), "unsupported PNG filter")
    pixels = b"".join(scanlines[i + 1:i + stride] for i in range(0, expected, stride))
    return width, height, pixels, sha(raw)


def select_case(plan, manifest, manifest_hash, case_id, width, height, rgb_hash, png_hash):
    require(manifest.get("schema") == "gufo-vision-images-v1", "unsupported images manifest")
    cases = [case for case in manifest["cases"] if case["id"] == case_id]
    require(len(cases) == 1, "expected exactly one image case")
    case = cases[0]
    require(type(case["width"]) is int and type(case["height"]) is int and
            (case["width"], case["height"], case["sha256"], case["rgb_sha256"]) ==
            (width, height, png_hash, rgb_hash), "manifest dimensions/PNG/RGB hash mismatch")
    require(plan.get("schema") == "gufo-vision-tensors-v1" and plan.get("scope") == "encoder"
            and plan.get("image_set_sha256") == manifest_hash, "plan schema/scope/image manifest hash mismatch")
    tensors = [t for t in plan["tensors"] if t["case_id"] == case_id]
    require(len(tensors) == 29 and {t["stage"] for t in tensors} == set(STAGES), "expected exactly 29 stages")
    require(len({t["id"] for t in tensors}) == 29, "duplicate tensor id")
    for tensor in tensors:
        stage = tensor["stage"]
        rows = width * height // (1024 if stage == "embedding" else 256)
        indices = tensor["row_indices"]
        require(type(tensor["width"]) is int and tensor["width"] == (2560 if stage == "embedding" else 1152)
                and tensor["input_sha256"] == rgb_hash, "tensor width/RGB hash mismatch")
        require(isinstance(tensor["id"], str) and tensor["id"] and
                isinstance(tensor.get("stratum"), str) and tensor["stratum"] and
                type(tensor.get("head")) is int and tensor["head"] == -1 and
                type(tensor.get("layer")) is int and
                tensor["layer"] == (int(stage[5:]) if stage.startswith("layer") else -1), "invalid tensor descriptor")
        require(isinstance(indices, list) and indices and all(type(i) is int and 0 <= i < rows for i in indices)
                and all(a < b for a, b in zip(indices, indices[1:])), "indices must be in range, sorted and unique")
    return {**plan, "tensors": tensors}


def prepare(args):
    expected = DEVELOPMENT if args.development else PRODUCTION
    require(Path(args.model).stat().st_size == expected[0], "model size mismatch")
    model = identity(args.model)
    require((model["size"], model["sha256"]) == expected, "model SHA256 mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision) and args.toolchain.strip(), "full revision/toolchain required")
    manifest, manifest_hash = load(args.images)
    plan, plan_hash = load(args.plan)
    width, height, pixels, png_hash = decode_png(args.image)
    rgb_hash = sha(pixels)
    case_plan = select_case(plan, manifest, manifest_hash, args.case, width, height, rgb_hash, png_hash)
    sources, source_hash = source_identity()
    inputs = {name: identity(getattr(args, name)) for name in ("binary", "images", "plan", "image")}
    require(inputs["images"]["sha256"] == manifest_hash and inputs["plan"]["sha256"] == plan_hash
            and inputs["image"]["sha256"] == png_hash, "input changed during prepare")
    inputs["model"] = model
    producer = dict(implementation="independent_vision_encoder_cpu", input_origin="pixels", independent=True,
                    independence="Independent PNG decoder and CPU forward from pixels and GGUF weights; no Gufo encoder, generator imports or candidate hidden states.",
                    source_sha256=source_hash, model_sha256=model["sha256"], revision=args.revision,
                    toolchain=args.toolchain, binary_sha256=inputs["binary"]["sha256"],
                    accumulator_dtype="f32_qk_f64_softmax_pv", arithmetic_plan=ARITHMETIC_PLAN,
                    arithmetic="F32 projections and QK; F64 full-key softmax and PV; F32 output; query blocks 128; no BF16 rounding; FMA off.",
                    production_or_development="NON_QUALIFYING" if args.development else "production",
                    parent_plan_sha256=plan_hash, image_case=args.case, image_sha256=png_hash,
                    rgb_sha256=rgb_hash, transformers_revision="50e35cc0ca18ec19b80653690de553e6decf74f5",
                    hf_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540", normalization="(F32(pixel)-127.5f)/127.5f",
                    axial_theta=10000, resize=False, threads=4, fma=False)
    header = dict(schema="gufo-vision-capture-v1", role="reference", dtype="f32le",
                  rows=sum(len(t["row_indices"]) for t in case_plan["tensors"]),
                  plan_sha256=sha(encoded(case_plan)), producer=producer)
    by_stage = {t["stage"]: t["row_indices"] for t in case_plan["tensors"]}
    selection = "".join(f"{s} {len(by_stage[s])} {' '.join(map(str, by_stage[s]))}\n" for s in STAGES)
    out = Path(args.out_dir).absolute()
    out.mkdir(mode=0o700)
    out.chmod(0o700)
    artifacts = {}
    for name, data in (("pixels.rgb8", pixels), ("selection.txt", selection.encode()),
                       ("case-plan.json", encoded(case_plan)), ("header.json", encoded(header))):
        save(out / name, data)
        artifacts[name] = dict(size=len(data), sha256=sha(data))
    packet = dict(schema="gufo-vision-encoder-packet-v1", inputs=inputs, sources=sources,
                  artifacts=artifacts, width=width, height=height, development=args.development,
                  guardrails=GUARDRAILS)
    save(out / "packet.json", encoded(packet))
    return sha(encoded(packet))


def verify(packet, directory):
    require(packet["schema"] == "gufo-vision-encoder-packet-v1", "unsupported packet")
    require(packet["guardrails"] == GUARDRAILS, "resource guardrail pin mismatch")
    require(set(packet["inputs"]) == {"binary", "model", "images", "plan", "image"}
            and set(packet["artifacts"]) == {"pixels.rgb8", "selection.txt", "case-plan.json", "header.json"},
            "incomplete packet pins")
    for item in [*packet["inputs"].values(), *packet["sources"].values()]:
        require(identity(item["path"]) == item, f"input/source hash drift: {item['path']}")
    sources, source_hash = source_identity()
    require(sources == packet["sources"], "current source identity mismatch")
    for name, expected in packet["artifacts"].items():
        actual = identity(directory / name)
        require(not (directory / name).is_symlink() and
                {k: actual[k] for k in expected} == expected, f"prepared artifact hash drift: {name}")
    header, _ = load(directory / "header.json")
    producer = header["producer"]
    model = packet["inputs"]["model"]
    require(type(packet["development"]) is bool, "invalid development label")
    require((model["size"], model["sha256"]) == (DEVELOPMENT if packet["development"] else PRODUCTION)
            and producer["production_or_development"] == ("NON_QUALIFYING" if packet["development"] else "production")
            and producer["source_sha256"] == source_hash, "model/source/qualification pin mismatch")
    return header


def proc_kib(path, field):
    # proc files are not normal-sized regular files; use a separate bounded read.
    with open(path, "rb") as stream:
        lines = stream.read(65536).decode().splitlines()
    for line in lines:
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    if field == "VmRSS" and any(line.startswith("State:") and "Z" in line for line in lines):
        return 0
    raise ValueError(f"missing resource metric {path}:{field}")


def resources(directory, pid=None):
    rss = proc_kib(f"/proc/{os.getpid()}/status", "VmRSS")
    if pid is not None:
        try:
            rss += proc_kib(f"/proc/{pid}/status", "VmRSS")
        except FileNotFoundError:  # child exited between poll and /proc read
            pass
    return dict(available=proc_kib("/proc/meminfo", "MemAvailable"), rss=rss,
                disk_free=shutil.disk_usage(directory).free)


def guard(directory, pid, receipt):
    sample = resources(directory, pid)
    receipt["peak_rss"] = max(receipt.get("peak_rss", 0), sample["rss"])
    for metric in ("available", "disk_free"):
        key = "lowest_" + metric
        receipt[key] = min(receipt.get(key, sample[metric]), sample[metric])
    require(sample["available"] >= MIN_AVAILABLE and sample["disk_free"] >= MIN_DISK
            and sample["rss"] <= MAX_RSS, "resource guardrail breached")


def stop(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def capture_index(directory, header):
    plan, _ = load(directory / "case-plan.json")
    require(not (directory / "raw").is_symlink(), "output directory must not be a symlink")
    with private(directory / "capture.jsonl.partial") as output:
        output.write(encoded(header))
        for tensor in plan["tensors"]:
            relative = f"raw/{tensor['stage']}.f32le"
            path = directory / relative
            width = tensor["width"] * 4
            require(not path.is_symlink() and path.is_file() and
                    path.stat().st_size == len(tensor["row_indices"]) * width, "output size/type mismatch")
            with path.open("rb") as source:
                for offset, index in enumerate(tensor["row_indices"]):
                    row = source.read(width)
                    require(len(row) == width and all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", row)),
                            "truncated/nonfinite output row")
                    output.write(encoded(dict(tensor_id=tensor["id"], row=index, file=relative,
                                              offset=offset * width, sha256=sha(row))))
                require(not source.read(1), "extra output bytes")
    os.rename(directory / "capture.jsonl.partial", directory / "capture.jsonl")


def run(directory, expected_hash):
    directory = Path(directory).resolve(strict=True)
    packet, packet_hash = load(directory / "packet.json")
    require(packet_hash == expected_hash, "externally frozen packet SHA256 mismatch")
    require(not directory.stat().st_mode & 0o077, "packet directory must be private")
    prepared = {"packet.json", "pixels.rgb8", "selection.txt", "case-plan.json", "header.json"}
    require({p.name for p in directory.iterdir()} == prepared, "existing run outputs or incomplete packet")
    save(directory / "run.lock", b"exclusive attempt; no reuse\n")
    started, child, receipt = time.monotonic(), None, dict(status="FAILED", partial=True,
        packet_sha256=packet_hash, guardrails=GUARDRAILS)
    old_umask = os.umask(0o077)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        header = verify(packet, directory)
        guard(directory, None, receipt)
        (directory / "raw").mkdir(mode=0o700)
        env = os.environ.copy()
        env.update({key: "4" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                                         "OMP_THREAD_LIMIT")})
        env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
        command = ["nice", "-n", "19", packet["inputs"]["binary"]["path"], "--encode",
                   packet["inputs"]["model"]["path"], str(directory / "pixels.rgb8"),
                   str(packet["width"]), str(packet["height"]), str(directory / "selection.txt"), str(directory / "raw")]
        with private(directory / "stdout.log") as stdout, private(directory / "stderr.log") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=True)
            while True:
                guard(directory, child.pid, receipt)
                if child.poll() is not None:
                    break
                time.sleep(0.5)
        require(child.returncode == 0, f"backend exit {child.returncode}")
        verify(packet, directory)
        require(load(directory / "packet.json")[1] == packet_hash, "packet changed during run")
        capture_index(directory, header)
        receipt.update(status="referencecaptured", partial=False,
                       production_or_development=header["producer"]["production_or_development"], kernelqualified=False)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # A second interrupt must not strand the child during shutdown.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop(child)
            receipt.update(elapsed_seconds=time.monotonic() - started,
                           exit_code=None if child is None else child.returncode)
            save(directory / "resource.json", encoded(receipt))
            if receipt["status"] != "referencecaptured":
                save(directory / "failure.json", encoded(receipt))
        finally:
            os.umask(old_umask)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    completion = {**receipt, "capture_sha256": identity(directory / "capture.jsonl")["sha256"]}
    save(directory / "completion.json.partial", encoded(completion))
    os.rename(directory / "completion.json.partial", directory / "completion.json")
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("binary", "model", "images", "plan", "image", "out-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("case", "revision", "toolchain"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--development", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--packet", type=Path, required=True)
    execute.add_argument("--packet-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args) if args.command == "prepare" else run(args.packet, args.packet_sha256)
    except (ValueError, OSError, KeyError, TypeError, RecursionError, zlib.error, KeyboardInterrupt) as error:
        parser.exit(1, f"vision_encoder_capture: {error}\n")
    print(result if isinstance(result, str) else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
