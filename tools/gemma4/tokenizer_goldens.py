#!/usr/bin/env python3
"""Gemma 4 tokenizer goldens from the artifact's own vocabulary and merges.

The reference is a hand port of llama.cpp's `gemma4` tokenizer at
6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3 (src/llama-vocab.cpp: vocabulary
load, end-of-generation overrides, special-token partition, the BPE session
and its `<0xXX>` fallback; src/unicode.cpp: permissive UTF-8 decoding and the
"[^\\n]+|[\\n]+" splitter). It is written independently of Gufo's C++
tokenizer and reads only the GGUF metadata section, never tensor data.

When --donor-fixture is given, every corpus entry is also compared with the
llama.cpp ids recorded by gufo-org/gufo PR #426 (llama.cpp build 11069,
68d9053a, on a Gemma 4 31B GGUF); agreement is recorded per entry.

    python3 -I tools/gemma4/tokenizer_goldens.py --model TARGET.gguf \\
        --manifest MANIFEST.json [--donor-fixture DONOR.json]
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import platform
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_metadata import read_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/models/gemma4/fixtures/tokenizer_goldens.json"
LLAMA_CPP = "6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3"

# llama_token_attr bits.
UNKNOWN, UNUSED, NORMAL, CONTROL, USER_DEFINED, BYTE = 1, 2, 4, 8, 16, 32
TYPE_TO_ATTR = {2: UNKNOWN, 5: UNUSED, 1: NORMAL, 3: CONTROL, 4: USER_DEFINED,
                6: BYTE}
ESCAPED_SPACE = "▁".encode()

# Token texts llama.cpp promotes to end-of-generation control tokens
# (llama-vocab.cpp, "maintain a list of tokens that cause end-of-generation").
EOG_TEXTS = [
    "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
    "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>",
    "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
    "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
    "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>"]
# Every other text-matched override in that function (EOT/EOM/FIM/pad
# detection, the gpt-oss attribute hack, beam's marker). The port does not
# implement them, so it refuses vocabularies that contain one.
UNPORTED_TEXTS = [
    '<MID>',
    '<PAD>',
    '<PRE>',
    '<REPO>',
    '<SUF>',
    '<fim-middle>',
    '<fim-pad>',
    '<fim-prefix>',
    '<fim-repo>',
    '<fim-suffix>',
    '<fim_middle>',
    '<fim_pad>',
    '<fim_prefix>',
    '<fim_suffix>',
    '<reponame>',
    '<|channel|>',
    '<|code_middle|>',
    '<|code_prefix|>',
    '<|code_suffix|>',
    '<|constrain|>',
    '<|file_sep|>',
    '<|fim_middle|>',
    '<|fim_pad|>',
    '<|fim_prefix|>',
    '<|fim_repo|>',
    '<|fim_suffix|>',
    '<|message|>',
    '<|middle|>',
    '<|observation|>',
    '<|plamo:eos|>',
    '<|plamo:file_separator|>',
    '<|plamo:fim_middle|>',
    '<|plamo:fim_prefix|>',
    '<|plamo:fim_suffix|>',
    '<|prefix|>',
    '<|repo_name|>',
    '<|start|>',
    '<|suffix|>',
    '<|user|>',
    '<｜fim▁begin｜>',
    '<｜fim▁end｜>',
    '<｜fim▁hole｜>',
    '[PAD]',
    '▁<MID>',
    '▁<PRE>',
    '▁<SUF>',
    '<rfl:parameter name="',
]


def cpt_from_utf8(data: bytes, offset: int) -> tuple[int, int] | None:
    """unicode_cpt_from_utf8: (code point, length), or None when invalid."""
    b = data[offset]
    n = len(data)
    cont = lambda i: i < n and data[i] & 0xC0 == 0x80  # noqa: E731
    if not b & 0x80:
        return b, 1
    if not b & 0x40:
        return None
    if not b & 0x20:
        if not cont(offset + 1):
            return None
        cpt = (b & 0x1F) << 6 | data[offset + 1] & 0x3F
        return (cpt, 2) if cpt >= 0x80 else None
    if not b & 0x10:
        if not (cont(offset + 1) and cont(offset + 2)):
            return None
        cpt = ((b & 0x0F) << 12 | (data[offset + 1] & 0x3F) << 6
               | data[offset + 2] & 0x3F)
        return (cpt, 3) if cpt >= 0x800 else None
    if not b & 0x08:
        if not (cont(offset + 1) and cont(offset + 2) and cont(offset + 3)):
            return None
        cpt = ((b & 0x07) << 18 | (data[offset + 1] & 0x3F) << 12
               | (data[offset + 2] & 0x3F) << 6 | data[offset + 3] & 0x3F)
        return (cpt, 4) if 0x10000 <= cpt <= 0x10FFFF else None
    return None


def cpts_from_utf8(data: bytes) -> list[int]:
    """unicode_cpts_from_utf8: each invalid lead byte becomes U+FFFD."""
    out, offset = [], 0
    while offset < len(data):
        decoded = cpt_from_utf8(data, offset)
        if decoded is None:
            out.append(0xFFFD)
            offset += 1
        else:
            out.append(decoded[0])
            offset += decoded[1]
    return out


def cpt_to_utf8(cpt: int) -> bytes:
    """unicode_cpt_to_utf8, which also encodes surrogates."""
    if cpt <= 0x7F:
        return bytes([cpt])
    if cpt <= 0x7FF:
        return bytes([0xC0 | cpt >> 6 & 0x1F, 0x80 | cpt & 0x3F])
    if cpt <= 0xFFFF:
        return bytes([0xE0 | cpt >> 12 & 0x0F, 0x80 | cpt >> 6 & 0x3F,
                      0x80 | cpt & 0x3F])
    return bytes([0xF0 | cpt >> 18 & 0x07, 0x80 | cpt >> 12 & 0x3F,
                  0x80 | cpt >> 6 & 0x3F, 0x80 | cpt & 0x3F])


def utf8_len(lead: int) -> int:
    """unicode_len_utf8."""
    return (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)[lead >> 4]


class Vocab:
    def __init__(self, md: dict) -> None:
        if md["tokenizer.ggml.model"] != b"gemma4":
            raise SystemExit("tokenizer.ggml.model is not gemma4")
        tokens = md["tokenizer.ggml.tokens"]
        types = md["tokenizer.ggml.token_type"]
        self.text = [t if t else b"[EMPTY_%d]" % i for i, t in enumerate(tokens)]
        self.attr = [TYPE_TO_ATTR.get(t, 0) for t in types]
        self.to_id = {}
        for i, t in enumerate(self.text):
            self.to_id[t] = i  # token_to_id[word] = i: the last one wins
        self.bos = md["tokenizer.ggml.bos_token_id"]
        self.eos = md["tokenizer.ggml.eos_token_id"]
        self.add_bos = bool(md.get("tokenizer.ggml.add_bos_token", False))
        self.ranks = {}
        for rank, merge in enumerate(md["tokenizer.ggml.merges"]):
            sep = merge.find(b" ", 1) if merge else -1
            pair = (merge[:sep], merge[sep + 1:]) if sep >= 0 else (b"", b"")
            self.ranks.setdefault(pair, rank)  # bpe_ranks.emplace: first wins
        for text in UNPORTED_TEXTS:
            if text.encode() in self.to_id:
                raise SystemExit(f"unported llama.cpp override applies: {text}")
        self.eog = set()
        for text in EOG_TEXTS:
            if (i := self.to_id.get(text.encode())) is not None:
                self.eog.add(i)
                self.attr[i] |= CONTROL
        self.eog.add(self.eos)
        # gemma4, plamo and beam: "</s>" is a normal token when the
        # vocabulary also has "<|tool_response>" as an end-of-generation token.
        has_marker = any(self.text[i] == b"<|tool_response>" for i in self.eog)
        s_id = self.to_id.get(b"</s>")
        if has_marker and s_id in self.eog:
            self.eog.discard(s_id)
            self.attr[s_id] = NORMAL
        special = [i for i, a in enumerate(self.attr)
                   if a & (CONTROL | USER_DEFINED | UNKNOWN)]
        # std::sort by length; ties between equal-length texts cannot match
        # the same span differently, so their order does not matter.
        self.special = sorted(special, key=lambda i: -len(self.text[i]))

    def partition(self, text: bytes, parse_special: bool) -> list:
        """tokenizer_st_partition: raw fragments (bytes) and token ids."""
        fragments: list = [text] if text else []
        for sid in self.special:
            if not parse_special and self.attr[sid] & (CONTROL | UNKNOWN):
                continue
            needle = self.text[sid]
            out = []
            for f in fragments:
                if isinstance(f, int):
                    out.append(f)
                    continue
                start = 0
                while (match := f.find(needle, start)) >= 0:
                    if match > start:
                        out.append(f[start:match])
                    out.append(sid)
                    start = match + len(needle)
                if start < len(f):
                    out.append(f[start:])
            fragments = out
        return fragments

    def bpe_word(self, word: bytes, out: list[int]) -> None:
        if word.strip(b"\n") == b"" and word in self.to_id:
            out.append(self.to_id[word])  # whole newline run (llama.cpp #21343)
            return
        # Symbols: [text bytes or None when merged away].
        sym, offset = [], 0
        while offset < len(word):
            n = min(len(word) - offset, utf8_len(word[offset]))
            sym.append(word[offset:offset + n])
            offset += n
        prev = list(range(-1, len(sym) - 1))
        nxt = list(range(1, len(sym) + 1))
        nxt[-1] = -1
        queue: list = []

        def add(left: int, right: int) -> None:
            if left < 0 or right < 0:
                return
            rank = self.ranks.get((sym[left], sym[right]))
            if rank is not None:
                heapq.heappush(queue, (rank, left, right,
                                       sym[left] + sym[right]))

        for i in range(1, len(sym)):
            add(i - 1, i)
        while queue:
            _, left, right, text = heapq.heappop(queue)
            if not sym[left] or not sym[right] or sym[left] + sym[right] != text:
                continue
            sym[left] = text
            sym[right] = b""
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            add(prev[left], left)
            add(left, nxt[left])
        i = 0
        while i != -1:
            if sym[i]:
                if (token := self.to_id.get(sym[i])) is not None:
                    out.append(token)
                else:
                    for b in sym[i]:
                        byte = self.to_id.get(b"<0x%02X>" % b)
                        if byte is not None:
                            out.append(byte)
            i = nxt[i]

    def encode(self, text: bytes, add_bos: bool, parse_special: bool) -> list[int]:
        out = [self.bos] if add_bos and self.add_bos else []
        for f in self.partition(text, parse_special):
            if isinstance(f, int):
                out.append(f)
                continue
            cpts = cpts_from_utf8(f.replace(b" ", ESCAPED_SPACE))
            start = 0
            while start < len(cpts):
                newline = cpts[start] == 0x0A
                end = start
                while end < len(cpts) and (cpts[end] == 0x0A) == newline:
                    end += 1
                self.bpe_word(b"".join(cpt_to_utf8(c) for c in cpts[start:end]),
                              out)
                start = end
        return out


def corpus() -> list[tuple[str, bytes, bool, bool]]:
    """(name, text, parse_special, add_bos). The first 20 mirror PR #426's
    corpus so they can be compared with its llama.cpp ids."""
    special = True
    entries = [
        ("ascii", b"Hello world, this is a test."),
        ("spaces", b"hello  world   return x;    indented\t\tword"),
        ("leading_trailing_space", b"  padded text  "),
        ("newlines", b"line one\nline two\n\nline four\n\n\n\nend\n"),
        ("crlf", b"windows\r\nline\r\n"),
        ("only_newlines", b"\n\n\n\n\n"),
        ("code", b"def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n"),
        ("numbers", b"3.14159 2026-09-26 1,000,000 0x1F 1e-5"),
        ("punctuation", "Wait... what?! (Really) — “quoted” ‘single’".encode()),
        ("italian", "Perché l'àncora è così pesante? Città, università.".encode()),
        ("cjk", "今日はいい天気です。你好，世界！한국어".encode()),
        ("emoji", "Party \U0001F389\U0001F44D\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467".encode()),
        ("rare_unicode", "ᚠᚢ \U0001d49c\U0001d4b7 ༀ ☃".encode()),
        ("special_markup", b"<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Hello<turn|>"),
        ("tool_markup", b"<|tool_call>call:f{a:<|\"|>x y<|\"|>,n:1}<tool_call|><|tool_response>"),
        ("near_special", b"<|turn <turn| <|\"| |> <eos <bos>"),
        ("byte_like", b"<0x41> <0xFF> literal"),
        ("invalid_utf8", b"ok \xff\xfe bad \xe2\x82 trunc \xc3"),
        ("zero_width", "a​b‌c﻿d".encode()),
        ("long_line", ("The quick brown fox jumps over the lazy dog. " * 460).encode()),
    ]
    out = [(name, text, special, False) for name, text in entries]
    out += [
        ("whitespace_runs", b"a \t \t b" + b" " * 33 + b"c\t\t\t\td\r\r\n", special, False),
        ("nbsp_and_ideographic_space", "a b　c d".encode(), special, False),
        ("long_newline_run", b"x" + b"\n" * 40 + b"y", special, False),
        ("spaces_around_newlines", b" \n \n  \n\t\n", special, False),
        ("emoji_sequences", "\U0001F1FA\U0001F1F8 1️⃣ \U0001F44B\U0001F3FF \U0001F3F3️‍\U0001F308".encode(), special, False),
        ("byte_fallback", "\U000F0000\U0010FFFD\U00013000".encode(), special, False),
        ("overlong_and_out_of_range", b"a\xc0\x80b\xe0\x80\x80c\xf4\x90\x80\x80d\xf8\x88\x80\x80\x80e\x80", special, False),
        ("surrogate_bytes", b"x\xed\xa0\x80y", special, False),
        ("chat_controls", b"<bos><|turn>system\n<|think|>You are helpful.<turn|>\n<|turn>user\n<|image|><|audio|><|video|>Hi<turn|>\n<|turn>model\n", special, False),
        ("tool_controls", b"<|tool>declaration:f{description:<|\"|>d<|\"|>}<tool|><|tool_call>call:f{}<tool_call|><|tool_response>response:f{}<tool_response|>", special, False),
        ("eog_and_markers", b"<eos><pad><unk><mask><|image><image|><|audio><audio|></s>", special, False),
        ("special_markup_plain", b"<|turn>user\nHi<turn|><|channel>t<channel|><|tool_response><eos>", False, False),
        ("add_bos", b"Hello", special, True),
    ]
    return out


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def tokenizer_sha256(md: dict) -> str:
    """Hash of the vocabulary, token types and merges as stored."""
    h = hashlib.sha256()
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
        for item in md[key]:
            h.update(struct.pack("<Q", len(item)) + item)
    h.update(struct.pack(f"<{len(md['tokenizer.ggml.token_type'])}i",
                         *md["tokenizer.ggml.token_type"]))
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="library MANIFEST.json recording the model sha256")
    parser.add_argument("--donor-fixture", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    md, header_bytes = read_metadata(args.model)
    manifest = json.loads(args.manifest.read_text())
    record = next(f for f in manifest["files"] if f["file"] == args.model.name)
    vocab = Vocab(md)
    donor = {}
    if args.donor_fixture:
        for e in json.loads(args.donor_fixture.read_text())["entries"]:
            donor[e["name"]] = e

    entries, agree, differ = [], 0, []
    for name, text, parse_special, add_bos in corpus():
        ids = vocab.encode(text, add_bos, parse_special)
        entry = {"name": name, "text_hex": text.hex(),
                 "parse_special": parse_special, "add_bos": add_bos}
        if len(ids) > 64:
            entry.update({"count": len(ids), "ids_sha256": ids_sha256(ids)})
        else:
            entry["ids"] = ids
        if (d := donor.get(name)) is not None and not add_bos and parse_special:
            same = (d.get("llama_cpp_ids") == ids if "llama_cpp_ids" in d else
                    d.get("llama_cpp_count") == len(ids)
                    and d.get("llama_cpp_ids_sha256") == ids_sha256(ids))
            entry["donor_llama_cpp_b11069"] = "agrees" if same else "differs"
            agree += same
            if not same:
                differ.append(name)
        entries.append(entry)

    args.output.write_text(json.dumps({
        "schema": "gufo.gemma4-tokenizer-goldens.v2",
        "generator": "tools/gemma4/tokenizer_goldens.py",
        "semantics": f"llama.cpp {LLAMA_CPP} gemma4 tokenizer, ported to Python",
        "python": platform.python_version(),
        "model": {"repo": record["repo"], "revision": record["revision"],
                  "file": record["file"], "sha256": record["sha256"],
                  "metadata_bytes_read": header_bytes,
                  "tokenizer_sha256": tokenizer_sha256(md)},
        "vocabulary": {"tokens": len(vocab.text),
                       "merges": len(md["tokenizer.ggml.merges"]),
                       "bos": vocab.bos, "eos": vocab.eos,
                       "end_of_generation": sorted(vocab.eog)},
        "donor_cross_check": None if not donor else {
            "source": "gufo-org/gufo PR #426 tests/models/gemma4/fixtures/"
                      "tokenizer_goldens.json at 25ba634f6cc4e172829b41dddc82fa2684c90399",
            "tool": "llama.cpp build 11069 (68d9053a), Gemma 4 31B GGUF",
            "agree": agree, "differ": differ},
        "entries": entries,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries; donor agreement {agree}, "
          f"differences {differ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
