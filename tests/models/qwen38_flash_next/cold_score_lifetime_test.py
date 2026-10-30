"""Extract exact lifetime control bodies; run CPU proxy in isolated processes."""
import argparse
from pathlib import Path
import subprocess


def block(source, marker):
    first = source.index('{', source.index(marker))
    end, depth = first + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[first:end]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--binary')
    a = p.parse_args()
    if a.binary:
        for mode in ('success', 'partial'):
            run = subprocess.run([a.binary, mode], capture_output=True, text=True, timeout=10)
            logs = run.stdout + run.stderr
            if (run.returncode or logs.count('the model stays loaded until the server restarts') != 1
                    or logs.count('event=full_logits_static_claim') != 1):
                raise RuntimeError(f'{mode}: rc={run.returncode}\n{run.stdout}\n{run.stderr}')
        print('PASS: exact Model approval/deleter/Create-prefix lifetime controls; no HIP execution')
        return
    source = (a.source_root / 'src/models/qwen38_flash_next/engine.cpp').read_text()
    native = (a.source_root / 'src/models/qwen38_flash_next/kernels/rocm/executor.cpp').read_text()
    begin = native.index('  std::unique_ptr<Executor> ordinary;', native.index('Executor::Create('))
    end = native.index('  e->model_ = &model;', begin)
    a.output.write_text('auto ModelDeleter = [](Model* model) ' + block(source, '[](Model* model)') + ';\n'
                        + 'std::unique_ptr<Executor> Create(std::unique_ptr<Executor>* diagnostic_failure_owner, bool fail, std::string* error_msg) {\n'
                        + native[begin:end] + '\n e->magic=37; if(fail) return nullptr; return std::move(e); }\n'
                        + 'std::shared_ptr<Model> Load(const Options& options, bool partial, std::string* error_msg) {\n'
                        + ' auto m=std::shared_ptr<Model>(new Model, ModelDeleter);\n'
                        + ' const auto capacity=gufo::models::qwen38_flash_next::DiagnosticLogitCapacity::Plan(3,8,1,options.enable_diagnostic_logits);\n'
                        + ' if(options.enable_diagnostic_logits) ' + block(source, 'if (options.enable_diagnostic_logits) {')
                        + '\n auto e=Create(m->diagnostic_approved_ ? &m->executor_ : nullptr, partial, error_msg);\n'
                        + ' if(!e) return nullptr;\n m->executor_=std::move(e);\n return m;\n}\n')


if __name__ == '__main__':
    main()
