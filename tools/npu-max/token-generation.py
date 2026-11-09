#!/usr/bin/env python3
"""Serial predictor controls and foreground-only A/B/A timing on resident models.

No background override. Checks retain all-head logits separately from timing.
Raw artifacts stay in the ignored output directory; no private voice is needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/android'))
from adreno_telemetry import sample


def top_app(cgroup):
    return any(line.split(':', 2)[1:] == ['cpuset', '/top-app']
               for line in (cgroup or '').splitlines())


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def run(args, size, backend):
    if not top_app(Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Termux is not in cpuset /top-app; no run started')
    directory = args.output / (size + '-' + backend)
    directory.mkdir(parents=True, exist_ok=False)
    executable = ROOT / '.cache/npu-max/app-build/tools/npu/npu-max-test-token-generation'
    model = args.models_dir / ('qwen3-tts-' + size + '-q8_0.gguf')
    base = ROOT / '.cache/npu-max/build/src'
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('QWEN3_TTS_', 'GGML_HEXAGON_', 'GGML_OPENCL_', 'GGML_PROFILE'))}
    env.update(GGML_HEXAGON_PROFILE='0', QWEN_TOKEN_GEN_CANDIDATE=args.candidate)
    if args.candidate.startswith('trace-'):
        env['QWEN_TOKEN_GEN_TRACE_PATH'] = str(directory / 'placement.jsonl')
        if backend == 'hexagon':
            env['QWEN3_TTS_MAX_STRICT'] = '1'
    if args.mode == 'check':
        env['QWEN3_TTS_EXECUTION_TRACE'] = str(directory / 'placement.jsonl')
        if backend == 'hexagon':
            env['QWEN3_TTS_MAX_STRICT'] = '1'
    command = ['sh', str(ROOT / 'tools/npu-max/native-env.sh'), str(executable),
               args.mode, str(model), backend, str(directory), str(args.calls)]
    paths = [model, executable, ROOT / 'src/tts_transformer.cpp', ROOT / 'src/gguf_loader.cpp', Path(__file__).resolve(),
             ROOT / 'tools/npu-max/test-token-generation.cpp',
             *[base / name for name in ('libggml.so', 'libggml-base.so', 'libggml-cpu.so',
               'ggml-opencl/libggml-opencl.so', 'ggml-hexagon/libggml-hexagon.so',
               'ggml-hexagon/libggml-htp-v79.so')]]
    manifest = {'command': command, 'mode': args.mode, 'size': size, 'backend': backend,
                'candidate': args.candidate,
                'app_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'ggml_head': subprocess.check_output(['git', '-C', 'ggml', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'identities': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): identity(p) for p in paths},
                'initial_device': sample(os.getpid()), 'thermal_policy': 'measurement_only',
                'timing_scope': ('32 complete generated frames, zero speaker vector; excludes vocoder and loading' if args.mode == 'generate' else '15 dependent predictor heads on fixed public input; excludes talker, vocoder and loading')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not top_app(Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('foreground lost during provenance collection')
    loaded = set()
    reason = None
    with (directory / 'stdout.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err, \
         (directory / 'telemetry.jsonl').open('w') as telemetry:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        start = time.monotonic()
        try:
            while child.poll() is None:
                observed = sample(child.pid)
                telemetry.write(json.dumps(observed) + '\n'); telemetry.flush()
                try:
                    loaded.update(line.split()[-1] for line in Path(f'/proc/{child.pid}/maps').read_text().splitlines()
                                  if 'libggml' in line and '/' in line)
                except OSError:
                    pass
                if child.poll() is not None:
                    break
                # A zombie has finished all work; Android has already removed
                # its scheduling assignment. Keep the sample without treating
                # this terminal state as an active foreground violation.
                stat = observed.get('process_stat_raw') or ''
                state = stat.rsplit(')', 1)[-1].split()
                if state and state[0] in ('Z', 'X'):
                    child.wait(timeout=5)
                    break
                if not top_app(observed.get('cgroup')) or not top_app(Path('/proc/self/cgroup').read_text()):
                    # Android may move an exiting process before waitpid sees it.
                    try:
                        child.wait(timeout=.03)
                        break
                    except subprocess.TimeoutExpired:
                        reason = 'foreground lost or unreadable; run invalid'
                        child.terminate()
                        break
                if time.monotonic() - start > 240:
                    reason = '240-second timeout'
                    child.terminate()
                    break
                time.sleep(.1)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); code = child.wait()
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
    rows = [json.loads(line) for line in (directory / 'stdout.jsonl').read_text().splitlines() if line.startswith('{')]
    valid = code == 0 and reason is None and bool(rows) and rows[-1].get('event') == 'complete' and rows[-1].get('pass') is True
    result = {'mode': args.mode, 'size': size, 'backend': backend, 'valid': valid,
              'candidate': args.candidate, 'exit_code': code, 'reason': reason, 'loaded_libraries': sorted(loaded)}
    if args.mode != 'check':
        times = [row for row in rows if row.get('event') == 'timing']
        blocks = [[r['elapsed_ms'] for r in times if r['block'] == b] for b in range(3)]
        result.update(block_medians_ms=[statistics.median(b) if b else None for b in blocks],
                      calls_per_block=[len(b) for b in blocks],
                      all_sampled_codes_identical=len({tuple(r['codes']) for r in times}) == 1)
        result['valid'] = valid and all(len(b) == args.calls for b in blocks) and result['all_sampled_codes_identical']
    else:
        heads = [r for r in rows if r.get('event') == 'head']
        cases = [r for r in rows if r.get('event') == 'case']
        result.update(head_comparisons=len(heads), cases=len(cases),
                      nonexact_head_comparisons=sum(not r['bit_exact'] for r in heads),
                      maximum_nrmse=max((r['nrmse'] for r in heads), default=None),
                      maximum_absolute_error=max((r['max_abs'] for r in heads), default=None))
        result['valid'] = valid and len(heads) == 180 and len(cases) == 6
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not result['valid']:
        raise RuntimeError('invalid run; inspect ' + str(directory))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('swiglu', 'strided-kv', 'both', 'trace-summary', 'trace-off'), default='both')
    parser.add_argument('--mode', choices=('check', 'bench', 'generate'), default='check')
    parser.add_argument('--sizes', nargs='+', choices=('0.6b', '1.7b'), default=['0.6b', '1.7b'])
    parser.add_argument('--backends', nargs='+', choices=('cpu', 'opencl', 'hexagon'), default=['cpu', 'hexagon', 'opencl'])
    parser.add_argument('--models-dir', type=Path, default=ROOT.parent.parent / 'models')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calls', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.calls <= 128:
        parser.error('--calls must be 1..128')
    args.output = args.output.resolve()
    args.models_dir = args.models_dir.resolve()
    results = []
    for size in args.sizes:
        for backend in args.backends:
            results.append(run(args, size, backend))
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
