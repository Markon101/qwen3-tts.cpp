#!/usr/bin/env python3
"""Validate native CLI vocoder-window parsing without loading any model."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ['sh', str(ROOT / 'tools/npu-max/native-env.sh'),
           str(ROOT / '.cache/npu-max/app-build/qwen3-tts-cli')]


def main():
    accepted = {'17', '100', '128', '200', '256'}
    cases = ['17', '100', '128', '200', '256', '0', '-1', '16', '257', '1024',
             '1.5', '200junk', '4294967396', '2147483648',
             '999999999999999999999999', 'nan', 'inf', '', None]
    results = []
    for value in cases:
        args = ['--vocoder-chunk-frames'] + ([] if value is None else [value])
        run = subprocess.run(COMMAND + args, capture_output=True, text=True, timeout=15)
        if value in accepted:
            passed = 'model directory required' in run.stderr
        elif value is None:
            passed = 'Missing value for --vocoder-chunk-frames' in run.stderr
        else:
            passed = ('vocoder-chunk-frames' in run.stderr and
                      'model directory required' not in run.stderr)
        passed = passed and run.returncode == 1 and 'Loading models' not in run.stderr
        results.append({'value': value, 'accepted_value': value in accepted,
                        'pass': passed, 'exit_code': run.returncode,
                        'message': run.stderr.splitlines()[0] if run.stderr else ''})
    report = {'tests': len(results), 'passed': sum(r['pass'] for r in results),
              'hardware_execution': False, 'results': results}
    print(json.dumps(report, indent=2))
    return 0 if all(r['pass'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
