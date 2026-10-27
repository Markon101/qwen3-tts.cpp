#!/usr/bin/env python3
"""Run a local command with measurement-only temperature/clock/IO logging."""
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from adreno_telemetry import ROOT, sample


def main(command, env=None):
    if not command:
        raise SystemExit('missing command')
    logdir = ROOT / '.cache/tts-runs'
    logdir.mkdir(parents=True, exist_ok=True)
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}'
    path = logdir / (tag + '.jsonl')
    proc = None
    stop_signal = None
    child_code = None
    launch_error = None
    started = time.monotonic()

    def forward(signum):
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signum)
            except ProcessLookupError:
                pass

    def on_signal(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum
        forward(signum)

    handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.signal(sig, on_signal) for sig in handled}
    try:
        # Open the trace before starting a child so log-creation failure cannot
        # leave a synthesis process orphaned. Inherit stdin/stdout for JSON mode.
        with path.open('x') as trace:
            proc = subprocess.Popen(command, env=env, start_new_session=True)
            if stop_signal is not None:
                forward(stop_signal)
            while proc.poll() is None:
                if stop_signal is not None:
                    break
                row = sample(proc.pid)
                if proc.poll() is not None:
                    break
                trace.write(json.dumps(row) + '\n')
                trace.flush()
                time.sleep(.1)
            try:
                child_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forward(signal.SIGKILL)
                child_code = proc.wait()
    except OSError as error:
        launch_error = str(error)
        print('Profile runner: ' + launch_error, file=sys.stderr)
        child_code = 127
    finally:
        if proc is not None and proc.poll() is None:
            forward(signal.SIGKILL)
            proc.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    status_code = 128 + stop_signal if stop_signal is not None else (
        child_code if child_code is not None and child_code >= 0 else 128 - (child_code or 0))
    status = {'command': command, 'exit_code': child_code, 'exit_status': status_code,
              'stop_reason': 'signal ' + signal.Signals(stop_signal).name if stop_signal else None,
              'launch_error': launch_error, 'wall_s': time.monotonic() - started,
              'trace': str(path), 'temperature_limits': None, 'thermal_policy': 'measurement_only'}
    path.with_suffix('.status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(f'Thermal/clock trace: {path}', file=sys.stderr)
    return status_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
