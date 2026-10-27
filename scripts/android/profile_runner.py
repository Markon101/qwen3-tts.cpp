#!/usr/bin/env python3
"""Run a local profile with measurement-only thermal/clock/IO logging."""
import datetime,json,os,pathlib,subprocess,sys,time
from adreno_telemetry import ROOT,sample
logdir=ROOT/'.cache/tts-runs'
logdir.mkdir(parents=True,exist_ok=True)
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+f'-{os.getpid()}'
path=logdir/(tag+'.jsonl')
command=sys.argv[1:]
if not command:raise SystemExit('missing command')
proc=subprocess.Popen(command)
reason=None;started=time.monotonic()
with path.open('x') as f:
    try:
        while proc.poll() is None:
            row=sample(proc.pid)
            if proc.poll() is not None:break
            f.write(json.dumps(row)+'\n');f.flush()
            time.sleep(.1)
        try:code=proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
    except KeyboardInterrupt:
        proc.terminate();proc.wait();code=130
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
status={'command':command,'exit_code':code,'stop_reason':reason,'wall_s':time.monotonic()-started,'trace':str(path),'temperature_limits':None,'thermal_policy':'measurement_only'}
path.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
print(f'Thermal/clock trace: {path}',file=sys.stderr)
raise SystemExit(1 if reason else code if code>=0 else 128-code)
