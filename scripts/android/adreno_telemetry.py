"""Read-only device sampling and bounded process monitoring."""
import datetime,json,os,pathlib,subprocess,time
SCRIPT_DIR=pathlib.Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent.parent
P=ROOT/'.cache/tts-runs'
REPO=ROOT
BASE=REPO/'ggml/build-opencl/src'
THERMAL={}
for d in pathlib.Path('/sys/class/thermal').glob('thermal_zone*'):
    try:
        name=(d/'type').read_text().strip()
        if name.startswith(('gpuss-','cpuss-','cpu-','nsphvx-','nsphmx-')) or name in ('battery','ddr'): THERMAL[name]=d/'temp'
    except OSError: pass
KGSL=pathlib.Path('/sys/class/kgsl/kgsl-3d0')
COUNTERS={f'gpu/{name}':KGSL/name for name in ['devfreq/cur_freq','devfreq/max_freq','devfreq/min_freq','devfreq/gpu_load','devfreq/target_freq','gpu_busy_percentage','gpubusy','thermal_pwrlevel','throttling','temp','clock_mhz']}
for d in pathlib.Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
    for name in ['scaling_cur_freq','scaling_max_freq','scaling_min_freq','affected_cpus']:
        COUNTERS[f'cpu/{d.name}/{name}']=d/name

def read(path):
    try: return path.read_text().strip()
    except OSError: return None

def sample(pid=None):
    temps={}
    for name,path in THERMAL.items():
        value=read(path)
        if value is not None:
            try: temps[name]=int(value)/1000
            except ValueError: pass
    s={'monotonic_s':time.monotonic(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'temperatures_c':temps,'counters_raw':{k:read(p) for k,p in COUNTERS.items()}}
    if pid is not None:
        s['cgroup']=read(pathlib.Path(f'/proc/{pid}/cgroup'))
        status=read(pathlib.Path(f'/proc/{pid}/status'))
        s['cpus_allowed_list']=next((line.split(':',1)[1].strip() for line in (status or '').splitlines() if line.startswith('Cpus_allowed_list:')),None)
        s['process_io_raw']=read(pathlib.Path(f'/proc/{pid}/io'))
        s['process_stat_raw']=read(pathlib.Path(f'/proc/{pid}/stat'))
        s['rss_raw']=next((line for line in (status or '').splitlines() if line.startswith('VmRSS:')),None)
    return s

def violation(s):
    # Measurement-only protocol requested by the user. Android continues to
    # manage its own thermal/frequency behavior; this logger imposes no gate.
    return None

def env():
    e=os.environ.copy();e['OCL_ICD_VENDORS']='/vendor/lib64/libOpenCL.so'
    e['LD_LIBRARY_PATH']=f'{BASE}:{BASE}/ggml-opencl:{REPO}/build-opencl:/system/lib64:/vendor/lib64:'+e.get('LD_LIBRARY_PATH','')
    return e

def monitored(command,tag,extra_env=None,timeout=120):
    P.mkdir(parents=True,exist_ok=True)
    if (P/f'{tag}.status.json').exists() or (P/f'{tag}.stdout.log').exists(): raise RuntimeError('result already exists')
    before=sample(); reason=violation(before)
    if reason: raise RuntimeError(reason)
    e=env();e.update(extra_env or {})
    loaded=set(); reason=None
    with (P/f'{tag}.stdout.log').open('w') as out,(P/f'{tag}.stderr.log').open('w') as err,(P/f'{tag}.telemetry.jsonl').open('w') as tf:
        p=subprocess.Popen(command,cwd=ROOT,env=e,stdout=out,stderr=err)
        started=time.monotonic()
        try:
            while p.poll() is None:
                s=sample(p.pid)
                if p.poll() is not None: break
                tf.write(json.dumps(s)+'\n');tf.flush()
                maps=read(pathlib.Path(f'/proc/{p.pid}/maps')) or ''
                loaded.update(line.split()[-1] for line in maps.splitlines() if 'libggml' in line and '/' in line)
                reason=violation(s)
                if not reason and time.monotonic()-started>timeout: reason='timeout'
                if reason:
                    # Android can move an exiting process to the root cpuset.
                    # Preserve the observation, but accept a natural successful exit.
                    if reason=='foreground cpuset lost':
                        try:
                            if p.wait(timeout=.03)==0:
                                reason=None
                                break
                        except subprocess.TimeoutExpired:
                            pass
                    p.terminate();break
                time.sleep(.1)
            try: code=p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill();code=p.wait()
        finally:
            if p.poll() is None: p.kill();p.wait()
    result={'thermal_policy':'measurement_only','temperature_limits':None,'command':list(map(str,command)),'exit_code':code,'reason':reason,'wall_s':time.monotonic()-started,'loaded_libraries':sorted(loaded),'before':before,'after':sample()}
    (P/f'{tag}.status.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import sys
    r=monitored(sys.argv[2:],sys.argv[1]);print(json.dumps(r,indent=2));raise SystemExit(0 if r['exit_code']==0 and not r['reason'] else 1)
