#!/usr/bin/env python3
"""Foreground-only experimental comparison; logs temperatures without limits.

All assets/results stay in .cache. The CPU variants target S25 Ultra cores 6-7.
"""
import argparse,os,sys,pathlib,json,time,subprocess,wave,array,math,hashlib,datetime,re
R=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--models-dir',type=pathlib.Path,default=pathlib.Path(os.environ.get('QWEN_MODELS_DIR',str(R/'models'))))
parser.add_argument('--embedding',type=pathlib.Path,required=True)
parser.add_argument('--model-size',choices=['0.6b','1.7b'],default='0.6b')
parser.add_argument('--frames',type=int,default=64)
parser.add_argument('--requests',type=int,default=2)
parser.add_argument('--cases',nargs='+',choices=['cpu-a','htp2','htp4','htp6','htp4-poll','gpu','cpu-b'],default=['cpu-a','htp4','htp4-poll','cpu-b'])
parser.add_argument('--tag',default='npu-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'))
args=parser.parse_args()
if args.frames<1 or args.requests<1 or not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):parser.error('invalid frames, requests or tag')
models=args.models_dir.resolve();embedding=args.embedding.resolve()
if not embedding.is_file():parser.error('embedding file missing')
sys.path.insert(0,str(R/'scripts/android'))
from adreno_telemetry import sample
from code_provenance import associate_code_dumps
B=R/'.cache/npu/build/src';P=R/'.cache/tts-runs';D=R/'.cache/npu'/args.tag;D.mkdir(parents=True,exist_ok=False);P.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for k in list(env):
 if k.startswith(('GGML_HEXAGON_','GGML_OPENCL_','QWEN3_TTS_','GGML_PROFILE')):env.pop(k,None)
env.update(LD_LIBRARY_PATH=f'{B}:{B}/ggml-hexagon:{B}/ggml-opencl:/system/lib64:/vendor/lib64',LD_PRELOAD='',ADSP_LIBRARY_PATH=str(B/'ggml-hexagon'),OCL_ICD_VENDORS='/vendor/lib64/libOpenCL.so',GGML_HEXAGON_HOSTBUF='0',GGML_HEXAGON_PROFILE='0',QWEN3_TTS_DECODER_BACKEND='opencl',QWEN3_TTS_SEED='20261026',QWEN3_TTS_VALIDATE_AUDIO='1',QWEN3_TTS_VALIDATE_LOGITS='1',GGML_OPENCL_BINARY_CACHE_DIR=str(R/'.cache/adreno830-opencl'))
command=[str(R/'.cache/npu/app-build/qwen3-tts-cli'),'-m',str(models/('qwen3-tts-'+args.model_size+'-q8_0.gguf')),'--tokenizer',str(models/'qwen3-tts-tokenizer-f16-f32.gguf'),'--temperature','.9','--sub-temperature','.9','--threads','2','--server']
cases={'cpu-a':('cpu',2),'cpu-b':('cpu',2),'htp2':('hexagon',2),'htp4':('hexagon',4),'htp6':('hexagon',6),'htp4-poll':('hexagon',4),'gpu':('opencl',4)}
variants=[]
seen={}
for name in args.cases:
 seen[name]=seen.get(name,0)+1
 label=name if seen[name]==1 else name+'-'+str(seen[name])
 variants.append((label,*cases[name]))
rows=[]
failed_requests=False
for name,backend,nhvx in variants:
 tag=args.tag+'-'+name
 if (P/(tag+'.status.json')).exists():raise RuntimeError('already exists '+tag)
 e=env.copy();e['QWEN3_TTS_TRANSFORMER_BACKEND']=backend;e['GGML_HEXAGON_NHVX']=str(nhvx)
 if 'poll' in name:e['GGML_HEXAGON_OPPOLL']='1'
 if backend=='opencl' and args.model_size=='0.6b':e['GGML_OPENCL_Q8_SMALL2']='1'
 codes=D/(name+'-codes');codes.mkdir(exist_ok=True);e['QWEN3_TTS_CODES_DIR']=str(codes)
 cmd=command.copy();cmd[cmd.index('--threads')+1]='4' if backend=='opencl' else '2'
 if backend=='cpu':cmd=['taskset','-c','6,7']+cmd
 requests=[]
 for i,frames in enumerate([min(8,args.frames)]+[args.frames]*args.requests):requests.append({'text':'The little fox wandered through the quiet forest, following the silver river toward the distant mountains. Along the way, it listened to the birds and watched sunlight dancing through the leaves.','embedding_in':str(embedding),'language':'english','max_tokens':frames,'output':str(D/(name+f'-{i}.wav'))})
 reqfile=D/(name+'.jsonl');reqfile.write_text(''.join(json.dumps(x)+'\n' for x in requests))
 started=time.monotonic();before=sample()
 with reqfile.open() as inp,(P/(tag+'.stdout.log')).open('x') as out,(P/(tag+'.stderr.log')).open('x') as err,(P/(tag+'.telemetry.jsonl')).open('x') as tf:
  p=subprocess.Popen(cmd,stdin=inp,stdout=out,stderr=err,env=e)
  try:
   while p.poll() is None:
    tf.write(json.dumps(sample(p.pid))+'\n');tf.flush()
    if time.monotonic()-started>240:
     p.terminate()
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:p.kill();p.wait()
     break
    time.sleep(.1)
   code=p.wait()
  finally:
   if p.poll() is None:p.kill();p.wait()
 status={'command':cmd,'environment':{k:v for k,v in e.items() if k.startswith(('GGML_HEXAGON','GGML_OPENCL','QWEN3_TTS'))},'exit_code':code,'wall_s':time.monotonic()-started,'thermal_policy':'measurement_only','temperature_limits':None,'before':before,'after':sample()}
 (P/(tag+'.status.json')).write_text(json.dumps(status,indent=2)+'\n')
 if code:raise RuntimeError('failed '+tag)
 responses=[json.loads(line) for line in (P/(tag+'.stdout.log')).read_text().splitlines() if line.startswith('{')]
 assert len(responses)==args.requests+1
 code_mapping,ambiguous=associate_code_dumps(responses,[p.name for p in codes.glob('codes-*.i32')])
 if ambiguous:failed_requests=True
 for i,r in enumerate(responses):
  r.update(codes_dump_file=code_mapping[i],codes_mapping_ambiguous=ambiguous)
  if not r.get('success'):
   failed_requests=True;r.update(variant=name,request=i,fixed_frame_match=False,raw_logits_guard=True)
   rows.append(r);continue
  with wave.open(r['output'],'rb') as f:a=array.array('h',f.readframes(f.getnframes()))
  assert any(a)
  r.update(variant=name,request=i,raw_logits_guard=True,frames_actual=round(r['duration_s']/.08),fixed_frame_match=abs(r['duration_s']-requests[i]['max_tokens']*.08)<.001,nonzero=sum(v!=0 for v in a),peak=max(abs(v) for v in a),rms=math.sqrt(sum((v/32768)**2 for v in a)/len(a)),wav_sha256=hashlib.sha256(pathlib.Path(r['output']).read_bytes()).hexdigest())
  rows.append(r)
 (D/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,[(r['frames_actual'],r['t_generate_ms'],r['t_decode_ms'],r['t_total_ms']) if r.get('success') else ('FAILED',r.get('error')) for r in responses],flush=True)
 if not all(r['fixed_frame_match'] for r in responses):print('Failed request or early EOS; do not compare total times as fixed work.',flush=True)

raise SystemExit(1 if failed_requests else 0)
