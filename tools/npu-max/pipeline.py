#!/usr/bin/env python3
"""Experimental bounded process pipeline; prefix replay preserves decoder history.

Per-prefix re-decoding costs extra compute; latency/quality must be compared.
Each process owns a distinct HTP session; no thread-unsafe shared queue.
"""
import argparse, array, datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time, wave
R=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--size',choices=['0.6b','1.7b'],default='0.6b')
p.add_argument('--models-dir',type=pathlib.Path,default=R.parent.parent/'models')
p.add_argument('--embedding',type=pathlib.Path,required=True)
p.add_argument('--text',default='This is a test of the experimental audio pipeline.')
p.add_argument('--frames',type=int,default=32)
p.add_argument('--chunk-frames',type=int,default=8)
p.add_argument('--decoder',choices=['htp','opencl','cpu'],default='opencl')
p.add_argument('--serial',action='store_true',help='matched producer-then-consumer control')
p.add_argument('--codes-in',type=pathlib.Path,help='decode identical saved codes for chunk/audio parity')
a=p.parse_args()
if not 1<=a.frames<=2048 or not 1<=a.chunk_frames<=100:p.error('invalid bounds')
D=R/'.cache/npu-max/pipeline'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid()))
D.mkdir(parents=True,exist_ok=False)
e=os.environ.copy()
for k in list(e):
    if k.startswith(('QWEN3_TTS_','GGML_HEXAGON_','GGML_OPENCL_')):e.pop(k,None)
e.update(QWEN3_TTS_TRANSFORMER_BACKEND='hexagon',QWEN3_TTS_VALIDATE_LOGITS='1',QWEN3_TTS_VALIDATE_AUDIO='1')
exe=R/'.cache/npu-max/app-build/tools/npu/npu-max-pipeline-stage'
base=['sh',str(R/'tools/npu-max/native-env.sh'),str(exe)]
cmd=base+['generate',str(a.models_dir/('qwen3-tts-'+a.size+'-q8_0.gguf')),str(a.embedding.resolve()),a.text,str(D/'codes.i32'),str(a.frames)]
de=e.copy();de.update(QWEN3_TTS_DECODER_BACKEND='hexagon' if a.decoder=='htp' else a.decoder,
    QWEN3_TTS_MAX_VOCODER='1' if a.decoder=='htp' else '0',QWEN3_TTS_MAX_STRICT='1' if a.decoder=='htp' else '0')
children=[]
def forward(signum,_frame):
    for proc in children:
        if proc.poll() is None:
            try:os.killpg(proc.pid,signum)
            except ProcessLookupError:pass
    raise KeyboardInterrupt
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,forward)
start=time.monotonic();first_audio=None;codes_source=None
try:
    with (D/'generator.log').open('x') as gl,(D/'decoder.log').open('x') as dl:
        if a.codes_in:
            codes_source=a.codes_in.open('rb');source=codes_source
        else:
            generator=subprocess.Popen(cmd,env=e,stdout=subprocess.PIPE,stderr=gl,start_new_session=True);children.append(generator)
            if a.serial:
                with (D/'stream.i32').open('xb') as f:
                    while True:
                        data=generator.stdout.read(4096)
                        if not data:break
                        f.write(data)
                if generator.wait():raise RuntimeError('generation failed')
                codes_source=(D/'stream.i32').open('rb');source=codes_source
            else:source=generator.stdout
        decoder=subprocess.Popen(base+['decode',str(a.models_dir/'qwen3-tts-tokenizer-f16-f32.gguf'),str(a.chunk_frames)],
            env=de,stdin=source,stdout=subprocess.PIPE,stderr=dl,start_new_session=True);children.append(decoder)
        if not a.codes_in and not a.serial:generator.stdout.close()
        count=0
        with (D/'audio.f32').open('xb') as raw,wave.open(str(D/'audio.wav'),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000)
            while True:
                data=decoder.stdout.read(7680)
                if not data:break
                if first_audio is None:first_audio=time.monotonic()-start
                raw.write(data);f=array.array('f');f.frombytes(data);count+=len(f)
                if any(not (-float('inf')<x<float('inf')) for x in f):raise RuntimeError('nonfinite audio')
                pcm=array.array('h',(max(-32768,min(32767,round(x*32767))) for x in f));wav.writeframes(pcm.tobytes())
        for proc in children:
            if proc.wait(timeout=30):raise RuntimeError('stage failed; inspect '+str(D))
    codefile = a.codes_in if a.codes_in else D/'codes.i32'
    rawcodes = codefile.read_bytes()
    if not rawcodes or len(rawcodes)%64 or count != len(rawcodes)//64*1920:
        raise RuntimeError('pipeline code/audio shape mismatch')
    report={'codes_sha256':hashlib.sha256(rawcodes).hexdigest(),'decoder':a.decoder,'serial':a.serial,'chunk_frames':a.chunk_frames,'first_audio_s':first_audio,
            'wall_s':time.monotonic()-start,'samples':count,'prefix_replay':True,'quality_gate':'compare same codes at chunk100 before use',
            'scope':'correctness/latency experiment; cold loading and pipe buffering included'}
    (D/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));print(D)
finally:
    if codes_source:codes_source.close()
    for proc in children:
        if proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
