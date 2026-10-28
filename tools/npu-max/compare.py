#!/usr/bin/env python3
"""Compare private saved embeddings or raw F32 waveform under fixed codes.

Autoregressive waveform comparison is only meaningful when input codes match.
"""
import argparse,array,json,math,pathlib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline',type=pathlib.Path);p.add_argument('candidate',type=pathlib.Path)
p.add_argument('--embedding',action='store_true')
p.add_argument('--min-cosine',type=float,default=.999)
p.add_argument('--max-nrmse',type=float,default=.05)
a=p.parse_args()
def read(path):
 if path.suffix.lower() == '.wav':
  import wave
  with wave.open(str(path), 'rb') as w:
   data = w.readframes(w.getnframes())
   x = array.array('h'); x.frombytes(data)
   return array.array('f', (v / 32768.0 for v in x))
 data=path.read_bytes()
 if a.embedding:
  import struct
  if len(data)<4:raise ValueError('missing embedding header')
  n=struct.unpack('<I',data[:4])[0]
  if len(data)!=4+4*n:raise ValueError('embedding length mismatch')
  data=data[4:]
 x=array.array('f');x.frombytes(data);return x
x,y=read(a.baseline),read(a.candidate)
if not x or len(x)!=len(y):raise SystemExit('different or empty shapes')
bad=sum(not math.isfinite(v) for v in x)+sum(not math.isfinite(v) for v in y)
if bad:raise SystemExit('nonfinite outputs: '+str(bad))
ex=sum(v*v for v in x);ey=sum(v*v for v in y);se=sum((u-v)**2 for u,v in zip(x,y))
cos=sum(u*v for u,v in zip(x,y))/math.sqrt(ex*ey) if ex*ey else float(x==y)
nrmse=math.sqrt(se/max(ex,1e-30))
r={'count':len(x),'cosine':cos,'nrmse':nrmse,'max_abs':max(abs(u-v) for u,v in zip(x,y)),
   'rms_baseline':math.sqrt(ex/len(x)),'rms_candidate':math.sqrt(ey/len(y)),'bit_identical':a.baseline.read_bytes()==a.candidate.read_bytes(),
   'gate':cos>=a.min_cosine and nrmse<=a.max_nrmse,'quality_scope':'numerical only; listening and transcription remain separate'}
print(json.dumps(r));raise SystemExit(0 if r['gate'] else 1)
