#!/usr/bin/env python3
import json,pathlib,statistics
p=pathlib.Path(__file__).resolve().parent
runs=json.loads((p/'results.json').read_text())['runs']
def totals(names):
 return [q['t_total_ms'] for name in names for q in runs[name]['requests'][1:]]
a=totals(['unlimited-gpu-a','unlimited-gpu-b']);b=totals(['unlimited-fast-prime2'])
assert len(a)==4 and len(b)==2
for name in ['unlimited-gpu-a','unlimited-fast-prime2','unlimited-gpu-b']:
 assert runs[name]['all_foreground'] and runs[name]['temperature_limits'] is None
 assert all(q['duration_s']==5.12 and q['frame_cap']==64 and q['pcm_stats']['nonzero']>0 for q in runs[name]['requests'][1:])
print('GPU median ms:',statistics.median(a))
print('Hybrid median ms:',statistics.median(b))
print('Time reduction percent:',100*(1-statistics.median(b)/statistics.median(a)))
mask=json.loads((p/'microbench-results.json').read_text())['mask-probe-1']['rows']
assert sum(r['variant']=='fixed' for r in mask)==15
assert all(r['wrong_values']==0 and r['guard_corruptions']==0 for r in mask if r['variant']=='fixed')
print('Recorded fixed-mask cases: 15/15 correct')
