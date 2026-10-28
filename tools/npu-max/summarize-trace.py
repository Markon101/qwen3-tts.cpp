#!/usr/bin/env python3
"""Summarize observed graph placement/timing; host preprocessing remains separate."""
import argparse,collections,json,pathlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=pathlib.Path);a=p.parse_args()
components=collections.defaultdict(lambda:{'placements':0,'htp_compute_nodes_max':0,'cpu_compute_nodes_max':0,'other_compute_nodes_max':0,'compute_calls':0,'compute_ms':0,'fallback_ops':[]})
for line in a.trace.read_text().splitlines():
 r=json.loads(line);c=components[r['component']]
 if r['event']=='placement':
  c['placements']+=1
  for key in ('htp_compute_nodes','cpu_compute_nodes','other_compute_nodes'):c[key+'_max']=max(c[key+'_max'],r[key])
  c['fallback_ops']=sorted(set(c['fallback_ops'])|{n['op'] for n in r['nodes'] if not n['metadata'] and not n['backend'].startswith('HTP')})
 elif r['event']=='compute':c['compute_calls']+=1;c['compute_ms']+=r['elapsed_ms']
print(json.dumps({'scope':'scheduled GGML compute only; elapsed times include diagnostic overhead','components':dict(components)},indent=2))
