#!/usr/bin/env python3
"""Freeze editing SQ scales from three complete 1/2/3-reference observations.

The capture directory must contain calibration-requests.json and
calibration-runs/<request name>/calibration.json. Captures contain per-channel
activation_max and original weight_max for all 392 production projections.
This is an offline development tool; it is never packaged or run by the App.
"""
import argparse
from pathlib import Path
import hashlib,json
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--captures',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.captures.resolve()
DEST=args.output.resolve()
DEST.mkdir(parents=True,exist_ok=True)
requests=json.loads((OUT/'calibration-requests.json').read_text())
paths=[OUT/'calibration-runs'/r['name']/'calibration.json' for r in requests]
records=[json.loads(p.read_text()) for p in paths]
assert len(records)==3 and len({r['source_sha256'] for r in records})==1
assert [len(r['reference_sha256']) for r in requests]==[1,2,3]
assert all(set(r['layers'])==set(records[0]['layers']) for r in records)
meta={'schema':'qwen21-sq-convrot-profile-v1','source_sha256':records[0]['source_sha256'],
      'method':'SmoothQuant per-input-channel alpha=0.5, scale clip [1/16,16], inverse rounded FP16; paired W uses exact reciprocal of stored inverse',
      'calibration_route':'original BF16-derived DiT W8A8 + unchanged frozen ANE; image.edit; 1/2/3 resized-512 references; DBCache off; all 40 steps; compiler disabled only during capture',
      'calibration_files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
      'calibration_requests':requests,
      'evaluation_prompts_excluded':['fox-blue','text-color','objects-color','multi-reference-composition'],
      'operation':'image.edit'}
meta['identity']=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
payload=bytearray();entries=[];stats={}
for key in sorted(records[0]['layers']):
 rows=[r['layers'][key] for r in records]
 a=np.max([r['activation_max'] for r in rows],axis=0).astype(np.float32)
 w=np.asarray(rows[0]['weight_max'],np.float32)
 assert all(np.array_equal(w,np.asarray(r['weight_max'],np.float32)) for r in rows)
 assert all(r['calls']>0 for r in rows) and np.isfinite(a).all() and np.isfinite(w).all()
 raw=np.sqrt(np.maximum(a,1e-8)/np.maximum(w,1e-8))
 scale=np.clip(raw,1/16,16);scale[(a==0)|(w==0)]=1
 inverse=np.asarray(1/scale,dtype='<f2')
 entries.append({'key':key,'offset':len(payload)//2,'channels':len(inverse)})
 payload.extend(inverse.tobytes())
 stats[key]={'calls':sum(r['calls'] for r in rows),'rows':sum(r['rows'] for r in rows),
             'channels':len(inverse),'clipped_channels':int(((raw<1/16)|(raw>16)).sum())}
assert len(entries)==392 and len(payload)==1925440*2
meta['payload_sha256']=hashlib.sha256(payload).hexdigest();meta['entries']=entries
(DEST/'sq_convrot_edit_v1.f16le').write_bytes(payload)
(DEST/'sq_convrot_edit_v1.json').write_text(json.dumps(meta,indent=2)+'\n')
(DEST/'calibration-summary.json').write_text(json.dumps({'identity':meta['identity'],'payload_sha256':meta['payload_sha256'],'projection_count':len(entries),'channels':len(payload)//2,'layers':stats},indent=2)+'\n')
print('EDIT PROFILE',meta['identity'],'392 projections',len(payload)//2,'channels')
