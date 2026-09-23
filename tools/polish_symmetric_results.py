#!/usr/bin/env python3
"""Export tighter verified records from a symmetric_polyhedra.py search directory.

Writes to a NEW destination, leaving the original best/previous-best untouched.
Adds explicit regular-tetrahedral score candidates for tetrahedron/octahedron.
Example: python tools/polish_symmetric_results.py --input runs/symmetric --output runs/symmetric_polished
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import partitions_slsqp as p
from symmetric_polyhedra import NAMES


def polish(source: Path,target_spec: dict):
    record=json.loads((source/'best.json').read_text());t=p.Target.from_vertices(np.array(target_spec['vertices']))
    m,z=p.model_from_record(t,record)
    cfg=p.SearchConfig(bound_screen=False,coarse_maxiter=1000,coarse_ftol=1e-13,
                       polish_maxiter=2000,polish_ftol=3e-15)
    zz,info=p.solve_model(m,cfg,z);new=m.record(zz)
    if new['diameter']<=record['diameter']:
        record.update(new);record['solver']=info;record['post_search_polishing']=True
    name=target_spec['solid']
    if name in ('tetrahedron','octahedron'):
        scores=np.array([v for v in itertools.product((-1,1),repeat=3) if math.prod(v)==1],float)
        ref=p.build_diagram(t,t.normalize(.1*scores));simple=p.Model(t,ref);candidate=simple.record(np.zeros(simple.q))
        if candidate['diameter']<=record['diameter']:
            record.update(candidate);record['source']='explicit_regular_tetrahedral_scores'
            record['solver']={'status':'explicit_construction','converged':True};record['post_search_polishing']=False
    record['target']=t.record()
    record['comparison_note']='Tight post-search local solve and explicit symmetric candidate for tetrahedron/octahedron.'
    if not record['geometry']['valid_numerically']:raise ValueError('Numerical geometry failed')
    return record


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--solids',nargs='+',choices=NAMES,default=list(NAMES));args=ap.parse_args(argv)
    if args.output.exists():ap.error('Choose a new output directory.')
    args.output.mkdir(parents=True);summary={}
    for name in args.solids:
        source=args.input/name;spec=json.loads((args.input/f'target_{name}.json').read_text())
        record=polish(source,spec);dst=args.output/name;dst.mkdir()
        for old,new in [('best.json','search_best.json'),('state.json','search_state.json')]:shutil.copy2(source/old,dst/new)
        (dst/'target.json').write_text(json.dumps(spec,indent=2)+'\n')
        (dst/'model.json').write_text(json.dumps(record,indent=2)+'\n')
        p.write_partition(dst/'partition.txt',np.array(record['points']),record['groups'])
        summary[name]={'diameter':record['diameter'],'points':len(record['points'])}
        print(name,record['diameter'],flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
