#!/usr/bin/env python3
"""Search four-part covers of Grünbaum's 3-truncated regular octahedron.

Normalization: the regular octahedron has distance 1 between opposite faces,
so it is |x|+|y|+|z| <= sqrt(3)/2.  The three positive axial vertices are
truncated by x<=1/2, y<=1/2, z<=1/2.

Without --run, write target.json.  With --run, invoke partitions_slsqp.py.
"""
from __future__ import annotations
import argparse, itertools, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull


def target_record():
    c = math.sqrt(3.0)/2.0
    A=[]; b=[]
    for signs in itertools.product((-1.0,1.0), repeat=3):
        A.append(signs); b.append(c)
    A += [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    b += [0.5,0.5,0.5]
    A=np.asarray(A); b=np.asarray(b)
    pts=[]
    for triple in itertools.combinations(range(len(A)),3):
        M=A[list(triple)]
        if abs(np.linalg.det(M))<1e-12: continue
        x=np.linalg.solve(M,b[list(triple)])
        if np.max(A@x-b)<=1e-10 and not any(np.linalg.norm(x-y)<=1e-9 for y in pts):
            pts.append(x)
    V=np.asarray(pts)
    h=ConvexHull(V)
    V=V[np.unique(h.simplices)]
    V=np.asarray(sorted(V.tolist()))
    d=float(np.linalg.norm(V[:,None]-V[None,:],axis=2).max())
    return {
        'case':'grunbaum_3_truncated_regular_octahedron',
        'vertices':V.tolist(), 'vertex_count':len(V),
        'target_volume':float(ConvexHull(V).volume), 'target_diameter':d,
        'normalization':'opposite octahedron faces distance 1; x,y,z <= 1/2',
        'analytic':'|x|+|y|+|z| <= sqrt(3)/2, x<=1/2, y<=1/2, z<=1/2',
        'exact_volume':'5/2-sqrt(3)'
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=Path('grunbaum_search'))
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--search-script',type=Path,default=Path(__file__).with_name('partitions_slsqp.py'))
    ap.add_argument('--starts',type=int,default=5000)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--seed',type=int,default=37)
    ap.add_argument('--hours',type=float)
    ap.add_argument('--seconds-per-start',type=float)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--quiet',action='store_true')
    args=ap.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rec=target_record(); target=args.output_dir/'target.json'
    target.write_text(json.dumps(rec,indent=2)+'\n')
    print(f"vertices={rec['vertex_count']} volume={rec['target_volume']:.15f} diameter={rec['target_diameter']:.15f}")
    if not args.run: return 0
    cmd=[sys.executable,str(args.search_script),'--vertices',str(target),'--parts','4',
         '--starts',str(args.starts),'--workers',str(args.workers),'--seed',str(args.seed),
         '--output-dir',str(args.output_dir)]
    if args.hours is not None: cmd += ['--hours',str(args.hours)]
    if args.seconds_per_start is not None: cmd += ['--seconds-per-start',str(args.seconds_per_start)]
    if args.resume: cmd += ['--resume']
    if args.quiet: cmd += ['--quiet']
    return subprocess.run(cmd).returncode

if __name__=='__main__':
    raise SystemExit(main())
