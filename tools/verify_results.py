#!/usr/bin/env python3
"""Recheck primary saved covers using ONLY final data and exact rational geometry.

Default: explicit 1e-12 facet padding. --raw tests the unmodified decimal hulls.
--full additionally checks exact 3D inclusion-exclusion (15 intersections/case).
Numerical target vertices are not used: analytic target.json halfspaces are used.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import exact_cover_check as ec

def analytic_target(spec,inner=False):
    if spec.get('family')=='symmetric_comparison':
        from symmetric_polyhedra import rational_outer_halfspaces, rational_inner_halfspaces
        maker=rational_inner_halfspaces if inner else rational_outer_halfspaces
        enclosed=maker(spec['solid'],spec['normalization'],spec.get('enclosure_digits',40))
        hs=[ec.plane(n+[h]) for n,h in enclosed]
        vs=ec.poly_vertices(hs)
        if not ec.full_dimension(vs):raise ValueError('Degenerate analytic target.')
        return hs,vs,None
    if spec.get('base_pairwise_offset')!='1/sqrt(2)':raise ValueError('Unsupported analytic target.')
    scale=10**30;m=math.isqrt(scale*scale//2)
    t=F(m if inner else m+1,scale)
    if inner:
        if not 2*t*t<1:raise ArithmeticError('Bad inner sqrt bound.')
    else:
        if not 2*t*t>1:raise ArithmeticError('Bad outer sqrt bound.')
    hs=[]
    for i,j in ((0,1),(0,2),(1,2)):
        for a,b in it.product((-1,1),repeat=2):
            n=[0,0,0];n[i],n[j]=a,b;hs.append(ec.plane(n+[t]))
    hs.extend(ec.plane(h['normal']+[F(h['rhs'])]) for h in spec['axial_halfspaces'])
    vs=ec.poly_vertices(hs)
    if not ec.full_dimension(vs):raise ValueError('Degenerate analytic target.')
    return hs,vs,t

def verify_entry(entry,raw=False,full=False,overlaps=False):
    p=ROOT/entry['partition'];spec=json.loads((ROOT/entry['target']).read_text())
    ps,gs,stored=ec.load_partition(p);hs,vs,t=analytic_target(spec)
    eps=F(0) if raw else F(entry['verification_padding'])
    start=time.monotonic()
    out=ec.verify(ps,gs,hs,vs,padding=eps,bound=F(entry['certified_bound']),full=full,overlaps=overlaps)
    out.update(result_id=entry['id'],input=entry['partition'],input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
               target_definition=entry['target'],target_definition_sha256=hashlib.sha256((ROOT/entry['target']).read_bytes()).hexdigest(),
               checked_target_halfspaces_exact=[[str(x) for x in h] for h in hs],
               rational_outer_pairwise_offset=None if t is None else str(t),covers_ideal_target=True if out['covers_checked_target'] else None)
    if out['covers_checked_target'] is False:
        ih,iv,_=analytic_target(spec,True)
        cells=[[tuple(map(int,h)) for h in cell] for cell in out['checked_cell_halfspaces_exact']]
        inner=ec.boundary_cover(ih,iv,cells)
        if not inner['covered']:
            out['covers_ideal_target']=False;out['ideal_target_counterexample']=inner['witness']
    out['elapsed_seconds']=time.monotonic()-start
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only',nargs='+',help='Subset of result IDs in results.json.')
    ap.add_argument('--raw',action='store_true')
    ap.add_argument('--full',action='store_true')
    ap.add_argument('--overlaps',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'runs/verification')
    args=ap.parse_args();entries=json.loads((ROOT/'results.json').read_text())['results']
    known={x['id'] for x in entries}
    if args.only and not set(args.only)<=known:ap.error('Unknown result ID.')
    args.output_dir.mkdir(parents=True,exist_ok=True);success=True;summary=[]
    for entry in entries:
        if args.only and entry['id'] not in args.only:continue
        rec=verify_entry(entry,args.raw,args.full,args.overlaps)
        status=rec['covers_ideal_target'] is True and rec['diameter_bound_proved'] is True
        success &= status
        name=entry['id']+('_raw' if args.raw else '_padded')+'.json'
        (args.output_dir/name).write_text(json.dumps(rec,indent=2)+'\n')
        summary.append({'id':entry['id'],'coverage_ideal':rec['covers_ideal_target'],
                        'diameter_upper':rec['checked_max_diameter_upper'],'bound_proved':rec['diameter_bound_proved'],
                        'padding':rec['padding'],'full_inclusion_exclusion':bool(args.full),
                        'report':name})
        print(entry['id'], 'PASS' if status else 'FAIL/INCONCLUSIVE',
              'D <=',rec['checked_max_diameter_upper'],flush=True)
    (args.output_dir/('summary_raw.json' if args.raw else 'summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if success else 1
if __name__=='__main__':raise SystemExit(main())
