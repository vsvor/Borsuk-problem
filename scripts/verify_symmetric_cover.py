#!/usr/bin/env python3
"""Verify a FINAL cover of an ideal symmetric target with exact rational arithmetic.

Unlike passing a decimal target vertex array to exact_cover_check.py, this
command rigorously encloses the ideal radical coordinates/planes. No numerical
optimizer or original topology is trusted. Python standard library only.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import exact_cover_check as ec
from symmetric_polyhedra import NAMES, rational_outer_halfspaces, rational_inner_halfspaces


def verify(path,solid,normalization='inradius',padding=F('1e-12'),bound=None,full=False,overlaps=False):
    if padding<0:raise ValueError('Padding must be nonnegative.')
    if bound is not None and bound<=0:raise ValueError('Diameter bound must be positive.')
    ps,gs,_=ec.load_partition(path)
    hs=[ec.plane(n+[h]) for n,h in rational_outer_halfspaces(solid,normalization)]
    vs=ec.poly_vertices(hs)
    rec=ec.verify(ps,gs,hs,vs,padding=padding,bound=bound,full=full,overlaps=overlaps)
    rec.update(input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),solid=solid,normalization=normalization,
               ideal_target_enclosure='Integer-square-root rational intervals at 40 decimal places',
               checked_target_halfspaces_exact=[[str(x) for x in h] for h in hs],
               covers_ideal_target=True if rec['covers_checked_target'] else None)
    if rec['covers_checked_target'] is False:
        ih=[ec.plane(n+[h]) for n,h in rational_inner_halfspaces(solid,normalization)]
        iv=ec.poly_vertices(ih);cells=[[tuple(map(int,h)) for h in c] for c in rec['checked_cell_halfspaces_exact']]
        check=ec.boundary_cover(ih,iv,cells)
        if not check['covered']:
            rec['covers_ideal_target']=False;rec['ideal_target_counterexample']=check['witness']
    return rec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--solid',choices=NAMES,required=True)
    p.add_argument('--normalization',choices=['inradius','min-width'],default='inradius')
    p.add_argument('--pad',type=F,default=F('1e-12'));p.add_argument('--diameter-bound',type=F)
    p.add_argument('--full-volume',action='store_true');p.add_argument('--overlaps',action='store_true')
    p.add_argument('--report',type=Path);args=p.parse_args(argv)
    if args.report and args.report.resolve()==args.input.resolve():p.error('Do not overwrite the input file.')
    try:
        r=verify(args.input,args.solid,args.normalization,args.pad,args.diameter_bound,args.full_volume,args.overlaps)
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(r,indent=2)+'\n')
        ok=r['covers_ideal_target'] is True and r['diameter_bound_proved'] is not False
        print('PASS' if ok else 'FAIL/INCONCLUSIVE',args.solid,'diameter <=',r['checked_max_diameter_upper'])
        return 0 if ok else 1
    except (ValueError,OSError) as e:p.error(str(e))
if __name__=='__main__':raise SystemExit(main())
