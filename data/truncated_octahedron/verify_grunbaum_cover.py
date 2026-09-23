#!/usr/bin/env python3
"""Exact rational verification of a four-hull cover of the ideal Grünbaum body.

The ideal target is |x|+|y|+|z| <= sqrt(3)/2 and x,y,z <= 1/2.
A rational OUTER enclosure of sqrt(3)/2 is checked, so PASS proves coverage
of the ideal target.  Decimal partition coordinates are interpreted exactly.
"""
from __future__ import annotations
import argparse, itertools as it, json, math, sys
from fractions import Fraction as F
from pathlib import Path
import exact_cover_check as ec


def target(places=30, enclosure='outer'):
    scale=10**places
    # m/scale brackets sqrt(3)/2.
    # choose m so 4 m^2 >= 3 scale^2, with strict outer unless exact.
    n=3*scale*scale
    m=math.isqrt(n//4)
    while 4*m*m < n: m+=1
    if 4*m*m == n: pass
    if enclosure=='outer':
        c=F(m,scale)
        if 4*c*c == 3: c += F(1,scale)
    elif enclosure=='inner':
        c=F(m,scale)
        if 4*c*c >= 3: c -= F(1,scale)
    else: raise ValueError('enclosure must be outer or inner')
    assert (4*c*c > 3) if enclosure=='outer' else (4*c*c < 3)
    hs=[]
    for s in it.product((-1,1), repeat=3):
        hs.append(ec.plane([s[0],s[1],s[2],c]))
    hs += [ec.plane([1,0,0,F(1,2)]),ec.plane([0,1,0,F(1,2)]),ec.plane([0,0,1,F(1,2)])]
    vs=ec.poly_vertices(hs)
    return hs,vs,{'kind':f'rational_{enclosure}_enclosure','sqrt3_over_2':str(c),'vertex_count':len(vs)}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    ap.add_argument('--pad',type=F,default=F(0))
    ap.add_argument('--diameter-bound',type=F)
    ap.add_argument('--full-volume',action='store_true')
    ap.add_argument('--overlaps',action='store_true')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args(argv)
    ps,groups,stored=ec.load_partition(args.input)
    oh,ov,meta=target(enclosure='outer')
    ans=ec.verify(ps,groups,oh,ov,padding=args.pad,bound=args.diameter_bound,
                  full=args.full_volume,overlaps=args.overlaps)
    ans.update(target=meta,input=str(args.input),stored_diameter=str(stored))
    ans['covers_ideal_target']=True if ans['covers_checked_target'] else None
    if ans['covers_checked_target'] is False:
        ih,iv,_=target(enclosure='inner')
        inner=ec.verify(ps,groups,ih,iv,padding=args.pad,bound=args.diameter_bound,full=args.full_volume)
        if inner['covers_checked_target'] is False:
            ans['covers_ideal_target']=False
            ans['ideal_target_counterexample']=inner.get('boundary',{}).get('witness')
    report=args.report or args.input.with_suffix('.grunbaum_check.json')
    report.write_text(json.dumps(ans,indent=2,allow_nan=False)+'\n')
    print('Coverage outer rational target:',ans['covers_checked_target'])
    print('Coverage ideal target:',ans['covers_ideal_target'])
    print('Original maximum diameter <=',ans['original_max_diameter_upper'])
    print('Checked maximum diameter  <=',ans['checked_max_diameter_upper'])
    print('Diameter bound proved:',ans['diameter_bound_proved'])
    print('Report:',report)
    return 0 if ans['covers_ideal_target'] is True and ans['diameter_bound_proved'] is not False else 1

if __name__=='__main__': raise SystemExit(main())
