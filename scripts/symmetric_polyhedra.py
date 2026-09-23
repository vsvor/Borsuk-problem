#!/usr/bin/env python3
"""6-TRD and five Platonic solids, with inradius 1/2, for four-cell searches.

--normalization min-width instead gives least caliper width 1. Only the
regular tetrahedron changes (by 1/sqrt(3)); inradius 1/2 gives its altitude 2.
Ideal geometry is specified by formulas. Rational intervals enclose radicals
for an independent outer target. Floats are used only for numerical searches.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import itertools as it
import json
import math
from pathlib import Path
import subprocess
import sys

NAMES=('six_trd','tetrahedron','cube','octahedron','dodecahedron','icosahedron')

@dataclass(frozen=True)
class Interval:
    lo:F
    hi:F
    def __post_init__(self):
        object.__setattr__(self,'lo',F(self.lo));object.__setattr__(self,'hi',F(self.hi))
        if self.lo>self.hi:raise ValueError('Reversed interval')
    @staticmethod
    def exact(x):return Interval(F(x),F(x))
    @staticmethod
    def coerce(x):return x if isinstance(x,Interval) else Interval.exact(x)
    def __add__(self,b):
        b=self.coerce(b);return Interval(self.lo+b.lo,self.hi+b.hi)
    __radd__=__add__
    def __neg__(self):return Interval(-self.hi,-self.lo)
    def __sub__(self,b):return self+-self.coerce(b)
    def __rsub__(self,b):return self.coerce(b)+-self
    def __mul__(self,b):
        b=self.coerce(b);v=[x*y for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        return Interval(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=self.coerce(b)
        if b.lo<=0<=b.hi:raise ZeroDivisionError('Interval includes zero')
        return self*Interval(1/b.hi,1/b.lo)
    def __rtruediv__(self,b):return self.coerce(b)/self
    def sqrt(self,digits=50):
        if self.lo<0:raise ValueError('Negative square root')
        s=10**digits
        def fl(x):return math.isqrt(x.numerator*s*s//x.denominator)
        a,b=fl(self.lo),fl(self.hi)
        return Interval(F(a,s),F(b,s) if F(b*b,s*s)==self.hi else F(b+1,s))
    @property
    def mid(self):return (self.lo+self.hi)/2
    def __float__(self):return float(self.mid)


def _axes():return [[s if j==k else 0 for j in range(3)] for k in range(3) for s in (-1,1)]

def _cycle(a,b):
    return [list(v[k:]+v[:k]) for s,t in it.product((-1,1),repeat=2)
            for v in [(0,s*a,t*b)] for k in range(3)]


def ideal_geometry(name,normalization='inradius',digits=50):
    """Interval vertices and facet (normal,offset) pairs of the ideal solid.

    phi=(1+sqrt(5))/2. Ico raw vertices: cyclic (0,+/-1,+/-phi).
    Dodeca raw vertices: (+/-1)^3 and cyclic (0,+/-phi,+/-1/phi).
    Their scales are sqrt(3)/(2phi^2) and sqrt(phi+2)/(2phi^2).
    Tetra raw vertices have coordinates +/-1 with product +1, scale sqrt(3)/2.
    Octa raw vertices are +/-coordinate unit vectors, scale sqrt(3)/2.
    """
    if name not in NAMES:raise ValueError('Unknown solid: '+name)
    if normalization not in ('inradius','min-width'):raise ValueError('Unknown normalization')
    I=Interval.exact;s3=I(3).sqrt(digits);half=I(F(1,2))
    signs=[list(x) for x in it.product((-1,1),repeat=3)]
    if name=='six_trd':
        t=1/I(2).sqrt(digits);a=t-half
        vertices=[[s*t/2 for s in v] for v in signs]
        for k in range(3):vertices += [[s*(half if j==k else a) for j,s in enumerate(v)] for v in signs]
        planes=[]
        for i,j in ((0,1),(0,2),(1,2)):
            for s,u in it.product((-1,1),repeat=2):
                n=[0,0,0];n[i],n[j]=s,u;planes.append((n,t))
        planes += [(n,half) for n in _axes()]
    elif name=='tetrahedron':
        raw=[v for v in signs if math.prod(v)==1];scale=s3/2
        vertices=[[scale*x for x in v] for v in raw]
        planes=[([-x for x in v],scale) for v in raw]
    elif name=='cube':
        vertices=[[half*x for x in v] for v in signs];planes=[(n,half) for n in _axes()]
    elif name=='octahedron':
        vertices=[[s3*x/2 for x in v] for v in _axes()];planes=[(n,s3/2) for n in signs]
    else:
        phi=(1+I(5).sqrt(digits))/2;phi2=phi*phi
        ico=_cycle(I(1),phi);dod=signs+_cycle(phi,1/phi)
        if name=='icosahedron':scale=s3/(2*phi2);raw,ns=ico,dod;offset=s3/2
        else:
            root=(phi+2).sqrt(digits);scale=root/(2*phi2);raw,ns=dod,ico;offset=root/2
        vertices=[[scale*x for x in v] for v in raw];planes=[(n,offset) for n in ns]
    factor=1/s3 if name=='tetrahedron' and normalization=='min-width' else I(1)
    vertices=[[Interval.coerce(x)*factor for x in v] for v in vertices]
    planes=[([Interval.coerce(x) for x in n],Interval.coerce(h)*factor) for n,h in planes]
    return vertices,planes


def rational_outer_halfspaces(name,normalization='inradius',digits=40):
    """Contains IDEAL target: round normals q, increase rhs by M*||q-n||_1.

    All ideal vertices have |x_j|<=M. Thus n*x<=h implies
    q*x<=h_upper + M*sum_j max|q_j-n_j|. Arithmetic is exact.
    """
    vs,hs=ideal_geometry(name,normalization,digits+10)
    M=max(max(abs(x.lo),abs(x.hi)) for v in vs for x in v);grid=10**digits;answer=[]
    for n,h in hs:
        q=[F(round(x.mid*grid),grid) for x in n]
        error=sum(max(abs(y-x.lo),abs(y-x.hi)) for y,x in zip(q,n))
        rhs=h.hi+M*error;rhs=F(-((-rhs.numerator*grid)//rhs.denominator),grid)
        assert rhs>=h.hi+M*error
        answer.append((q,rhs))
    return answer


def rational_inner_halfspaces(name,normalization='inradius',digits=40):
    """Contained in the IDEAL target; includes an explicit bounding box.

    q*x<=h_lower-M*sum(max|q_j-n_j|), |x_j|<=M imply n*x<=h.
    """
    vs,hs=ideal_geometry(name,normalization,digits+10)
    M=max(max(abs(x.lo),abs(x.hi)) for v in vs for x in v);grid=10**digits
    # A simple rational bounding box is enough for the error calculation.
    M=F(math.ceil(M));answer=[]
    for n,h in hs:
        q=[F(round(x.mid*grid),grid) for x in n]
        error=sum(max(abs(y-x.lo),abs(y-x.hi)) for y,x in zip(q,n))
        rhs=h.lo-M*error;rhs=F((rhs.numerator*grid)//rhs.denominator,grid)
        assert rhs<=h.lo-M*error
        answer.append((q,rhs))
    answer.extend(([F(x) for x in n],M) for n in _axes())
    return answer


def make_target(name,normalization='inradius'):
    import numpy as np
    from scipy.spatial import ConvexHull
    vs,hs=ideal_geometry(name,normalization)
    V=np.array([[float(x) for x in v] for v in vs]);V=np.array(sorted(V,key=lambda v:tuple(np.round(v,12))))
    hull=ConvexHull(V);radii=-hull.equations[:,3]/np.linalg.norm(hull.equations[:,:3],axis=1)
    radius=float(min(radii));expected=1/(2*math.sqrt(3)) if name=='tetrahedron' and normalization=='min-width' else .5
    if not np.allclose(radii,expected,rtol=0,atol=2e-14):raise ArithmeticError('Wrong facet distances')
    planes=np.array([[float(x) for x in n]+[float(h)] for n,h in hs])
    if (V@planes[:,:3].T-planes[:,3]).max()>2e-14:raise ArithmeticError('Invalid analytic facets')
    outer=rational_outer_halfspaces(name,normalization)
    return dict(format_version=1,family='symmetric_comparison',solid=name,normalization=normalization,
        inradius=radius,minimum_width=2*radius*(math.sqrt(3) if name=='tetrahedron' else 1),
        tetrahedral_altitude=4*radius if name=='tetrahedron' else None,
        vertices=V.tolist(),vertex_count=len(V),fixed_count=len(V),facet_count=len(hs),
        target_volume=float(hull.volume),target_diameter=float(np.linalg.norm(V[:,None]-V[None,:],axis=2).max()),
        analytic_definition='scripts/symmetric_polyhedra.py:ideal_geometry',
        rational_outer_halfspaces=[{'normal':[str(x) for x in n],'rhs':str(h)} for n,h in outer],enclosure_digits=40,
        note='Default 2*inradius=1, not unit diameter. The tetrahedral altitude and minimum width differ. Rational outer target uses integer-square-root interval bounds.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--solids',nargs='+',choices=NAMES,default=list(NAMES))
    p.add_argument('--normalization',choices=['inradius','min-width'],default='inradius')
    p.add_argument('--output',type=Path,default=Path('runs/symmetric'));p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true')
    args,extra=p.parse_known_args(argv)
    if extra and not args.run:p.error('Search arguments require --run: '+' '.join(extra))
    args.output.mkdir(parents=True,exist_ok=True);results={}
    for name in args.solids:
        rec=make_target(name,args.normalization);tp=args.output/f'target_{name}.json'
        if tp.exists():
            old=json.loads(tp.read_text())
            identity=('solid','normalization','vertices','rational_outer_halfspaces')
            if any(old.get(k)!=rec[k] for k in identity):p.error(f'Different target exists: {tp}')
            # Keep original metadata rather than rewriting tiny version-dependent
            # differences in the diagnostic floating-point area/volume fields.
            rec=old
        else:tp.write_text(json.dumps(rec,indent=2)+'\n')
        print(f'{name}: V={rec["vertex_count"]} F={rec["facet_count"]} r={rec["inradius"]:.12f} diam={rec["target_diameter"]:.12f}',flush=True)
        if args.run:
            cmd=[sys.executable,str(Path(__file__).with_name('partitions_slsqp.py')),'--vertices',str(tp),'--parts','4','--output-dir',str(args.output/name)]+extra
            if args.resume:cmd+=['--resume']
            rc=subprocess.run(cmd,check=False).returncode
            if rc:return rc
            best=json.loads((args.output/name/'best.json').read_text());state=json.loads((args.output/name/'state.json').read_text())
            results[name]=dict(diameter=best['diameter'],points=len(best['points']),completed=state['completed_total'],counts=state['counts'])
    if args.run:(args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
