#!/usr/bin/env python3
"""Exact four-cell constructions for tetrahedron, cube, octahedron, icosahedron.

Standard library only. Coordinates are in a rational/quadratic RAW scale;
physical_scale_squared gives the conversion to inradius 1/2. All diameter and
coverage decisions are exact. For the icosahedron the final boundary is explicitly
subdivided and checked; common origin extends boundary coverage to the solid.
Global optimality is proved here only for tetrahedron and octahedron.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from decimal import Decimal,localcontext
from fractions import Fraction as F
from functools import total_ordering
import itertools as it
import json
import math
from pathlib import Path
import exact_cover_check as ec
@total_ordering
@dataclass(frozen=True)
class Q5:
    """a+b*sqrt(5), with exact comparison (squaring rational numbers only)."""
    a: F = F(0)
    b: F = F(0)
    def __post_init__(self):
        object.__setattr__(self, 'a', F(self.a))
        object.__setattr__(self, 'b', F(self.b))
    @staticmethod
    def coerce(x):
        return x if isinstance(x, Q5) else Q5(x)
    def __add__(self, other):
        o=self.coerce(other); return Q5(self.a+o.a, self.b+o.b)
    __radd__=__add__
    def __neg__(self): return Q5(-self.a,-self.b)
    def __sub__(self, other): return self+-self.coerce(other)
    def __rsub__(self, other): return self.coerce(other)+-self
    def __mul__(self, other):
        o=self.coerce(other)
        return Q5(self.a*o.a+5*self.b*o.b, self.a*o.b+self.b*o.a)
    __rmul__=__mul__
    def __truediv__(self, other):
        o=self.coerce(other); d=o.a*o.a-5*o.b*o.b
        if not d: raise ZeroDivisionError('Division by zero in Q(sqrt(5)).')
        return Q5((self.a*o.a-5*self.b*o.b)/d,(self.b*o.a-self.a*o.b)/d)
    def __rtruediv__(self, other): return self.coerce(other)/self
    def __pow__(self, n):
        if not isinstance(n,int) or n<0: raise ValueError('Nonnegative integer power required.')
        x,r=self,Q5(1)
        while n:
            if n&1:r=r*x
            x=x*x;n//=2
        return r
    def sign(self):
        a,b=self.a,self.b
        if not b:return (a>0)-(a<0)
        if not a:return (b>0)-(b<0)
        if (a>0)==(b>0):return 1 if a>0 else -1
        d=a*a-5*b*b
        return ((d>0)-(d<0))*(1 if a>0 else -1)
    def __lt__(self, other):return (self-self.coerce(other)).sign()<0
    def __eq__(self, other):
        if not isinstance(other,(Q5,int,F)):return NotImplemented
        o=self.coerce(other);return self.a==o.a and self.b==o.b
    def __hash__(self):return hash((self.a,self.b)) if self.b else hash(self.a)
    def __bool__(self):return bool(self.a or self.b)
    def __abs__(self): return self if self.sign()>=0 else -self
    def decimal(self, digits=40):
        # Display only. NEVER used for geometric decisions.
        with localcontext() as ctx:
            ctx.prec=digits
            return str(Decimal(self.a.numerator)/Decimal(self.a.denominator)+
                       Decimal(5).sqrt()*Decimal(self.b.numerator)/Decimal(self.b.denominator))
    def record(self):return {'rational':str(self.a),'sqrt5':str(self.b)}
    def expression(self):return f'({self.a}) + ({self.b})*sqrt(5)'

def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q5())
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def area(poly):
    if len(poly)<3:return Q5()
    return sum((cross2(a,b) for a,b in zip(poly,poly[1:]+poly[:1])),Q5())/2

def ccw(poly):return list(poly) if area(poly)>=0 else list(reversed(poly))
def intersect2(poly,clip):
    out=list(poly)
    for a,b in zip(clip,clip[1:]+clip[:1]):
        inp=out;out=[]
        if not inp:break
        for s,e in zip(inp,inp[1:]+inp[:1]):
            fs,fe=cross2(sub(b,a),sub(s,a)),cross2(sub(b,a),sub(e,a))
            if fs>=0:out.append(s)
            if (fs<0<fe) or (fe<0<fs):
                t=fs/(fs-fe);out.append(tuple(x+t*(y-x) for x,y in zip(s,e)))
    return out

def sqrt_bracket(x,places=15):
    s=10**places;scaled=x*s*s;lo,hi=0,s
    while Q5(hi*hi)<scaled:hi*=2
    while hi-lo>1:
        m=(lo+hi)//2
        if Q5(m*m)<=scaled:lo=m
        else:hi=m
    if Q5(hi*hi)==scaled:lo=hi
    hi=lo if Q5(lo*lo)==scaled else lo+1
    return [f'{n//s}.{n%s:0{places}d}' for n in (lo,hi)]

def pair_record(points,groups,scale2):
    records=[];best=Q5()
    for k,g in enumerate(groups):
        ds=[(dot(sub(points[i],points[j]),sub(points[i],points[j]))*scale2,[i,j]) for i,j in it.combinations(g,2)]
        d=max(x[0] for x in ds);best=max(best,d)
        records.append(dict(cell=k+1,memberships=g,diameter_squared=d.record(),
                            diameter_interval=sqrt_bracket(d),maximizing_pairs=[p for v,p in ds if v==d]))
    return best,records

def rational_construction(name):
    signs=list(it.product((-1,1),repeat=3));scores=[v for v in signs if math.prod(v)==1]
    axes=[tuple(s if j==i else 0 for j in range(3)) for i in range(3) for s in (-1,1)]
    if name=='tetrahedron':
        hs=[ec.plane([-x for x in v]+[1]) for v in scores];scale2=F(3,4)
    elif name=='octahedron':hs=[ec.plane(list(v)+[1]) for v in signs];scale2=F(3,4)
    elif name=='cube':hs=[ec.plane(list(v)+[F(1,2)]) for v in axes];scale2=F(1)
    else:raise ValueError(name)
    tv=ec.poly_vertices(hs);cells=[]
    if name=='cube':
        for a,b in it.product((-1,1),repeat=2):cells.append(hs+[ec.plane([a,0,0,0]),ec.plane([0,b,0,0])])
    else:
        for i in range(4):
            cells.append(hs+[ec.plane([scores[j][k]-scores[i][k] for k in range(3)]+[0]) for j in range(4) if j!=i])
    allpts=list(tv);groups=[]
    for cell in cells:
        vs=ec.poly_vertices(cell);g=[]
        for v in vs:
            if v not in allpts:allpts.append(v)
            g.append(allpts.index(v))
        groups.append(g)
    cover=ec.volume_cover(hs,tv,cells)
    if not cover['missing_volume_exact_zero']:raise ArithmeticError('Coverage failed')
    raw=[tuple(Q5(F(v[j],v[3])) for j in range(3)) for v in allpts]
    best,cell_records=pair_record(raw,groups,Q5(scale2))
    expected=Q5(F(9,4) if name=='tetrahedron' else F(3,2))
    if best!=expected:raise ArithmeticError('Unexpected maximum diameter')
    # Each distinct pair has opposite separating inequalities in its cell systems.
    # Verify no pairwise full-dimensional intersections as well.
    for i,j in it.combinations(range(4),2):
        if ec.full_dimension(ec.poly_vertices(list(dict.fromkeys(cells[i]+cells[j])))):
            raise ArithmeticError('Unexpected interior overlap')
    lower=(dict(proved=True,diameter_squared=expected.record(),
           argument='Four tetrahedral vertices are pairwise distance sqrt(6). Below 3/2 they must lie in distinct cells. The origin lies in some cell and is distance 3/2 from that cell\'s vertex.') if name=='tetrahedron' else
        dict(proved=True,diameter_squared=expected.record(),argument='Six octahedral vertices in four covering sets force two vertices into one set. Every distinct pair is at least sqrt(3/2) apart.') if name=='octahedron' else dict(proved=False))
    return dict(solid=name,normalization='inradius_1/2',raw_points_exact=[[q.record() for q in v] for v in raw],
        physical_scale_squared=Q5(scale2).record(),fixed_count=len(tv),point_count=len(raw),groups=groups,cells=cell_records,
        maximum_diameter_squared=best.record(),maximum_diameter_interval=sqrt_bracket(best),
        coverage_proved=True,coverage_method='Exact raw rational volume inclusion-exclusion',full_volume_check=cover,
        convex_interiors_disjoint=True,global_lower_bound=lower)

def icosahedron_construction():
    phi=Q5(F(1,2),F(1,2));phi2=phi*phi;zero=Q5()
    def cycle(a,b):return [tuple(v[k:]+v[:k]) for s,t in it.product((-1,1),repeat=2) for v in [(zero,s*a,t*b)] for k in range(3)]
    vertices=sorted(cycle(Q5(1),phi));normals=[tuple(Q5(x) for x in v) for v in it.product((-1,1),repeat=3)]+cycle(phi,1/phi)
    hs=[(n,phi2) for n in normals]
    if len(set(vertices))!=12:raise ArithmeticError('Wrong vertex count')
    if any(dot(n,v)>h for n,h in hs for v in vertices):raise ArithmeticError('Invalid target')
    faces=[[i for i,v in enumerate(vertices) if dot(n,v)==h] for n,h in hs]
    if len(faces)!=20 or any(len(f)!=3 for f in faces):raise ArithmeticError('Wrong triangular faces')
    # Four mutually disjoint original triangular faces choose the corner owners.
    colors=[3,1,2,1,3,2,3,1,2,0,0,0]
    groups=[[i for i,c in enumerate(colors) if c==k] for k in range(4)]
    if any(sorted(g) not in [sorted(f) for f in faces] for g in groups):raise ArithmeticError('Corner triples are not target faces')
    points=list(vertices);index={v:i for i,v in enumerate(points)}
    def add(v,owners):
        v=tuple(v)
        if v not in index:index[v]=len(points);points.append(v)
        j=index[v]
        for k in owners:
            if j not in groups[k]:groups[k].append(j)
        return j
    origin=add((zero,zero,zero),range(4));checks=[]
    def midpoint(i,j):return add([(vertices[i][k]+vertices[j][k])/2 for k in range(3)],{colors[i],colors[j]})
    for fi,((n,h),f) in enumerate(zip(hs,faces)):
        owners=set(colors[i] for i in f);polygons=[]
        if len(owners)==1:polygons=[(next(iter(owners)),f)]
        elif len(owners)==2:
            for owner in sorted(owners):
                g=[]
                for i,j in zip(f,f[1:]+f[:1]):
                    if colors[i]==owner:g.append(i)
                    if colors[i]!=colors[j]:g.append(midpoint(i,j))
                polygons.append((owner,g))
        else:
            center=add([sum((vertices[i][k] for i in f),Q5())/3 for k in range(3)],owners)
            for pos,i in enumerate(f):
                j,k=f[(pos+1)%3],f[(pos+2)%3]
                polygons.append((colors[i],[i,midpoint(i,j),center,midpoint(k,i)]))
        axes=[k for k in range(3) if k!=max(range(3),key=lambda j:abs(n[j]))]
        def project(g):return ccw([tuple(points[i][k] for k in axes) for i in g])
        Fpoly=project(f);pieces=[project(g) for owner,g in polygons]
        if any(not set(g)<=set(groups[owner]) for owner,g in polygons):raise ArithmeticError('Ownerless polygon')
        if any(area(intersect2(p,Fpoly))!=area(p) for p in pieces):raise ArithmeticError('Patch outside face')
        if any(area(intersect2(a,b))!=0 for a,b in it.combinations(pieces,2)):raise ArithmeticError('Overlapping patches')
        if sum((area(p) for p in pieces),Q5())!=area(Fpoly):raise ArithmeticError('Missing face area')
        checks.append(dict(face=fi,vertices=f,patches=[dict(cell=k+1,vertices=g) for k,g in polygons],missing_projected_area_exact_zero=True))
    for v in points:
        if any(dot(n,v)>h for n,h in hs):raise ArithmeticError('Point outside target')
    scale2=3/(4*phi**4);best,cell_records=pair_record(points,groups,scale2)
    expected=9/(4*phi2)
    if best!=expected:raise ArithmeticError('Diameter is not 3/(2phi)')
    return dict(solid='icosahedron',normalization='inradius_1/2',raw_points_exact=[[q.record() for q in v] for v in points],
        physical_scale_squared=scale2.record(),fixed_count=12,point_count=len(points),groups=groups,cells=cell_records,
        maximum_diameter_squared=best.record(),maximum_diameter_interval=sqrt_bracket(best),
        maximum_diameter_expression='3/(2*phi) = 3*(sqrt(5)-1)/4',coverage_proved=True,
        coverage_method='Exact subdivision of every ideal target face into nonoverlapping owner patches, with common origin',
        boundary_faces=checks,common_point_index=origin,convex_interiors_disjoint=None,global_lower_bound=dict(proved=False))

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solids',nargs='+',choices=['tetrahedron','cube','octahedron','icosahedron'],default=['tetrahedron','cube','octahedron','icosahedron'])
    ap.add_argument('--output',type=Path,default=Path('runs/platonic_exact'))
    ap.add_argument('--export-displays',action='store_true',help='Also export decimal five-line files for visualization, NOT certificates.')
    args=ap.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.solids:
        r=icosahedron_construction() if name=='icosahedron' else rational_construction(name)
        (args.output/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        if args.export_displays:
            def qfloat(obj):return float(Q5(F(obj['rational']),F(obj['sqrt5'])).decimal(40))
            scale=math.sqrt(qfloat(r['physical_scale_squared']))
            points=[[qfloat(x)*scale for x in v] for v in r['raw_points_exact']]
            diameter=max(math.dist(points[i],points[j]) for g in r['groups'] for i,j in it.combinations(g,2))
            (args.output/(name+'_display.txt')).write_text('4\n'+repr(diameter)+'\n'+str(len(points))+'\n'+json.dumps(r['groups'])+'\n'+json.dumps(points)+'\n')
        print(name,r['maximum_diameter_interval'],'global optimum:',r['global_lower_bound']['proved'],flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
