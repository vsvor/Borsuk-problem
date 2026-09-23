#!/usr/bin/env python3
"""Rationalize four compatible scores and certify the resulting conical partition.

Standard library only. All geometric decisions use exact Q(sqrt(2)) arithmetic.
The target is the IDEAL 23-vertex positive three-truncation body. The JSON
certificate is authoritative; the five-line decimal export is for plotting.
This certifies a rational approximation, not an exact formula for an optimizer.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import total_ordering
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import sys

@total_ordering
@dataclass(frozen=True)
class Q2:
    """a+b*sqrt(2), with exact comparison (squaring rational numbers only)."""
    a: F = F(0)
    b: F = F(0)
    def __post_init__(self):
        object.__setattr__(self, 'a', F(self.a))
        object.__setattr__(self, 'b', F(self.b))
    @staticmethod
    def coerce(x):
        return x if isinstance(x, Q2) else Q2(x)
    def __add__(self, other):
        o=self.coerce(other); return Q2(self.a+o.a, self.b+o.b)
    __radd__=__add__
    def __neg__(self): return Q2(-self.a,-self.b)
    def __sub__(self, other): return self+-self.coerce(other)
    def __rsub__(self, other): return self.coerce(other)+-self
    def __mul__(self, other):
        o=self.coerce(other)
        return Q2(self.a*o.a+2*self.b*o.b, self.a*o.b+self.b*o.a)
    __rmul__=__mul__
    def __truediv__(self, other):
        o=self.coerce(other); d=o.a*o.a-2*o.b*o.b
        if not d: raise ZeroDivisionError('Division by zero in Q(sqrt(2)).')
        return Q2((self.a*o.a-2*self.b*o.b)/d,(self.b*o.a-self.a*o.b)/d)
    def __rtruediv__(self, other): return self.coerce(other)/self
    def __pow__(self, n):
        if not isinstance(n,int) or n<0: raise ValueError('Nonnegative integer power required.')
        x,r=self,Q2(1)
        while n:
            if n&1:r=r*x
            x=x*x;n//=2
        return r
    def sign(self):
        a,b=self.a,self.b
        if not b:return (a>0)-(a<0)
        if not a:return (b>0)-(b<0)
        if (a>0)==(b>0):return 1 if a>0 else -1
        d=a*a-2*b*b
        return ((d>0)-(d<0))*(1 if a>0 else -1)
    def __lt__(self, other):return (self-self.coerce(other)).sign()<0
    def __eq__(self, other):
        if not isinstance(other,(Q2,int,F)):return NotImplemented
        o=self.coerce(other);return self.a==o.a and self.b==o.b
    def __hash__(self):return hash((self.a,self.b)) if self.b else hash(self.a)
    def __bool__(self):return bool(self.a or self.b)
    def decimal(self, digits=40):
        # Display only. NEVER used for geometric decisions.
        with localcontext() as ctx:
            ctx.prec=digits
            return str(Decimal(self.a.numerator)/Decimal(self.a.denominator)+
                       Decimal(2).sqrt()*Decimal(self.b.numerator)/Decimal(self.b.denominator))
    def record(self):return {'rational':str(self.a),'sqrt2':str(self.b)}
    def expression(self):return f'({self.a}) + ({self.b})*sqrt(2)'

def dot(a,b):return sum((x*y for x,y in zip(a,b)),0)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def det(a,b,c):return dot(a,cross(b,c))
def check(condition,message):
    if not condition:raise ArithmeticError(message)

def primitive_plane(n,h):
    g=math.gcd(*n)
    if not g:raise ValueError('Zero normal.')
    return tuple(x//g for x in n),h/g

def target_planes():
    planes=[]
    for i,j in ((0,1),(0,2),(1,2)):
        for si,sj in it.product((-1,1),repeat=2):
            n=[0,0,0];n[i],n[j]=si,sj
            planes.append((tuple(n),Q2(0,F(1,2))))
    for i in range(3):
        n=[0,0,0];n[i]=1;planes.append((tuple(n),Q2(F(1,2))))
    return planes

def poly_vertices(hs):
    hs=list(dict.fromkeys(hs));out=set()
    for (a,ha),(b,hb),(c,hc) in it.combinations(hs,3):
        bc,ca,ab=cross(b,c),cross(c,a),cross(a,b)
        d=dot(a,bc)
        if not d:continue
        p=tuple((ha*bc[j]+hb*ca[j]+hc*ab[j])/d for j in range(3))
        if all(dot(n,p)<=h for n,h in hs):out.add(p)
    return sorted(out)

def dimension(v):
    if not v:return -1
    ds=[sub(p,v[0]) for p in v[1:]]
    a=next((d for d in ds if any(d)),None)
    if a is None:return 0
    n=next((cross(a,d) for d in ds if any(cross(a,d))),None)
    if n is None:return 1
    return 3 if any(dot(n,d) for d in ds) else 2

def face_hull(ps,axes):
    ps=sorted(set(ps),key=lambda p:tuple(p[j] for j in axes))
    if len(ps)<3:return []
    i,j=axes
    def orient(a,b,c):return (b[i]-a[i])*(c[j]-a[j])-(b[j]-a[j])*(c[i]-a[i])
    def half(seq):
        ans=[]
        for p in seq:
            while len(ans)>1 and orient(ans[-2],ans[-1],p)<=0:ans.pop()
            ans.append(p)
        return ans
    ans=half(ps)[:-1]+half(list(reversed(ps)))[:-1]
    return ans if len(ans)>=3 else []

def volume(hs,vs):
    if dimension(vs)<3:return Q2(0)
    total=Q2(0)
    for n,h in dict.fromkeys(hs):
        drop=max(range(3),key=lambda j:abs(n[j]));axes=tuple(j for j in range(3) if j!=drop)
        face=face_hull([p for p in vs if dot(n,p)==h],axes)
        if not face:continue
        sign=dot(n,cross(sub(face[1],face[0]),sub(face[2],face[0]))).sign()
        for b,c in zip(face[1:-1],face[2:]):total+=sign*det(face[0],b,c)/6
    check(total>0,'Nonpositive exact volume.')
    return total

def sqrt_bracket(x,places=15):
    """Floor/ceiling at a decimal grid, using only exact comparisons."""
    if x<0:raise ValueError('Negative square.')
    s=10**places;scaled=x*(s*s);lo,hi=0,s
    while Q2(hi*hi)<scaled:hi*=2
    while hi-lo>1:
        m=(lo+hi)//2
        if Q2(m*m)<=scaled:lo=m
        else:hi=m
    if Q2(hi*hi)==scaled:lo=hi
    hi=lo if Q2(lo*lo)==scaled else lo+1
    def show(n):return f'{n//s}.{n%s:0{places}d}'
    return show(lo),show(hi)

def integer_polynomial(d2):
    a,b=d2.a,d2.b
    co=[F(1),F(0),-2*a,F(0),a*a-2*b*b] if b else [F(1),F(0),-a]
    den=math.lcm(*(x.denominator for x in co))
    vals=[int(x*den) for x in co];g=math.gcd(*vals)
    return [str(x//g) for x in vals]

def rational_scores(source,denominator):
    raw=json.loads(Path(source).read_text(),parse_float=F)
    v=[[F(x) for x in row] for row in raw['scores']]
    if len(v)!=4 or any(len(r)!=3 for r in v):raise ValueError('Four 3-D score vectors required.')
    # Gauge: subtract the first score. Round jointly, not independent interfaces.
    scores=[tuple(round((x-y)*denominator) for x,y in zip(row,v[0])) for row in v]
    err=max(abs(F(scores[i][j],denominator)-(v[i][j]-v[0][j])) for i in range(4) for j in range(3))
    if not det(sub(scores[1],scores[0]),sub(scores[2],scores[0]),sub(scores[3],scores[0])):
        raise ValueError('Rounded scores are affinely dependent; increase denominator.')
    return scores,err

def certify(source,denominator=10**12,bound=F('0.97288131961')):
    if denominator<=0:raise ValueError('Denominator must be positive.')
    scores,error=rational_scores(source,denominator)
    th=target_planes();tv=poly_vertices(th);target_volume=volume(th,tv)
    check(len(tv)==23 and target_volume==Q2(F(7,2),-2),'Wrong ideal target.')
    planes=[]
    for i,j in it.combinations(range(4),2):
        n=primitive_plane(sub(scores[j],scores[i]),Q2())[0]
        planes.append({'cells':[i+1,j+1],'normal':[str(t) for t in n],'rhs':'0'})
    cells=[]
    for i in range(4):
        hs=th+[primitive_plane(sub(scores[j],scores[i]),Q2()) for j in range(4) if i!=j]
        vs=poly_vertices(hs);check(dimension(vs)==3,'Empty or flat cell.')
        cells.append((hs,vs))
    allpts=list(tv)
    for _,vs in cells:
        for p in vs:
            if p not in allpts:allpts.append(p)
    ix={p:i for i,p in enumerate(allpts)}
    groups=[[ix[p] for p in vs] for _,vs in cells]
    records=[];best=Q2();pair=None
    for i,(hs,vs) in enumerate(cells):
        ds=[]
        for a,b in it.combinations(vs,2):
            delta=sub(a,b);ds.append((dot(delta,delta),[ix[a],ix[b]]))
        d2=max(t[0] for t in ds);pairs=[p for d,p in ds if d==d2]
        if d2>best:best,pair=d2,{'cell':i+1,'pair':pairs[0]}
        vol=volume(hs,vs);lo,hi=sqrt_bracket(d2)
        records.append({'cell':i+1,'memberships':groups[i],
                        'diameter_squared':d2.record(),'diameter_squared_expression':d2.expression(),
                        'diameter_interval':[lo,hi],'maximizing_pairs':pairs,
                        'volume':vol.record(),'volume_decimal':vol.decimal(),
                        'diameter_bound_proved':d2<bound*bound})
    summed=sum((volume(hs,vs) for hs,vs in cells),Q2())
    check(summed==target_volume,'Cell volumes do not add to the target exactly.')
    interfaces=[]
    for i,j in it.combinations(range(4),2):
        n=sub(scores[j],scores[i]);face=[p for p in cells[i][1] if not dot(n,p)]
        dim=dimension(face);check(dim==2,'Missing two-dimensional interface.')
        # Pairwise constraints have opposite normals; intersection is in this plane.
        check(all(all(dot(m,p)<=h for m,h in cells[j][0]) for p in face),'Inconsistent shared face.')
        interfaces.append({'cells':[i+1,j+1],'dimension':dim,'vertices':[ix[p] for p in face]})
    low,up=sqrt_bracket(best)
    return {'format':'six_plane_exact_quadratic_v1','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'source_file':Path(source).name,'target':'positive_three_truncation_ideal',
            'score_grid_denominator':denominator,'score_numerators':[[str(x) for x in s] for s in scores],
            'max_score_coefficient_rounding_error':str(error),
            'score_convention':'a0=0; ai=score_numerators[i]/score_grid_denominator; cell i maximizes ai dot x',
            'plane_interfaces':planes,'fixed_count':23,'point_count':len(allpts),'groups':groups,
            'points_exact':[[q.record() for q in p] for p in allpts],
            'points_decimal_display':[[q.decimal() for q in p] for p in allpts],
            'cells':records,'maximum_diameter_squared':best.record(),
            'maximum_diameter_squared_expression':best.expression(),
            'maximum_diameter_interval':[low,up],'maximizing_pair':pair,
            'diameter_annihilating_polynomial_descending':integer_polynomial(best),
            'diameter_polynomial_note':'Primitive integer coefficients for a polynomial vanishing at D; not necessarily its minimal polynomial.',
            'diameter_bound':str(bound),'strict_diameter_bound_proved':best<bound*bound,
            'target_volume':target_volume.record(),'sum_cell_volumes':summed.record(),'exact_volume_identity':summed==target_volume,
            'interfaces':interfaces,'covers_ideal_target_exactly':True,'pairwise_interiors_disjoint':True,
            'coverage_reason':'Each point maximizes at least one score. Opposite pairwise score inequalities separate cell interiors. Exact volume identity and six 2-D interfaces checked independently.',
            'arithmetic':'Q(sqrt(2)), exact rational coefficients and signs. Decimal displays do not decide geometry.',
            'scope':'Rational approximation of a saved score solution. Not a proof of global optimality or an exact recovery of the numerical optimizer.'}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path,help='Saved six-plane JSON with four scores.')
    ap.add_argument('--denominator',type=int,default=10**12)
    ap.add_argument('--bound',type=F,default=F('0.97288131961'))
    ap.add_argument('--output',type=Path,default=Path('six_plane_exact.json'))
    ap.add_argument('--export',type=Path,help='Optional five-line rounded display file; NOT the exact certificate.')
    args=ap.parse_args(argv)
    try:
        if args.output.resolve()==args.source.resolve():raise ValueError('Do not overwrite the source.')
        if args.bound<=0:raise ValueError('Bound must be positive.')
        r=certify(args.source,args.denominator,args.bound)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n')
        if args.export:
            if args.export.resolve() in [args.source.resolve(),args.output.resolve()]:raise ValueError('Distinct export path required.')
            args.export.parent.mkdir(parents=True,exist_ok=True)
            p=[[float(v) for v in row] for row in r['points_decimal_display']]
            args.export.write_text('4\n'+r['maximum_diameter_interval'][1]+'\n'+str(len(p))+'\n'+json.dumps(r['groups'])+'\n'+json.dumps(p)+'\n')
        print('Exact four-cell partition of the ideal target: PASS')
        print('D in',r['maximum_diameter_interval'])
        print('Strict bound proved:',r['strict_diameter_bound_proved'])
        print('Certificate:',args.output)
        return 0 if r['strict_diameter_bound_proved'] else 1
    except (ValueError,KeyError,TypeError,OSError,ArithmeticError) as e:
        print('ERROR:',e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
