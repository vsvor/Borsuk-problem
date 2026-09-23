#!/usr/bin/env python3
"""Exact algebraic bounds for the three-edge diameter bottleneck.

Dependency: sympy. No optimization solver is used. All comparisons are exact
in Q(sqrt(2)); the printed decimal evaluations are for convenience only.
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

def positive(a):
    """Exact sign test for an element of Q(sqrt(2))."""
    a=sp.expand(a)
    b=a.coeff(sp.sqrt(2)); c=sp.simplify(a-b*sp.sqrt(2))
    if not (b.is_Rational and c.is_Rational):
        raise ValueError('Not a quadratic-field element: '+str(a))
    if b>=0:
        return bool(c>0 or (b>0 and (c>=0 or 2*b*b>c*c)))
    return bool(c>0 and c*c>2*b*b)

def certificate():
    u,v,w=sp.symbols('u v w',real=True)
    t=sp.sqrt(2)/2; a=t-sp.Rational(1,2)
    p=sp.Matrix([sp.Rational(1,2),-a,-u])
    q=sp.Matrix([-v,sp.Rational(1,2),a])
    r=sp.Matrix([w-t,-w,-w])
    f=[sp.expand((x-y).dot(x-y)) for x,y in ((p,q),(p,r),(q,r))]
    weights=[sp.Rational(2258922657,10**10),sp.Rational(4174331616,10**10),sp.Rational(3566745727,10**10)]
    assert sum(weights)==1 and all(z>0 for z in weights)
    F=sp.expand(sum(l*z for l,z in zip(weights,f)))
    z=[u,v,w]; H=sp.hessian(F,z); zero=dict(zip(z,[0,0,0]))
    h=sp.Matrix([sp.diff(F,x).subs(zero) for x in z]); k=F.subs(zero)
    assert all(H[:j,:j].det()>0 for j in (1,2,3))
    z_star=-H.inv()*h
    lower_squared=sp.expand(k-(h.T*H.inv()*h)[0]/2)
    lower_d=sp.Rational(96355124245749,10**14)
    upper_d=sp.Rational(96355124245750,10**14)
    assert positive(lower_squared-lower_d**2)
    trial={u:sp.Rational('0.095317360851758895'),v:sp.Rational('0.080491718419628059'),w:sp.Rational('0.258972961142285614')}
    trial_values=[sp.expand(ff.subs(trial)) for ff in f]
    assert all(positive(upper_d**2-ff) for ff in trial_values)
    # Feasibility on the actual bounded target edges of the five-cut target.
    assert all(positive(a-abs(trial[z])) for z in [u,v])
    assert positive(trial[w]-a) and positive(t/2-trial[w])
    return {'weights_exact':[str(z) for z in weights],
            'weighted_minimum_squared_exact':str(lower_squared),
            'weighted_minimum_diameter_decimal':str(sp.N(sp.sqrt(lower_squared),45)),
            'weighted_minimizer_exact':[str(sp.expand(z)) for z in z_star],
            'canonical_trial_exact':{str(z):str(val) for z,val in trial.items()},
            'diameter_lower_bound_exact':str(lower_d),
            'diameter_upper_bound_exact':str(upper_d),
            'certified_interval_decimal':['0.96355124245749','0.96355124245750'],
            'strict_comparisons_verified':True,
            'scope':'Three-point minimax on specified affine lines (and the stated five-cut edge segments), not a global lower bound for all four-part covers.'}

if __name__=='__main__':
    record=certificate()
    out=Path(__file__).with_name('triangle_certificate.json')
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
