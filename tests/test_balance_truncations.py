import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import balance_truncations as bt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'scripts'
EXAMPLES = ROOT/'data/balanced/seeds'

class SyntheticModel:
    q=1
    box=2.
    fixed_lower_squared=0.
    C=np.array([[[1.],[0.],[0.]],[[1.],[0.],[0.]]])
    v=np.array([[1.,0.,0.],[-1.,0.,0.]])
    L=np.array([[1.],[-1.]])
    r=np.array([2.,2.])
    def pair_values(self,z):
        d=self.v+np.einsum('ecq,q->ec',self.C,z)
        return np.sum(d*d,axis=1),2*np.einsum('ec,ecq->eq',d,self.C)


def synthetic_eval(delta, error=1e-14):
    # The unique crossing is 0.007123456789.
    a,b=1.-delta,1.-0.007123456789
    return dict(delta=delta,f_lower=a-b-2*error,f_upper=a-b+2*error,
                dummy={'lower':a-error,'upper':a+error,'selected':0},
                upper={'lower':b-error,'upper':b+error,'selected':0})


class TestArithmetic(unittest.TestCase):
    def test_analytic_bisection(self):
        r=bt.bracket_root(synthetic_eval,0,.02,1e-11)
        self.assertEqual(r['status'],'converged')
        self.assertLessEqual(r['left']['delta'],.007123456789)
        self.assertGreaterEqual(r['right']['delta'],.007123456789)
    def test_uncertain_midpoint_is_not_a_sign(self):
        r=bt.bracket_root(lambda d:synthetic_eval(d,error=1e-5),0,.02,1e-11)
        self.assertEqual(r['status'],'objective_uncertainty_or_equality_plateau')
        self.assertGreater(r['right']['delta']-r['left']['delta'],1e-11)
    def test_no_bracket_is_reported(self):
        r=bt.bracket_root(synthetic_eval,.01,.02,1e-11)
        self.assertEqual(r['status'],'not_bracketed')
    def test_dual_at_nonoptimal_points(self):
        for x in [-1.2,-.01,0.,.015,.9]:
            lb,_=bt.dual_lower_bound(SyntheticModel(),np.array([x]),1e-12)
            self.assertLessEqual(lb,1.+1e-13)
            self.assertGreaterEqual(lb,0.)
            if x == 0.:
                self.assertGreater(lb,.999999999)
    def test_geometry_signs(self):
        A,r,v=bt.family_geometry('dummy',.01,'adjacent',.5)
        self.assertAlmostEqual(r[12],1.)
        self.assertAlmostEqual(r[13],.49)
        self.assertEqual(v[13],-1.)
        A,r,v=bt.family_geometry('upper',.01,'adjacent',.5)
        self.assertAlmostEqual(r[12],.51)
        self.assertAlmostEqual(r[13],.5)
        self.assertEqual(v[12],1.)

class TestActualModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A=bt.Branch(EXAMPLES/'dummy_seed/best.json','dummy')
        cls.B=bt.Branch(EXAMPLES/'upper_seed/best.json','upper')
        cls.options=bt.Options()
    def test_source_targets(self):
        self.assertEqual(self.A.nfixed,23)
        self.assertEqual(self.B.nfixed,26)
        self.assertAlmostEqual(self.A.delta0,0.)
        self.assertAlmostEqual(self.B.delta0,.01)
    def test_topology_and_feasible_transport(self):
        for branch in [self.A,self.B]:
            for d in [0.,.005,.01,.02]:
                V=branch.validate_delta(d,independent=True)
                m=branch.model(d)
                self.assertLess(np.max(m.X@m.target.A.T-m.target.b),1e-12)
                self.assertTrue(m.geometry_check(np.zeros(m.q))['valid_numerically'])
                self.assertEqual(m.groups,branch.groups)
    def test_wrong_seed_rejected(self):
        with self.assertRaises(bt.BalanceError):
            bt.Branch(EXAMPLES/'upper_seed/best.json','dummy')
    def test_topology_change_rejected(self):
        with self.assertRaises(bt.BalanceError):
            self.A.validate_delta(.2,independent=True)
    def test_derivatives(self):
        m=self.B.model(.009)
        rng=np.random.default_rng(123)
        z=rng.normal(size=m.q)*.01
        values,g=m.pair_values(z)
        h=1e-6
        numeric=np.column_stack([(m.pair_values(z+np.eye(m.q)[i]*h)[0]-m.pair_values(z-np.eye(m.q)[i]*h)[0])/(2*h) for i in range(m.q)])
        self.assertLess(np.max(abs(numeric-g)),1e-8)
    def test_reproduced_values(self):
        a=self.A.evaluate(.01,self.options);b=self.B.evaluate(.01,self.options)
        self.assertAlmostEqual(a['diameter'],.9649655753222,11)
        self.assertAlmostEqual(b['diameter'],.9656571899538,11)
        for e in [a,b]:
            self.assertLess(e['gap'],1e-11)
            self.assertTrue(e['geometry']['valid_numerically'])
    def test_checkpoint_reload(self):
        e=self.A.evaluate(.009,self.options)
        rec=self.A.checkpoint(e)
        with TemporaryDirectory() as td:
            p=Path(td)/'seed.json';p.write_text(json.dumps(rec))
            b=bt.Branch(p,'dummy')
            ee=b.evaluate(.009,self.options)
            self.assertAlmostEqual(e['diameter'],ee['diameter'],11)
    def test_text_with_matching_json(self):
        b=bt.Branch(EXAMPLES/'upper_seed/best.txt','upper')
        self.assertEqual(b.groups,self.B.groups)
    def test_multiple_models(self):
        pool=bt.Pool([self.A,self.A],self.options)
        r=pool.evaluate(.01)
        self.assertEqual(r['selected'],0)
        self.assertLessEqual(r['lower'],r['upper'])
    def test_root_precision_and_rounding(self):
        pools=[bt.Pool([b],self.options) for b in [self.A,self.B]]
        def ev(d):
            a,b=[p.evaluate(d) for p in pools]
            return dict(delta=d,dummy=a,upper=b,f_lower=a['lower']-b['upper'],f_upper=a['upper']-b['lower'])
        r=bt.bracket_root(ev,.0088,.0089,1e-11)
        self.assertEqual(r['status'],'converged')
        lo,hi=r['left']['delta'],r['right']['delta']
        self.assertEqual(format(lo,'.8g'),'0.0088114234')
        self.assertEqual(format(hi,'.8g'),'0.0088114234')
        self.assertGreaterEqual(hi-lo,0.)
        self.assertLessEqual(hi-lo,1e-11)

if __name__=='__main__':
    unittest.main(verbosity=2)
