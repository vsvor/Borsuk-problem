from fractions import Fraction as F
from pathlib import Path
from decimal import Decimal,localcontext
import json
import random
import unittest
from six_plane_symbolic import Q2,certify,target_planes,poly_vertices,volume,dimension,sqrt_bracket,integer_polynomial
ROOT=Path(__file__).resolve().parents[1]
class QuadraticTests(unittest.TestCase):
    def test_arithmetic(self):
        r=Q2(0,1)
        self.assertEqual(r*r,2)
        self.assertEqual(1/(1+r),r-1)
        self.assertEqual((1+r)**2,3+2*r)
        self.assertEqual(hash(Q2(3)),hash(3))
        with self.assertRaises(ZeroDivisionError):_ = r/Q2()
    def test_sign_tiny_nonzero(self):
        # Pell-convergent errors smaller than binary floating-point precision.
        a,b=1,1
        for k in range(60):
            x=Q2(a,-b)
            self.assertEqual(x.sign(),1 if a*a-2*b*b>0 else -1)
            a,b=a+2*b,a+b
    def test_sign_random_decimal_crosscheck(self):
        rng=random.Random(2)
        with localcontext() as c:
            c.prec=80
            for _ in range(300):
                a,b=F(rng.randint(-10000,10000),73),F(rng.randint(-10000,10000),91)
                x=Q2(a,b);d=Decimal(a.numerator)/Decimal(a.denominator)+Decimal(2).sqrt()*Decimal(b.numerator)/Decimal(b.denominator)
                self.assertEqual(x.sign(),(d>0)-(d<0))
    def test_exact_target(self):
        h=target_planes();v=poly_vertices(h)
        self.assertEqual(len(v),23);self.assertEqual(dimension(v),3)
        self.assertEqual(volume(h,v),Q2(F(7,2),-2))
    def test_sqrt_bounds(self):
        for x in [Q2(0),Q2(1),Q2(2),Q2(3,-2),Q2(F(1,3),F(1,7))]:
            lo,hi=sqrt_bracket(x,18)
            self.assertLessEqual(Q2(F(lo)**2),x);self.assertGreaterEqual(Q2(F(hi)**2),x)
            self.assertLessEqual(F(hi)-F(lo),F(1,10**18))
    def test_rational_scores_certificate(self):
        source=ROOT/'data/three_truncations/six_planes/model.json'
        r=certify(source)
        self.assertTrue(r['exact_volume_identity'])
        self.assertTrue(r['strict_diameter_bound_proved'])
        self.assertTrue(r['pairwise_interiors_disjoint'])
        self.assertEqual(r['point_count'],42)
        stored=json.loads((ROOT/'symbolic/six_planes/fine_certificate.json').read_text())
        self.assertEqual(r,stored)
    def test_polynomial_identity(self):
        # The saved polynomial is in D, hence evaluate its even coefficients at D^2.
        x=Q2(F(7,13),F(-2,17));co=list(map(int,integer_polynomial(x)))
        self.assertEqual(co[0]*x*x+co[2]*x+co[4],0)
    def test_coarse_fine_are_different(self):
        c=json.loads((ROOT/'symbolic/six_planes/coarse_certificate.json').read_text())
        f=json.loads((ROOT/'symbolic/six_planes/fine_certificate.json').read_text())
        self.assertNotEqual(c['score_numerators'],f['score_numerators'])
        self.assertTrue(c['exact_volume_identity'] and c['strict_diameter_bound_proved'])
if __name__=='__main__':unittest.main()
