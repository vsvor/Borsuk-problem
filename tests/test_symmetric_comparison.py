from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import unittest
import numpy as np
from symmetric_polyhedra import Interval as I, NAMES, ideal_geometry,make_target,rational_outer_halfspaces,rational_inner_halfspaces
import platonic_exact as pe
ROOT=Path(__file__).resolve().parents[1]

class SymmetricTargetsTests(unittest.TestCase):
    def test_expected_vertex_facet_counts(self):
        expected={'six_trd':(32,18),'tetrahedron':(4,4),'cube':(8,6),'octahedron':(6,8),'dodecahedron':(20,12),'icosahedron':(12,20)}
        for name,pair in expected.items():
            t=make_target(name)
            self.assertEqual((t['vertex_count'],t['facet_count']),pair)
            self.assertAlmostEqual(t['inradius'],.5,places=14)
    def test_interval_square_roots(self):
        for n in (2,3,5,7,11):
            s=I.exact(n).sqrt(50)
            self.assertLessEqual(s.lo*s.lo,n);self.assertGreaterEqual(s.hi*s.hi,n)
            self.assertLessEqual(s.hi-s.lo,F(1,10**50))
        self.assertEqual(I.exact(4).sqrt().lo,2)
    def test_interval_arithmetic(self):
        a=I(F(-2),F(3));b=I(F(4),F(5))
        self.assertEqual(a*b,I(F(-10),F(15)))
        self.assertEqual(a/b,I(F(-1,2),F(3,4)))
        with self.assertRaises(ZeroDivisionError):b/a
    def test_outer_interval_vertex_containment(self):
        for name in NAMES:
            vertices,_=ideal_geometry(name)
            hs=rational_outer_halfspaces(name)
            for v in vertices:
                for n,h in hs:
                    self.assertLessEqual(sum((x*y for x,y in zip(n,v)),I.exact(0)).hi,h)
    def test_inner_offset_and_box(self):
        for name in NAMES:
            vertices,planes=ideal_geometry(name)
            inner=rational_inner_halfspaces(name)
            M=math.ceil(max(max(abs(x.lo),abs(x.hi)) for v in vertices for x in v))
            for (q,h),(n,t) in zip(inner,planes):
                error=sum(max(abs(a-b.lo),abs(a-b.hi)) for a,b in zip(q,n))
                self.assertLessEqual(h+M*error,t.lo)
            self.assertEqual(len(inner),len(planes)+6)
    def test_minimum_width_normalization(self):
        for name in NAMES:
            a=make_target(name);b=make_target(name,'min-width')
            self.assertAlmostEqual(b['minimum_width'],1.,places=14)
            factor=math.sqrt(3) if name=='tetrahedron' else 1
            np.testing.assert_allclose(np.array(b['vertices'])*factor,a['vertices'],rtol=0,atol=3e-16)
        self.assertAlmostEqual(make_target('tetrahedron')['tetrahedral_altitude'],2)
    def test_outer_targets_regenerate(self):
        for name in NAMES:
            spec=json.loads((ROOT/f'data/symmetric/{name}/target.json').read_text())
            fresh=make_target(name)
            for key in ('vertices','solid','normalization','rational_outer_halfspaces','vertex_count','facet_count'):
                self.assertEqual(fresh[key],spec[key])
            self.assertAlmostEqual(fresh['inradius'],spec['inradius'],places=14)
            self.assertAlmostEqual(fresh['target_volume'],spec['target_volume'],places=13)
    def test_bad_names(self):
        with self.assertRaises(ValueError):make_target('sphere')
        with self.assertRaises(ValueError):ideal_geometry('cube','unit_diagonal')
    def test_all_new_exact_certificates(self):
        for name in NAMES:
            r=json.loads((ROOT/f'certificates/current/{name}_padded.json').read_text())
            self.assertTrue(r['covers_ideal_target']);self.assertTrue(r['diameter_bound_proved'])
            self.assertTrue(r['full_volume_check']['missing_volume_exact_zero'])
            self.assertEqual(len(r['full_volume_check']['intersections']),15)
    def test_standalone_verifier_on_cube(self):
        from verify_symmetric_cover import verify
        rec=verify(ROOT/'data/symmetric/cube/partition.txt','cube',padding=F('1e-12'),bound=F('1.225'),full=True)
        self.assertTrue(rec['covers_ideal_target']);self.assertTrue(rec['diameter_bound_proved'])
        self.assertTrue(rec['full_volume_check']['missing_volume_exact_zero'])
    def test_no_global_six_trd_claim(self):
        r=json.loads((ROOT/'reports/symmetric_comparison.json').read_text())
        self.assertFalse(r['six_trd_global_optimality_proved'])
        self.assertEqual(r['normalization'],'inradius_1/2')

class ExactPlatonicTests(unittest.TestCase):
    def test_field_relations(self):
        s=pe.Q5(0,1);self.assertEqual(s*s,5);self.assertLess(s,pe.Q5(F(9,4)))
        phi=(1+s)/2;self.assertEqual(phi*phi,phi+1);self.assertEqual(phi*(1/phi),1)
        self.assertEqual(abs(-s),s)
    def test_three_rational_constructions(self):
        for name,square in [('tetrahedron',F(9,4)),('cube',F(3,2)),('octahedron',F(3,2))]:
            r=pe.rational_construction(name)
            self.assertEqual(r['maximum_diameter_squared'],pe.Q5(square).record())
            self.assertTrue(r['coverage_proved']);self.assertTrue(r['convex_interiors_disjoint'])
            self.assertEqual(r['global_lower_bound']['proved'],name!='cube')
    def test_exact_icosahedral_cover(self):
        r=pe.icosahedron_construction()
        self.assertEqual(r['point_count'],35)
        self.assertEqual(r['maximum_diameter_squared'],pe.Q5(F(27,8),F(-9,8)).record())
        self.assertTrue(r['coverage_proved']);self.assertEqual(len(r['boundary_faces']),20)
        self.assertTrue(all(f['missing_projected_area_exact_zero'] for f in r['boundary_faces']))
        self.assertFalse(r['global_lower_bound']['proved'])
    def test_symbolic_records_reproduce(self):
        for name in ('tetrahedron','cube','octahedron','icosahedron'):
            expected=json.loads((ROOT/f'symbolic/platonic/{name}.json').read_text())
            r=pe.icosahedron_construction() if name=='icosahedron' else pe.rational_construction(name)
            self.assertEqual(r,expected)
    def test_independent_quadratic_check(self):
        try:import sympy as s
        except ImportError:self.skipTest('Optional SymPy')
        r=pe.icosahedron_construction();ds=r['maximum_diameter_squared']
        expr=s.Rational(ds['rational'])+s.Rational(ds['sqrt5'])*s.sqrt(5)
        self.assertEqual(s.simplify(expr-(3*(s.sqrt(5)-1)/4)**2),0)
    def test_exact_polygon_gap_and_overlap_detection(self):
        Q=pe.Q5
        square=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1))]
        half=[(Q(F(1,2)),Q(0)),(Q(2),Q(0)),(Q(2),Q(1)),(Q(F(1,2)),Q(1))]
        self.assertEqual(pe.area(pe.intersect2(square,half)),Q(F(1,2)))
if __name__=='__main__':unittest.main()
