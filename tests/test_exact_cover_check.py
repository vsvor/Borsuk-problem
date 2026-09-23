"""Regression tests: python3 -m unittest -v test_exact_cover_check.py"""
import itertools as it
import json
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
import exact_cover_check as c


def box(lo, hi):
    return [c.from_xyz(v) for v in it.product(*[(F(a), F(b)) for a, b in zip(lo, hi)])]


def data(cells):
    ps, groups = [], []
    for cell in cells:
        groups.append(list(range(len(ps), len(ps)+len(cell))))
        ps.extend(cell)
    return ps, groups


def unit_cube():
    v = box([0]*3, [1]*3)
    return c.hull_planes(v), v


class ExactTests(unittest.TestCase):
    def test_cube_hull_and_vertices(self):
        h, v = unit_cube()
        self.assertEqual(len(h), 6)
        self.assertEqual(set(c.poly_vertices(h)), set(v))
        self.assertEqual(c.diameter_squared(v)[0], F(3))

    def test_full_dimension(self):
        p = [c.from_xyz(x) for x in [[0,0,0],[1,0,0],[0,1,0],[1,1,0]]]
        self.assertFalse(c.full_dimension(p))
        with self.assertRaises(ValueError):
            c.hull_planes(p)

    def test_two_closed_halfcubes_partition(self):
        h, v = unit_cube()
        cells = [box([0,0,0], [F(1,2),1,1]), box([F(1,2),0,0],[1,1,1])]
        p, g = data(cells)
        r = c.verify(p,g,h,v,overlaps=True)
        self.assertTrue(r['covers_checked_target'])
        self.assertTrue(r['clipped_cells_form_partition_up_to_shared_boundaries'])
        self.assertFalse(r['pairwise_overlaps'][0]['interiors_overlap_inside_target'])

    def test_four_quadrants_shared_edges(self):
        h, v = unit_cube()
        cells = [box([F(x,2),F(y,2),0], [F(x+1,2),F(y+1,2),1]) for x,y in it.product(range(2),repeat=2)]
        p,g = data(cells)
        r=c.verify(p,g,h,v,overlaps=True)
        self.assertTrue(r['covers_checked_target'])
        self.assertTrue(r['clipped_cells_form_partition_up_to_shared_boundaries'])

    def test_true_overlaps(self):
        h, v = unit_cube()
        p,g = data([box([0,0,0],[F(3,5),1,1]), box([F(2,5),0,0],[1,1,1])])
        r=c.verify(p,g,h,v,overlaps=True)
        self.assertTrue(r['covers_checked_target'])
        self.assertTrue(r['pairwise_overlaps'][0]['interiors_overlap_inside_target'])
        self.assertFalse(r['clipped_cells_form_partition_up_to_shared_boundaries'])

    def test_sub_float_gap_is_not_ignored(self):
        h, v = unit_cube()
        e=F(1,10**40)
        p,g=data([box([0,0,0],[F(1,2)-e,1,1]), box([F(1,2)+e,0,0],[1,1,1])])
        r=c.verify(p,g,h,v)
        self.assertFalse(r['covers_checked_target'])
        w=tuple(map(int,r['boundary']['witness']['homogeneous_exact']))
        self.assertTrue(F(1,2)-e < F(w[0],w[3]) < F(1,2)+e)
        # Gap closes under the explicitly supplied geometric expansion.
        r=c.verify(p,g,h,v,padding=e)
        self.assertTrue(r['covers_checked_target'])

    def test_boundary_only_hollow_shell_is_inconclusive(self):
        h, v=unit_cube()
        cells=[]
        for i in range(3):
            for side in range(2):
                lo=[0,0,0]; hi=[1,1,1]
                if side: lo[i]=F(3,4)
                else: hi[i]=F(1,4)
                cells.append(box(lo,hi))
        p,g=data(cells)
        r=c.verify(p,g,h,v)
        self.assertTrue(r['boundary']['covered'])
        self.assertIsNone(r['covers_checked_target'])
        self.assertIsNone(r['common_point'])

    def test_exact_volumes_and_full_coverage(self):
        h,v=unit_cube()
        self.assertEqual(c.volume(h,v),F(1))
        ps,gs=data([box([0,0,0],[F(3,5),1,1]),box([F(2,5),0,0],[1,1,1])])
        r=c.verify(ps,gs,h,v,full=True)
        self.assertTrue(r['covers_checked_target'])
        self.assertTrue(r['full_volume_check']['missing_volume_exact_zero'])
        # A hollow shell covers the boundary, but not the interior.
        cells=[]
        for i in range(3):
            for side in range(2):
                lo=[0,0,0];hi=[1,1,1]
                if side:lo[i]=F(3,4)
                else:hi[i]=F(1,4)
                cells.append(box(lo,hi))
        ps,gs=data(cells)
        r=c.verify(ps,gs,h,v,full=True)
        self.assertFalse(r['covers_checked_target'])
        self.assertEqual(r['full_volume_check']['missing_volume_approx'],0.125)

    def test_padding_diameter_is_recomputed(self):
        h, v = unit_cube()
        p,g=data([v])
        e=F(1,100)
        r=c.verify(p,g,h,v,padding=e,bound=F('1.74'))
        self.assertTrue(r['covers_checked_target'])
        self.assertEqual(F(r['cells'][0]['checked_diameter_squared_exact']),3*(1+2*e)**2)
        self.assertFalse(r['diameter_bound_proved'])

    def test_tetrahedron(self):
        vs=[c.from_xyz(p) for p in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]
        hs=c.hull_planes(vs)
        p,g=data([vs])
        self.assertEqual(len(hs),4)
        self.assertTrue(c.verify(p,g,hs,vs)['covers_checked_target'])

    def test_clipping_plane_coincident_with_face(self):
        axes=(0,1)
        poly=c.face_polygon([c.from_xyz(v) for v in [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]],axes)
        self.assertEqual(c.subtract(poly,[(0,0,1,0)],axes),[])
        self.assertEqual(c.subtract(poly,[(0,0,-1,0)],axes),[])
        pieces=c.subtract(poly,[(2,0,0,1)],axes)
        self.assertEqual(len(pieces),1)
        self.assertTrue(all(c.val((2,0,0,1),p)>=0 for p in pieces[0]))

    def test_input_decimal_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'in.txt'
            file.write_text('1\n1.0\n4\n[[0,1,2,3]]\n[[0.1,0,0],[1,0,0],[0,1,0],[0,0,1]]\n')
            p,g,_=c.load_partition(file)
            self.assertEqual(F(p[0][0],p[0][3]),F(1,10))

    def test_sqrt_upper_bound(self):
        for s in [F(0),F(1),F(2),F('0.96539917865')**2,F('123.45678987654')]:
            u=F(c.sqrt_upper(s))
            self.assertGreaterEqual(u*u,s)
            if u>0:
                self.assertLess((u-F(1,10**15))**2,s)

    def test_analytic_inner_and_outer_targets(self):
        for a,b in [(F(1,2),F('0.0088')),(F('0.0088'),F(0))]:
            h,v,m=c.rhombic_target(a,b,'adjacent')
            t=F(m['sqrt_half_enclosure_exact'])
            self.assertGreater(2*t*t,1)
            hi,vi,mi=c.rhombic_target(a,b,'adjacent',enclosure='inner')
            ti=F(mi['sqrt_half_enclosure_exact'])
            self.assertLess(2*ti*ti,1)
            self.assertTrue(all(c.val(g,p)<=0 for p in vi for g in h))

    def test_real_saved_covers_with_padding(self):
        root=Path(__file__).resolve().parents[1]/'data/balanced'
        for name,a,b in [('dummy',F(1,2),F('0.008811423368752003')),
                         ('upper',F('0.008811423368752003'),F(0))]:
            path=root/name/'partition.txt'
            if not path.exists():
                self.skipTest('Example data not installed.')
            p,g,_=c.load_partition(path)
            h,v,_=c.rhombic_target(a,b,'adjacent')
            r=c.verify(p,g,h,v,padding=F('1e-12'),bound=F('0.96539917866'))
            self.assertTrue(r['covers_checked_target'])
            self.assertTrue(r['diameter_bound_proved'])

    def test_real_raw_gap_in_inner_target(self):
        root=Path(__file__).resolve().parents[1]/'data/balanced'
        for name,a,b in [('dummy',F(1,2),F('0.008811423368752003')),
                         ('upper',F('0.008811423368752003'),F(0))]:
            path=root/name/'partition.txt'
            if not path.exists(): self.skipTest('Example data not installed.')
            p,g,_=c.load_partition(path)
            h,v,_=c.rhombic_target(a,b,'adjacent',enclosure='inner')
            r=c.verify(p,g,h,v)
            self.assertFalse(r['covers_checked_target'])
            w=tuple(map(int,r['boundary']['witness']['homogeneous_exact']))
            xyz=[F(t,w[3]) for t in w[:3]]
            self.assertTrue(all(2*(abs(xyz[i])+abs(xyz[j]))**2 <= 1 for i,j in [(0,1),(0,2),(1,2)]))
            self.assertTrue(-F(1,2)+b <= xyz[0] <= F(1,2)+a)
            self.assertTrue(xyz[1]>=-F(1,2) and xyz[2]>=-F(1,2))

if __name__ == '__main__':
    unittest.main()
