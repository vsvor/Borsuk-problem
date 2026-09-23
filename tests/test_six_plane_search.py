#!/usr/bin/env python3
"""Regression tests: python -m unittest -v test_six_plane_search.py"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import HalfspaceIntersection, ConvexHull
import six_plane_search as s

class SixPlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target=s.make_target()
        cls.rng=np.random.default_rng(137)
    def test_target(self):
        P=self.target
        self.assertEqual(len(P.vertices),23)
        self.assertEqual(len(P.A),15)
        self.assertEqual(len(P.edges),36)
        self.assertAlmostEqual(P.volume,3.5-2*np.sqrt(2),13)
        self.assertTrue(np.allclose(np.linalg.norm(P.A,axis=1),1))
        self.assertTrue(np.all(P.b==.5))
    def test_symmetric_target(self):
        P=s.make_target('symmetric')
        self.assertEqual(len(P.vertices),32)
        self.assertEqual(len(P.A),18)
        self.assertAlmostEqual(P.volume,7-4.5*np.sqrt(2),13)
    def test_score_derivatives(self):
        t=s.random_start(self.rng); v,j=s.score_matrix(t)
        for k in range(8):
            h=np.zeros(8); h[k]=1e-6
            fd=(s.score_matrix(t+h)[0]-s.score_matrix(t-h)[0])/2e-6
            self.assertLess(np.max(abs(fd-j[:,:,k])),2e-9)
    def test_plane_derivatives(self):
        t=s.random_start(self.rng); v,j=s.score_matrix(t); n,dn=s.planes_from_scores(v,j)
        for k in range(8):
            h=np.zeros(8); h[k]=1e-6
            fp=s.planes_from_scores(s.score_matrix(t+h)[0])[0]
            fm=s.planes_from_scores(s.score_matrix(t-h)[0])[0]
            self.assertLess(np.max(abs((fp-fm)/2e-6-dn[:,:,k])),2e-9)
    def test_random_four_cells(self):
        for _ in range(20):
            t=s.random_start(self.rng,.5)
            n,_=s.planes_from_scores(s.score_matrix(t)[0])
            c=s.check_topology(n,self.target)
            self.assertTrue(c['ok'],c)
            self.assertLess(c['relative_volume_error'],1e-10)
    def test_independent_planes_rejected(self):
        for _ in range(10):
            n=self.rng.normal(size=(6,3))
            c=s.check_topology(n,self.target)
            self.assertFalse(c['ok'],c)
    def test_vertices_match_independent_halfspaces(self):
        for _ in range(10):
            t=s.random_start(self.rng); n,dn=s.planes_from_scores(*s.score_matrix(t))
            for i in range(4):
                N,dN=s.cell_planes(n,dn,i)
                p,_=s.cell_vertices(self.target,N,dN)
                ray=np.linalg.solve(N,-np.ones(3)); ray/=np.linalg.norm(ray)
                inside=.1*ray
                A=np.vstack((self.target.A,N)); b=np.r_[self.target.b,np.zeros(3)]
                q=HalfspaceIntersection(np.c_[A,-b],inside).intersections
                self.assertLess(directed_hausdorff(q,p)[0],1e-8)
                self.assertLess(directed_hausdorff(p,q)[0],1e-8)
    def test_objective_derivatives(self):
        # Generic points, away from topology transitions and diameter ties.
        for _ in range(3):
            t=s.random_start(self.rng)
            obj=s.Objective(self.target,8)
            value,j=obj.evaluate(t)
            for k in range(8):
                h=np.zeros(8); h[k]=1e-7
                fp=s.Objective(self.target,8).evaluate(t+h)[0]
                fm=s.Objective(self.target,8).evaluate(t-h)[0]
                fd=(fp-fm)/2e-7
                self.assertLess(np.max(abs(fd-j[:,k])),1e-6)
    def test_plane_scaling(self):
        t=s.random_start(self.rng); n,_=s.planes_from_scores(s.score_matrix(t)[0])
        a=s.check_topology(n); b=s.check_topology(n*np.arange(1,7)[:,None])
        self.assertTrue(a['ok'] and b['ok'])
        self.assertAlmostEqual(a['diameter'],b['diameter'],12)
    def test_local_solver_export_and_checkpoints(self):
        t=s.random_start(np.random.default_rng(2))
        r=s.optimize_once(t,self.target,maxiter=80)
        self.assertTrue(r['valid'],r)
        self.assertLessEqual(r['diameter'],r['initial_diameter']+1e-9)
        p,g,d=s.export_partition(r,self.target)
        self.assertEqual(set.union(*(set(q) for q in g)),set(range(len(p))))
        self.assertTrue(np.allclose(p[:23],self.target.vertices))
        self.assertAlmostEqual(d,r['diameter'],10)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); s.save_record(r,self.target,path); s.save_record(r,self.target,path)
            self.assertEqual(set(q.name for q in path.iterdir()),
                             {'best.txt','best.json','previous_best.txt','previous_best.json'})
            saved=json.loads((path/'best.json').read_text())
            check=s.check_topology(np.array(saved['plane_normals']),self.target)
            self.assertTrue(check['ok'])
            lines=(path/'best.txt').read_text().splitlines()
            self.assertEqual(len(lines),5)
    def test_target_early_stop(self):
        t=s.random_start(self.rng)
        r=s.optimize_once(t,target_diameter=2.)
        self.assertTrue(r['valid'])
        self.assertEqual(r['iterations'],0)
        self.assertIn('target reached',r['message'])
    def test_time_limit(self):
        t=s.random_start(self.rng)
        r=s.optimize_once(t,seconds=1e-15,maxiter=10000)
        self.assertTrue(r['valid'])
        self.assertIn('time limit',r['message'])

if __name__=='__main__': unittest.main()
