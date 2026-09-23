"""Regression tests; run: python -m unittest -v test_partitions_slsqp.py"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import partitions_slsqp as ps
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'scripts'


def intersection_volume(equations):
    eq = np.vstack(equations)
    a, b = eq[:, :3], -eq[:, 3]
    try:
        c, radius = ps._interior(a, b)
        if radius < 1e-9:
            return 0.
        pts = HalfspaceIntersection(eq, c).intersections
        return float(ConvexHull(pts).volume)
    except (ps.GeometryError, ps.QhullError):
        return 0.


def union_volume(points, groups):
    equations = [ConvexHull(points[g]).equations for g in groups]
    return sum((-1.)**(len(ids) + 1) * intersection_volume([equations[i] for i in ids])
               for n in range(1, len(groups) + 1)
               for ids in itertools.combinations(range(len(groups)), n))


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.groups, _ = ps.read_partition(ROOT / 'data/original/r_dod_4_0966.txt')
        cls.target = ps.Target.from_vertices(cls.points[:23])
        cls.reference = ps.reference_from_input(cls.target, cls.points, cls.groups)
        cls.model = ps.Model(cls.target, cls.reference)
        cls.cfg = ps.SearchConfig(seed=9, quiet=True)
        cls.z, cls.info = ps.solve_model(cls.model, cls.cfg)

    def test_01_known_optimum_and_independent_volume(self):
        record = self.model.record(self.z)
        self.assertAlmostEqual(record['diameter'], 0.969767761861449, places=10)
        self.assertTrue(record['geometry']['valid_numerically'])
        volume = union_volume(self.model.points(self.z), self.model.groups)
        self.assertAlmostEqual(volume, self.target.volume, places=10)

    def test_02_analytic_pair_gradients(self):
        m = self.model
        rng = np.random.default_rng(61)
        z = rng.normal(0, .015, m.q)
        values, jac = m.pair_values(z)
        error = 0.
        for i in range(m.q):
            h = np.zeros(m.q); h[i] = 1e-6
            fd = (m.pair_values(z + h)[0] - m.pair_values(z - h)[0]) / 2e-6
            error = max(error, float(np.max(abs(fd - jac[:, i]))))
        self.assertLess(error, 2e-8)

    def test_03_bound_not_above_known_optimum(self):
        lb = ps.tangent_lower_bound(self.model, self.z) * self.target.scale
        self.assertLessEqual(lb, 0.969767761861449 + 1e-12)
        self.assertGreater(lb, 0.96976775)
        lb0 = ps.tangent_lower_bound(self.model, np.zeros(self.model.q)) * self.target.scale
        self.assertLessEqual(lb0, 0.969767761861449 + 1e-10)

    def test_04_weighted_diagram_and_physical_scale(self):
        cfg = ps.SearchConfig(seed=23, seed_method='random', power_weight_scale=.006)
        sites, weights, _ = ps.generate_sites(self.target, cfg, 1)
        ref = ps.build_diagram(self.target, sites, weights)
        m = ps.Model(self.target, ref)
        z, info = ps.solve_model(m, cfg)
        self.assertTrue(m.geometry_check(z)['valid_numerically'])
        volume = union_volume(m.points(z), m.groups)
        self.assertAlmostEqual(volume, self.target.volume, places=9)
        shifted_target = ps.Target.from_vertices(self.points[:23] * 3.7 + [4, -9, 1])
        self.assertAlmostEqual(shifted_target.scale, self.target.scale * 3.7, places=10)

    def test_05_five_cells_no_common_vertex_and_large_deformation(self):
        # Reference power complex, not the special four-cell common-point proof.
        cfg = ps.SearchConfig(parts=5, seed=119, seed_method='random')
        sites, weights, _ = ps.generate_sites(self.target, cfg, 0)
        ref = ps.build_diagram(self.target, sites, weights)
        self.assertFalse(set.intersection(*(set(g) for g in ref['groups'])))
        m = ps.Model(self.target, ref)
        rng = np.random.default_rng(63)
        p = m.X.copy()
        for j in range(len(self.target.V), len(p)):
            active = ref['active'][j]
            ids = np.flatnonzero(np.max(abs(self.target.V @ self.target.A[active].T - self.target.b[active]), axis=1) < 1e-8) if active else np.arange(len(self.target.V))
            weights = rng.dirichlet(np.full(len(ids), .25))
            p[j] = weights @ self.target.V[ids]
        z = m.coordinates(p)
        self.assertLess(float(np.max(abs(m.points(z) - p))), 1e-9)
        self.assertTrue(m.geometry_check(z)['valid_numerically'])
        self.assertAlmostEqual(union_volume(p, m.groups), self.target.volume, places=8)

    def test_06_record_round_trip_retains_original_incidence(self):
        record = dict(**self.model.record(self.z), target=self.target.record())
        model, z = ps.model_from_record(self.target, record)
        self.assertEqual(model.q, self.model.q)
        self.assertEqual(model.reference['active'], self.reference['active'])
        self.assertLess(np.max(abs(model.points(z) - self.model.points(self.z))), 1e-12)

    def test_07_canonical_topology_signature(self):
        ref = {**self.reference, 'groups': self.groups[::-1]}
        m = ps.Model(self.target, ref)
        self.assertEqual(m.signature(), self.model.signature())

    def test_08_run_target_and_global_target(self):
        cfg = ps.SearchConfig(run_target=1.2)
        z, info = ps.solve_model(self.model, cfg)
        self.assertEqual(info['status'], 'run_target')
        self.assertEqual(info['iterations'], 0)
        cfg = ps.SearchConfig(target_diameter=.98)
        z, info = ps.solve_model(self.model, cfg)
        self.assertEqual(info['status'], 'global_target_candidate')
        self.assertLessEqual(self.model.record(z)['diameter'], .98)

    def test_09_bound_pruning_and_optional_heuristic(self):
        z, info = ps.solve_model(self.model, ps.SearchConfig(bound_start=1, bound_every=1),
                                incumbent=lambda: .8)
        self.assertIn(info['status'], ['fixed_pair_pruned', 'lower_bound_pruned'])
        self.assertGreaterEqual(info['lower_bound'], .8)
        z, info = ps.solve_model(self.model, ps.SearchConfig(reject_above=.8, reject_after=1))
        self.assertEqual(info['status'], 'heuristic_rejection')
        self.assertTrue(self.model.geometry_check(z)['valid_numerically'])

    def test_10_repair_and_expired_deadline(self):
        z = self.model.repair(np.random.default_rng(81).normal(size=self.model.q) * 100)
        self.assertLessEqual(np.max(self.model.L @ z - self.model.r), 1e-12)
        z, info = ps.solve_model(self.model, ps.SearchConfig(), deadline=0.)
        self.assertEqual(info['status'], 'time_limit')
        self.assertTrue(self.model.geometry_check(z)['valid_numerically'])

    def test_11_spherical_and_packing_seed_stages(self):
        rng = np.random.default_rng(12)
        p = ps.sphere_directions(5, rng)
        self.assertLess(np.max(abs(np.linalg.norm(p, axis=1) - 1)), 1e-12)
        p = ps.packing_sites(self.target, rng, 4, 60)
        self.assertLess(np.max(p @ self.target.A.T - self.target.b), 0.)
        self.assertGreater(np.min(np.linalg.norm(p[:, None] - p[None, :], axis=2) + np.eye(4) * 99), .1)

    def test_12_resume_and_directory_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ps.SearchConfig(starts=3, output_dir=tmp, quiet=True, seed=3)
            result = ps.run_search(self.points[:23], cfg, (self.points, self.groups))
            self.assertAlmostEqual(result['best_diameter'], .969767761861449, places=9)
            with self.assertRaises(ValueError):
                ps.run_search(self.points[:23], cfg)
            state = json.loads((Path(tmp) / 'state.json').read_text())
            self.assertEqual(state['next_start_id'], 3)
            cfg = ps.SearchConfig(starts=2, output_dir=tmp, quiet=True, resume=True)
            result = ps.run_search(self.points[:23], cfg)
            state = json.loads((Path(tmp) / 'state.json').read_text())
            self.assertEqual(state['next_start_id'], 5)
            self.assertEqual(state['seed'], 3)
            best = json.loads((Path(tmp) / 'best.json').read_text())
            p, g, d = ps.read_partition(Path(tmp) / 'best.txt')
            self.assertAlmostEqual(d, best['diameter'], places=13)
            with ps._directory_lock(Path(tmp)):
                with self.assertRaises(ValueError):
                    with ps._directory_lock(Path(tmp)):
                        pass

    def test_13_parallel_cli_and_target_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(HERE / 'partitions_slsqp.py'), str(ROOT / 'data/original/r_dod_4_0966.txt'),
                   '--workers', '2', '--starts', '12', '--seed', '7', '--output-dir', tmp]
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('completed=12', result.stdout)
            state = json.loads((Path(tmp) / 'state.json').read_text())
            self.assertEqual(state['next_start_id'], 12)
            result = subprocess.run(cmd + ['--resume', '--target', '.98'], text=True, capture_output=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('target_reached', result.stdout)
            self.assertIn('completed=0', result.stdout)

    def test_14_no_torch_dependency_and_input_format(self):
        self.assertNotIn('torch', sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.txt'
            ps.write_partition(path, self.points, self.groups)
            points, groups, diameter = ps.read_partition(path)
            self.assertEqual(groups, self.groups)
            self.assertAlmostEqual(diameter, 1.044200534576963, places=12)
            self.assertTrue(np.array_equal(points, self.points))

    def test_15_one_cell_fixed_model(self):
        ref = ps.build_diagram(self.target, np.zeros((1, 3)))
        m = ps.Model(self.target, ref)
        z, info = ps.solve_model(m, ps.SearchConfig(parts=1))
        self.assertEqual(m.q, 0)
        self.assertEqual(info['status'], 'fixed')
        self.assertAlmostEqual(m.record(z)['diameter'], self.target.scale, places=11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
