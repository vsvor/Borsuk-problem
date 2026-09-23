#!/usr/bin/env python3
"""Regression tests; run beside optimizer and r_dod_4_0966.txt."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import optimize_partition as op

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'scripts'


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = op.read_data(ROOT / 'data/original/r_dod_4_0966.txt')
        cls.model = op.build_model(cls.data, 23, 1e-6)
        z0 = cls.model.coordinates(cls.data.points)
        cls.z, cls.info = op.solve_model(cls.model, z0, 2000, 1e-13, 1e-10, 0)
        cls.points = cls.model.positions(cls.z)
        cls.diameter = op.max_diameter(cls.points, cls.data.groups)

    def test_input_not_trusted_as_diameter(self):
        self.assertAlmostEqual(op.max_diameter(self.data.points, self.data.groups), 1.0442005345769627, places=13)
        self.assertGreater(abs(self.data.stored - op.max_diameter(self.data.points, self.data.groups)), 0.07)

    def test_boundary_certificate_and_dimensions(self):
        self.assertEqual(self.model.q, 26)
        self.assertEqual(len(self.model.pairs), 496)
        self.assertEqual(self.model.certificate['triangle_count'], 80)
        self.assertEqual(self.model.certificate['edges'], 120)
        self.assertEqual(self.model.common, 27)

    def test_gradients(self):
        self.assertLess(op.check_derivatives(self.model), 1e-7)

    def test_optimization(self):
        self.assertTrue(self.info['success'])
        self.assertAlmostEqual(self.diameter, 0.96976776186145, places=10)
        np.testing.assert_array_equal(self.points[:23], self.data.points[:23])
        self.assertTrue(op.check_cover(self.model, self.points)['valid_numerically'])

    def test_global_lower_bound(self):
        bound = op.lower_bound(self.model, self.z, self.diameter)
        self.assertTrue(bound['success'])
        self.assertGreaterEqual(bound['diameter_gap_with_allowance'], 0)
        self.assertLess(bound['diameter_gap_with_allowance'], 1e-8)

    def test_random_face_preserving_deformations_cover(self):
        # Deliberately move each boundary node anywhere on its original face/edge.
        # Some of these maps fold. Coverage must still hold, without orientation constraints.
        rng = np.random.default_rng(20260922)
        for _ in range(12):
            points = self.model.reference.copy()
            for j in range(23, len(points)):
                ids = [i for i in range(23)
                       if set(self.model.active[j]).issubset(self.model.active[i])]
                w = rng.dirichlet(np.ones(len(ids)))
                points[j] = w @ self.model.reference[ids]
            z = self.model.coordinates(points)
            check = op.check_cover(self.model, self.model.positions(z))
            self.assertTrue(check['valid_numerically'], check)

    def test_restart_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'optimized.txt'
            op.write_data(file, self.points, self.data.groups, self.diameter)
            saved = op.read_data(file)
            np.testing.assert_array_equal(saved.points, self.points)
            self.assertEqual(saved.groups, self.data.groups)
            model_dict = json.loads(json.dumps(op.model_record(self.model)))
            restart_model = op.build_model(saved, 23, 1e-6, model_dict)
            z = restart_model.coordinates(saved.points)
            np.testing.assert_allclose(restart_model.positions(z), self.points, atol=1e-15, rtol=0)
            self.assertEqual(restart_model.q, 26)

    def test_three_distant_starts(self):
        rng = np.random.default_rng(41)
        for _ in range(3):
            points = self.model.reference.copy()
            for j in range(23, len(points)):
                ids = [i for i in range(23)
                       if set(self.model.active[j]).issubset(self.model.active[i])]
                points[j] = rng.dirichlet(np.ones(len(ids))) @ self.model.reference[ids]
            start = self.model.coordinates(points)
            z, result = op.solve_model(self.model, start, 2000, 1e-12, 1e-10, 0)
            self.assertAlmostEqual(op.max_diameter(self.model.positions(z), self.data.groups),
                                   self.diameter, places=8)

    def test_malformed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'bad.txt'
            file.write_text('4\n1.0\n')
            with self.assertRaises(op.InputError):
                op.read_data(file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
