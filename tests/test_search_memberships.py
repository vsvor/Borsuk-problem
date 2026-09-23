#!/usr/bin/env python3
"""Run beside the scripts and r_dod_4_optimized.{txt,json}: python -m unittest -v test_search_memberships.py"""
from __future__ import annotations
import json
import itertools
from pathlib import Path
import unittest
import numpy as np
import optimize_partition as op
import search_memberships as sm

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'scripts'

class MembershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = op.read_data(ROOT / 'data/three_truncations/best_cover/partition.txt')
        saved = json.loads((ROOT / 'data/three_truncations/best_cover/model.json').read_text())['model']
        cls.base = op.build_model(data, 23, 1e-6, saved)
        cls.z = cls.base.coordinates(data.points)
        cls.d = op.max_diameter(cls.base.positions(cls.z), data.groups)

    def test_identical_model(self):
        m = sm.model_with_memberships(self.base, self.base.data.groups)
        self.assertEqual({tuple(p) for p in m.pairs}, {tuple(p) for p in self.base.pairs})
        np.testing.assert_array_equal(m.B, self.base.B)
        np.testing.assert_array_equal(m.reference, self.base.reference)
        self.assertEqual(m.q, 26)

    def test_common_point_is_protected(self):
        groups = [list(g) for g in self.base.data.groups]
        groups[0].remove(self.base.common)
        trial = sm.try_membership_change(self.base, self.z, groups)
        self.assertEqual(trial.status, 'certificate_rejected')
        self.assertIn('common point', trial.report['reason'])

    def test_unowned_triangle_is_rejected(self):
        groups = [list(g) for g in self.base.data.groups]
        groups[2].remove(40)
        trial = sm.try_membership_change(self.base, self.z, groups)
        self.assertEqual(trial.status, 'certificate_rejected')
        self.assertIn('ownerless', trial.report['reason'])

    def test_redundant_membership_improves(self):
        groups = [list(g) for g in self.base.data.groups]
        groups[1].append(0)
        worse = sm.model_with_memberships(self.base, groups)
        before = op.max_diameter(worse.positions(self.z), groups)
        self.assertGreater(before, self.d + .1)
        trial = sm.try_membership_change(worse, self.z, self.base.data.groups)
        self.assertEqual(trial.status, 'improved')
        self.assertLess(trial.diameter, self.d + 1e-8)
        self.assertTrue(trial.report['coverage']['valid_numerically'])
        self.assertIn(0, worse.data.groups[1]) # Inputs have not been mutated.
        self.assertNotIn(0, self.base.data.groups[1])

    def test_automatic_search_finds_redundancy(self):
        groups = [list(g) for g in self.base.data.groups]
        groups[1].append(0)
        worse = sm.model_with_memberships(self.base, groups)
        result = sm.check_membership_changes(worse, self.z, polish_start=False,
                max_rounds=1, max_candidates=1, verbose=False)
        self.assertTrue(result.improved)
        self.assertTrue(result.report['membership_changed'])
        self.assertEqual(result.report['status'], 'candidate_limit')
        self.assertLess(result.diameter, self.d + 1e-8)
        self.assertTrue(result.report['coverage_check']['valid_numerically'])

    def test_dynamic_pair_constraints(self):
        n = sm.Neighborhood(self.base, self.base.positions(self.z))
        labels = n.labels.copy(); labels[32] = 2
        groups = n.groups_from_labels(labels)
        m = sm.model_with_memberships(self.base, groups)
        expected = {tuple(sorted(p)) for g in groups for p in itertools.combinations(g, 2)}
        self.assertEqual(set(map(tuple, m.pairs)), expected)
        values, _ = m.pair_values(self.z)
        distances = np.linalg.norm(m.positions(self.z)[m.pairs[:, 0]]
                                   - m.positions(self.z)[m.pairs[:, 1]], axis=1)**2
        np.testing.assert_allclose(values, distances, rtol=0, atol=5e-15)

    def test_pruned_vs_unpruned(self):
        n = sm.Neighborhood(self.base, self.base.positions(self.z))
        labels = n.labels.copy(); labels[32] = 2
        groups = n.groups_from_labels(labels)
        trial1 = sm.try_membership_change(self.base, self.z, groups)
        trial2 = sm.try_membership_change(self.base, self.z, groups, prune=False)
        self.assertEqual(trial1.status, 'tangent_bound_pruned')
        self.assertEqual(trial2.status, 'no_improvement_bound')
        self.assertGreaterEqual(trial2.diameter, self.d - 1e-8)
        self.assertTrue(trial2.report['coverage']['valid_numerically'])

    def test_random_relabeling_and_face_preserving_deformations(self):
        rng = np.random.default_rng(921)
        n = sm.Neighborhood(self.base, self.base.positions(self.z))
        for _ in range(5):
            labels = rng.integers(0, len(self.base.data.groups), len(n.labels))
            m = sm.model_with_memberships(self.base, n.groups_from_labels(labels))
            X = m.reference.copy()
            for j in range(m.fixed, len(X)):
                ids = [v for v in range(m.fixed) if set(m.active[j]) <= set(m.active[v])]
                weights = rng.dirichlet(np.ones(len(ids)))
                X[j] = weights @ m.reference[ids]
            z = m.coordinates(X)
            report = op.check_cover(m, m.positions(z))
            self.assertTrue(report['valid_numerically'], report)

    def test_saved_model_restarts(self):
        n = sm.Neighborhood(self.base, self.base.positions(self.z))
        labels = n.labels.copy(); labels[32] = 2
        m = sm.model_with_memberships(self.base, n.groups_from_labels(labels))
        data = op.Data(m.positions(self.z), m.data.groups, self.d)
        restarted = op.build_model(data, 23, 1e-6, op.model_record(m))
        self.assertEqual(set(map(tuple, restarted.pairs)), set(map(tuple, m.pairs)))
        self.assertEqual(restarted.active, m.active)
        self.assertTrue(op.check_cover(restarted, data.points)['valid_numerically'])

    def test_search_budget_is_honored(self):
        result = sm.check_membership_changes(self.base, self.z, polish_start=False,
                    max_candidates=5, max_rounds=1, verbose=False)
        self.assertEqual(result.report['counts']['candidates'], 5)
        self.assertEqual(result.report['status'], 'candidate_limit')
        self.assertLessEqual(result.diameter, self.d + 1e-12)

    def test_arguments_and_invalid_indices(self):
        with self.assertRaises(ValueError):
            sm.check_membership_changes(self.base, self.z, max_patch_size=0)
        groups = [list(g) for g in self.base.data.groups]
        groups[0].append(999)
        self.assertEqual(sm.try_membership_change(self.base, self.z, groups).status,
                         'certificate_rejected')
        with self.assertRaises(ValueError):
            sm.try_membership_change(self.base, self.z, self.base.data.groups,
                                     improvement_tol=-1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
