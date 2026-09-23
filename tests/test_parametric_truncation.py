#!/usr/bin/env python3
"""Fast geometry/CLI regression tests; optional end-to-end search checks."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

import parametric_truncation_search as s

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'scripts'


class ParametricTests(unittest.TestCase):
    def test_unshifted_adjacent(self):
        r = s.make_target(0, 0, "adjacent")
        self.assertEqual(r["vertex_count"], 26)
        self.assertEqual(r["active_truncation_facets"], 4)
        self.assertEqual(r["facet_count"], 16)
        # Base volume minus four disjoint axial caps at distance 1/2.
        expected = 1 / math.sqrt(2) - (16 / 3) * (1 / math.sqrt(2) - 0.5)**3
        self.assertAlmostEqual(r["target_volume"], expected, places=13)
        self.assertAlmostEqual(r["target_diameter"], 1.2421325286834142, places=12)

    def test_unshifted_opposite(self):
        r = s.make_target(0, 0, "opposite")
        self.assertEqual(r["vertex_count"], 26)
        self.assertEqual(r["active_truncation_facets"], 4)
        self.assertAlmostEqual(r["target_diameter"], math.sqrt(2), places=13)
        self.assertAlmostEqual(r["target_volume"], s.make_target(0, 0)["target_volume"], places=13)

    def test_sign_of_lower_cut(self):
        r = s.make_target("0.02", "0.01")
        V = np.array(r["vertices"])
        self.assertAlmostEqual(V[:, 0].min(), -0.49, places=13)
        self.assertAlmostEqual(V[:, 0].max(), 0.52, places=13)
        self.assertAlmostEqual(r["parameters"]["x_slab_width"], 1.01, places=13)
        self.assertAlmostEqual(r["parameters"]["x_slab_center"], 0.015, places=13)

    def test_boundary_a_plus_b(self):
        s.make_target("-0.03", "0.03")
        with self.assertRaises(s.ParameterError):
            s.make_target("-0.030000000000000001", "0.03")

    def test_invalid_inputs(self):
        for a, b in (("nan", "0"), ("inf", "0"), ("0", "-inf"),
                     ("0", "-0.1"), ("0", "1"), ("0", "2"),
                     ("1", "1.3"), ("0", "0.9999999999")):
            with self.subTest(a=a, b=b), self.assertRaises(s.ParameterError):
                s.make_target(a, b)

    def test_does_not_require_origin(self):
        r = s.make_target("0.2", "0.6")
        self.assertFalse(r["origin_in_target"])
        self.assertGreater(r["inscribed_ball_radius"], 0)
        self.assertGreaterEqual(np.min(np.array(r["vertices"])[:, 0]), 0.1 - 1e-12)

    def test_redundant_cut(self):
        r = s.make_target("0.3", "0.0")
        self.assertEqual(r["active_truncation_facets"], 3)
        self.assertTrue(r["warnings"])
        with self.assertRaises(s.ParameterError):
            s.make_target("0.3", "0.0", require_four_cuts=True)

    def test_combinatorial_changes(self):
        r = s.make_target("0.0", "0.5")
        self.assertEqual(r["active_truncation_facets"], 4)
        self.assertNotEqual(r["vertex_count"], 26)

    def test_extreme_redundant_offset(self):
        r = s.make_target("1e100", "0.0")
        self.assertEqual(r["active_truncation_facets"], 3)
        self.assertTrue(np.all(np.isfinite(r["vertices"])))

    def test_qhull_independent_construction(self):
        for case, a, b in itertools.product(("adjacent", "opposite"), (0., .05, .2), (0., .07, .3)):
            with self.subTest(case=case, a=a, b=b):
                r = s.make_target(a, b, case)
                A, rhs = np.array(r["A"]), np.array(r["rhs"])
                h = HalfspaceIntersection(np.column_stack((A, -rhs)), np.array(r["interior_point"]))
                W, V = h.intersections, np.array(r["vertices"])
                d = np.linalg.norm(V[:, None] - W[None, :], axis=2)
                self.assertLess(max(d.min(axis=0).max(), d.min(axis=1).max()), 2e-9)
                self.assertAlmostEqual(ConvexHull(W).volume, r["target_volume"], places=11)

    def test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.json"
            r = s.make_target("0.02", "0.01")
            s._write_target_guarded(path, r)
            contents = path.read_bytes()
            s._write_target_guarded(path, s.make_target("2e-2", "1e-2"))
            self.assertEqual(contents, path.read_bytes())
            with self.assertRaises(s.ParameterError):
                s._write_target_guarded(path, s.make_target("0.03", "0.01"))
            self.assertEqual(contents, path.read_bytes())

    def test_cli_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(HERE / "parametric_truncation_search.py"),
                   "--a", "-0.01", "--b", "0.03", "--case", "opposite", "--generate-only",
                   "--output-dir", tmp]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            self.assertEqual(p.returncode, 0, msg=p.stderr)
            r = json.loads((Path(tmp) / "target.json").read_text())
            self.assertEqual(r["parameters"]["case"], "opposite")
            p = subprocess.run(cmd + ["--vertices", "other.json"], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
