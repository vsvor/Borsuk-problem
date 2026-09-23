from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repo_audit',ROOT/'tools/check_results.py')
auditmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditmod)
class RepositoryTests(unittest.TestCase):
    def test_registered_results_and_certificate_hashes(self):
        r=auditmod.audit()
        self.assertEqual(len(r),11)
        self.assertEqual([x['point_count'] for x in r[:5]],[42,42,43,43,47])
        self.assertEqual({x['id'] for x in r[5:]},{'six_trd','tetrahedron','cube','octahedron','dodecahedron','icosahedron'})
    def test_primary_raw_gaps_and_padded_exact_covers(self):
        entries=json.loads((ROOT/'results.json').read_text())['results']
        for e in entries[:5]:
            raw=json.loads((ROOT/'certificates/current'/f"{e['id']}_raw.json").read_text())
            padded=json.loads((ROOT/e['certificate']).read_text())
            self.assertFalse(raw['covers_ideal_target'])
            self.assertTrue(padded['covers_ideal_target'])
            self.assertTrue(padded['full_volume_check']['missing_volume_exact_zero'])
            self.assertEqual(len(padded['full_volume_check']['intersections']),15)
    def test_continuation_coordinates_and_memberships(self):
        for name in ['dummy','upper']:
            directory=ROOT/'data/balanced'/name
            rows=(directory/'partition.txt').read_text().splitlines()
            r=json.loads((directory/'continuation.json').read_text())
            self.assertEqual(json.loads(rows[3]),r['groups'])
            self.assertEqual(json.loads(rows[4]),r['points'])
            self.assertAlmostEqual(float(rows[1]),r['diameter'],places=14)
    def test_copied_source_files_unchanged(self):
        items=json.loads((ROOT/'reports/import_provenance.json').read_text())
        for r in items:
            if not r['modified']:
                self.assertEqual(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest(),r['sha256'],r['path'])
    def test_symbolic_formula_with_independent_sympy(self):
        try:import sympy as s
        except ImportError:self.skipTest('Optional SymPy is not installed.')
        for key in ['coarse','fine']:
            r=json.loads((ROOT/f'symbolic/six_planes/{key}_certificate.json').read_text())
            ds=r['maximum_diameter_squared']
            x=s.Rational(ds['rational'])+s.Rational(ds['sqrt2'])*s.sqrt(2)
            coeff=list(map(int,r['diameter_annihilating_polynomial_descending']))
            expression=0
            for i,c in enumerate(coeff):
                if c:
                    power=len(coeff)-1-i
                    self.assertEqual(power%2,0)
                    expression+=c*x**(power//2)
            self.assertEqual(s.expand(expression),0)
            self.assertLess(s.N(s.sqrt(x),35),s.Rational(r['diameter_bound']))
if __name__=='__main__':unittest.main()
