#!/usr/bin/env python3
"""Run portable regression tests; no search histories or network access needed."""
import argparse
import os
from pathlib import Path
import sys
import unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pattern',default='test*.py');p.add_argument('--quiet',action='store_true')
a=p.parse_args()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=a.pattern)
r=unittest.TextTestRunner(verbosity=1 if a.quiet else 2).run(suite)
raise SystemExit(0 if r.wasSuccessful() else 1)
