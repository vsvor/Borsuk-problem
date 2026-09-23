#!/usr/bin/env python3
"""Regenerate the checked-in SHA-256 manifest after an intentional repository edit."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.venv','venv','__pycache__','.pytest_cache','runs'}
rows=[]
for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file() or any(x in IGNORED for x in rel.parts):continue
    if p.name=='CHECKSUMS.sha256' or p.suffix in ('.pyc','.pyo','.tmp'):continue
    rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+rel.as_posix())
(ROOT/'CHECKSUMS.sha256').write_text('\n'.join(rows)+'\n')
print('Wrote checksums for',len(rows),'files.')
