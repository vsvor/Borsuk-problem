#!/usr/bin/env python3
"""Fast audit of data, memberships, stored diameters and certificate provenance.

Does not redo a coverage proof; use tools/verify_results.py for that. Standard
library only. --write-report writes fresh numerical pair tables to runs/audit/.
"""
from fractions import Fraction as F
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import exact_cover_check as ec

def audit():
    manifest=json.loads((ROOT/'results.json').read_text());out=[]
    for entry in manifest['results']:
        source=ROOT/entry['partition'];ps,groups,stored=ec.load_partition(source)
        if len(ps)!=entry['point_count']:raise ValueError('Point-count mismatch: '+entry['id'])
        if len(groups)!=4:raise ValueError('Not four cells: '+entry['id'])
        best=F(0);cells=[]
        for k,g in enumerate(groups):
            d,pair=ec.diameter_squared([ps[j] for j in g]);best=max(best,d)
            cells.append({'cell':k+1,'memberships':g,'diameter_upper':ec.sqrt_upper(d),'maximizing_pair':[g[i] for i in pair]})
        if abs(best-stored**2)>F('1e-11'):raise ValueError('Stored diameter mismatch: '+entry['id'])
        target=json.loads((ROOT/entry['target']).read_text())
        exact_prefix=[ec.from_xyz(x) for x in json.loads((ROOT/entry['target']).read_text(),parse_float=F)['vertices']]
        if exact_prefix!=ps[:entry['fixed_count']]:raise ValueError('Target prefix mismatch.')
        cert=json.loads((ROOT/entry['certificate']).read_text())
        if cert['input_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('Stale certificate input.')
        if cert['target_definition_sha256']!=hashlib.sha256((ROOT/entry['target']).read_bytes()).hexdigest():raise ValueError('Stale certificate target.')
        if cert['covers_ideal_target'] is not True or cert['diameter_bound_proved'] is not True:raise ValueError('Certificate is not a PASS.')
        if cert['full_volume_check']['missing_volume_exact_zero'] is not True:raise ValueError('Missing full volume PASS.')
        co={tuple(sorted(pair)) for g in groups for pair in itertools.combinations(g,2)};active=[]
        # D-d <= 1e-8 tested equivalently after exact rational bracketing of D.
        Dlo=F(ec.sqrt_upper(best,18))-F(1,10**18);Dhi=Dlo+F(1,10**18)
        for i,j in sorted(co):
            d,_=ec.diameter_squared([ps[i],ps[j]])
            if d>=(Dlo-F('1e-8'))**2:
                active.append({'pair':[i,j],'length_upper':ec.sqrt_upper(d),'cells':[k+1 for k,g in enumerate(groups) if i in g and j in g]})
        out.append({'id':entry['id'],'fixed_count':entry['fixed_count'],'point_count':len(ps),
                    'stored_diameter':str(stored),'recomputed_diameter_upper':ec.sqrt_upper(best),
                    'certificate_diameter_upper':cert['checked_max_diameter_upper'],
                    'cells':cells,'active_pair_tolerance':'1e-8','active_pairs':active,
                    'file_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    return out

def check_checksums():
    count=0
    for line in (ROOT/'CHECKSUMS.sha256').read_text().splitlines():
        checksum,path=line.split('  ',1);file=ROOT/path
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=checksum:raise ValueError('Checksum mismatch: '+path)
        count+=1
    return count

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checksums',action='store_true');p.add_argument('--write-report',type=Path)
    a=p.parse_args();out=audit()
    for r in out:print(r['id']+': raw D <= '+r['recomputed_diameter_upper']+'; certificate D <= '+r['certificate_diameter_upper'])
    if a.checksums:print('Checksums OK:',check_checksums(),'files')
    if a.write_report:
        a.write_report.parent.mkdir(parents=True,exist_ok=True);a.write_report.write_text(json.dumps(out,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
