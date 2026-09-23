#!/usr/bin/env python3
"""Recompute the five-cut / four-adjacent comparison from the archived records.

Usage: python compare_records.py [--directory PATH]
Requires numpy and scipy. Input files: five_best.json, four_adjacent_best.json.
All point indices are zero-based; displayed cell numbers are one-based.
This compares saved numerical covers; it does not run a new global search.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

S=np.array([[-1,0,0],[0,0,-1],[0,1,0]],dtype=int)
CELL_MAP=[3,4,2,1]  # one-based, five -> four-adjacent

def planes(case):
    A=[];b=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        for si,sj in itertools.product([-1,1],repeat=2):
            n=np.zeros(3);n[i]=si;n[j]=sj;A.append(n);b.append(1/np.sqrt(2))
    cuts={'five':[(0,1),(0,-1),(1,1),(1,-1),(2,1)],
          'four_adjacent':[(0,-1),(1,-1),(2,1),(2,-1)]}[case]
    for i,s in cuts:
        n=np.zeros(3);n[i]=s;A.append(n);b.append(.5)
    return np.array(A),np.array(b)

def pair_analysis(r,atol):
    p=np.asarray(r['points'],dtype=float);groups=r['groups']
    edges=sorted({tuple(sorted(e)) for g in groups for e in itertools.combinations(g,2)})
    vals={e:float(np.linalg.norm(p[e[0]]-p[e[1]])) for e in edges};D=max(vals.values())
    active={e for e,d in vals.items() if D-d<=atol}
    vs=sorted({i for e in active for i in e})
    triangles=[tri for tri in itertools.combinations(vs,3) if all(e in active for e in itertools.combinations(tri,2))]
    core=set(e for tri in triangles for e in itertools.combinations(tri,2))
    per_cell=[]
    for k,g in enumerate(groups):
        es=[e for e in edges if set(e)<=set(g)]
        per_cell.append({'cell':k+1,'memberships':g,'size':len(g),'diameter':max(vals[e] for e in es),
                         'active_pairs':[list(e) for e in es if e in active]})
    nfix=len(r['target']['vertices']);fd,fi,fj=max((d,i,j) for (i,j),d in vals.items() if i<nfix or j<nfix)
    return {'point_count':len(p),'fixed_target_vertices':nfix,'coordinate_variables':r['variables'],
            'pair_constraint_count':len(edges),'maximum_diameter':D,'active_tolerance':atol,
            'cells':per_cell,'active_pair_count':len(active),
            'active_pairs':[{'pair':list(e),'length':vals[e],'gap_from_maximum':D-vals[e],
                             'cells':[k+1 for k,g in enumerate(groups) if set(e)<=set(g)],
                             'in_triangular_core':e in core} for e in sorted(active)],
            'bottleneck_triangles':[list(t) for t in triangles],
            'next_inactive_gap':min(D-d for e,d in vals.items() if e not in active),
            'max_pair_with_fixed_endpoint':{'pair':[fi,fj],'length':fd,'slack':D-fd},
            'saved_geometry_checks':r['geometry'],'saved_model_lower_bound':r['solver'].get('lower_bound')}

def point_key(r,j,A,b,transform=None,cell_map=None):
    p=np.asarray(r['reference']['points'][j]);ids=np.flatnonzero(np.abs(A@p-b)<1e-8)
    ns=A[ids] if transform is None else A[ids]@transform.T
    support=tuple(sorted(tuple(n.astype(int).tolist()) for n in ns))
    owners=[k+1 for k,g in enumerate(r['groups']) if j in g]
    if cell_map is not None: owners=[cell_map[k-1] for k in owners]
    return tuple(sorted(owners)),support

def export_cover(path,points,groups,target):
    """Place target vertices first, retain all remaining generators, remap lists."""
    out=[np.array(v,dtype=float) for v in target];remap={}
    for i,p in enumerate(points):
        ds=np.linalg.norm(np.asarray(out)-p,axis=1);j=int(np.argmin(ds))
        if ds[j]>1e-10: j=len(out);out.append(np.array(p,dtype=float))
        remap[i]=j
    gs=[sorted({remap[j] for j in g}) for g in groups];out=np.array(out)
    D=max(float(np.linalg.norm(out[i]-out[j])) for g in gs for i,j in itertools.combinations(g,2))
    path.write_text(f'4\n{D:.17g}\n{len(out)}\n'+json.dumps(gs)+'\n'+json.dumps(out.tolist())+'\n')
    return {'filename':path.name,'point_count':len(out),'fixed_count':len(target),'diameter':D,'groups':gs,
            'input_index_to_output_index':remap}

def main(directory):
    r={k:json.loads((directory/(k+'_best.json')).read_text()) for k in ['five','four_adjacent']}
    for k in r:
        txt=(directory/(k+'_best.txt')).read_text().splitlines()
        assert json.loads(txt[3])==r[k]['groups']
        assert np.max(np.abs(np.array(json.loads(txt[4]))-np.array(r[k]['points'])))==0
    f,a=r['five'],r['four_adjacent']; x=np.array(f['points']);y=np.array(a['points']);xt=x@S.T
    analyses={k:pair_analysis(rr,1e-8) for k,rr in r.items()}
    nf=len(f['target']['vertices']);na=len(a['target']['vertices'])
    A5,b5=planes('five');A4,b4=planes('four_adjacent')
    mapping={};removed=[]
    for i in range(nf):
        d=np.linalg.norm(y[:na]-xt[i],axis=1);j=int(np.argmin(d))
        if d[j]<1e-12: mapping[i]=j
        else: removed.append(i)
    keys={}
    for j in range(na,len(y)):
        key=point_key(a,j,A4,b4)
        if key in keys: raise ValueError('Ambiguous incidence key')
        keys[key]=j
    for i in range(nf,len(x)):
        mapping[i]=keys[point_key(f,i,A5,b5,S,CELL_MAP)]
    assert len(mapping)==len(set(mapping.values()))==46
    new=[j for j in range(len(y)) if j not in mapping.values()]
    assert removed==[0,1,2,3] and new==[25]
    for k,g in enumerate(f['groups']):
        mapped={mapping[i] for i in g if i in mapping};expected=set(a['groups'][CELL_MAP[k]-1])-set(new)
        assert mapped==expected
    core5={i for tri in analyses['five']['bottleneck_triangles'] for i in tri}
    core4={i for tri in analyses['four_adjacent']['bottleneck_triangles'] for i in tri}
    assert {mapping[i] for i in core5}==core4
    tri_maps=[]
    for tri in analyses['five']['bottleneck_triangles']:
        tri_maps.append({'five':tri,'four_adjacent_corresponding_order':[mapping[i] for i in tri]})
    mov5={tuple(sorted((i,j))) for g in f['groups'] for i,j in itertools.combinations(g,2) if i>=nf and j>=nf}
    mov4={tuple(sorted((i,j))) for g in a['groups'] for i,j in itertools.combinations(g,2) if i>=na and j>=na}
    assert {tuple(sorted((mapping[i],mapping[j]))) for i,j in mov5}==mov4
    apex=np.array([1/np.sqrt(2),0,0]);add_cell=2;source_cell=CELL_MAP.index(add_cell)
    apd=np.linalg.norm(xt[f['groups'][source_cell]]-apex,axis=1);imax=int(np.argmax(apd))
    apexmax=float(apd[imax]);capbase=xt[removed]
    assert set(removed)<=set(f['groups'][source_cell]) and apexmax<analyses['five']['maximum_diameter']
    ordered=[None]*4
    for k,g in enumerate(f['groups']): ordered[CELL_MAP[k]-1]=g.copy()
    ordered[add_cell-1].append(len(x))
    extension=export_cover(directory/'four_adjacent_extended_from_five.txt',
                           np.vstack([xt,apex]),ordered,a['target']['vertices'])
    # Clip only L2 at x=1/2, replacing the apex by the four cap-base corners.
    assert max(y[j,0] for k,g in enumerate(a['groups']) if k!=1 for j in g)<.5
    q=.5;aa=1/np.sqrt(2)-q
    corners=np.array([[q,s*aa,t*aa] for s,t in itertools.product([-1,1],repeat=2)])
    yclip=np.vstack([y,corners]);clipgroups=[g.copy() for g in a['groups']]
    clipgroups[1]=[j for j in clipgroups[1] if j!=25]+list(range(len(y),len(y)+4))
    # The unused apex must not enter the saved point set.
    use=[j for j in range(len(yclip)) if j!=25];rr={j:k for k,j in enumerate(use)}
    clipgroups=[[rr[j] for j in g] for g in clipgroups]
    clip=export_cover(directory/'five_clipped_from_four_adjacent.txt',
                      yclip[use]@S,clipgroups,f['target']['vertices'])
    report={'source_sha256':{k:hashlib.sha256((directory/(k+'_best.json')).read_bytes()).hexdigest() for k in r},
            'indexing':'All point indices zero-based. Cells are one-based. Raw memberships refer to the unmodified input records.',
            'five':analyses['five'],'four_adjacent':analyses['four_adjacent'],
            'best_diameter_difference':analyses['four_adjacent']['maximum_diameter']-analyses['five']['maximum_diameter'],
            'alignment':{'map':'T(x,y,z)=(-x,-z,y)','matrix':S.tolist(),'cells_five_to_four':CELL_MAP,
                         'target_relation':'T(P5)=P4_adjacent intersect {x<=1/2}',
                         'matching_fixed_vertices':nf-len(removed),'matching_movable_vertices':len(x)-nf,
                         'removed_five_vertices':removed,'new_four_vertex':25,
                         'max_core_coordinate_error':max(float(np.linalg.norm(xt[i]-y[mapping[i]])) for i in core5),
                         'max_all_matching_coordinate_error':max(float(np.linalg.norm(xt[i]-y[mapping[i]])) for i in mapping),
                         'triangles':tri_maps,'all_point_matches':[{'five':i,'four_adjacent':j,'position_error':float(np.linalg.norm(xt[i]-y[j])),'in_core':i in core5} for i,j in mapping.items()],
                         'movable_movable_pair_constraints_identical':True,'movable_movable_pair_constraint_count':len(mov5),
                         'reference_movable_face_incidence_keys_identical':True},
            'cap_extension':{'apex':apex.tolist(),'five_cell_before_transform':source_cell+1,'four_cell_after_transform':add_cell,
                             'cap_base_original_five_indices':removed,'max_distance_apex_to_old_cell_vertices':apexmax,
                             'attaining_original_five_vertex':f['groups'][source_cell][imax],
                             'diameter_slack':analyses['five']['maximum_diameter']-apexmax,
                             'cap_volume':float(ConvexHull(np.vstack([capbase,apex])).volume),
                             'max_x_at_core_vertices':float(np.max(xt[sorted(core5),0])),
                             'distance_extra_cut_from_core':.5-float(np.max(xt[sorted(core5),0])),
                             'extended_cover':extension,'clipped_cover':clip},
            'arithmetic_note':'Pair distances and geometry are floating-point recomputations. Exact rational/algebraic bounds for the idealized three-point bottleneck are provided separately; they are not a global optimality certificate for arbitrary covers.'}
    if (directory/'triangle_certificate.json').exists():report['exact_triangle_certificate']=json.loads((directory/'triangle_certificate.json').read_text())
    (directory/'comparison_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'best_diameters':{k:analyses[k]['maximum_diameter'] for k in analyses},
                      'active_pairs':{k:analyses[k]['active_pair_count'] for k in analyses},
                      'core_max_error':report['alignment']['max_core_coordinate_error'],
                      'matching_movable_pairs':len(mov5),'extra_cap_max_distance':apexmax,
                      'extended_cover_diameter':extension['diameter'],'clipped_cover_diameter':clip['diameter']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    main(ap.parse_args().directory)
