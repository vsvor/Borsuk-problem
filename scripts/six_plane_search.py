#!/usr/bin/env python3
"""Four conical cells of a truncated rhombic dodecahedron; SLSQP multistart.

Python >= 3.10; dependencies: numpy, scipy. No torch/shapely/previous scripts.
The six planes are PAIRWISE INTERFACES, not six complete cutting planes.
Default target: the 23-vertex positive-axial truncation from r_dod_4_0966.txt.

  python six_plane_search.py --starts 1000 --workers 4 --output six_search
  python six_plane_search.py --check six_search/best.json

See README_six_plane_search.md for the model and numerical limitations.
"""
from __future__ import annotations
import os
# Avoid each subprocess starting a full BLAS thread pool; explicit override.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = os.environ.get('PARTITION_BLAS_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from scipy.spatial.transform import Rotation

PAIRS = list(combinations(range(4), 2))  # 01, 02, 03, 12, 13, 23
TETRA = np.array([[1.,1.,1.], [1.,-1.,-1.], [-1.,1.,-1.], [-1.,-1.,1.]])/np.sqrt(3.)
GEOM_TOL = 2e-10


class GeometryError(ValueError):
    pass


class StopRun(Exception):
    pass


@dataclass
class Target:
    A: np.ndarray
    b: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray
    volume: float
    name: str


def make_target(name: str = 'positive') -> Target:
    """All supporting planes have distance 1/2 from the origin."""
    if name not in ('positive', 'symmetric'):
        raise ValueError("Target must be 'positive' or 'symmetric'.")
    rows = []
    for i,j in combinations(range(3), 2):
        for si,sj in product((-1.,1.), repeat=2):
            n = np.zeros(3); n[i] = si/np.sqrt(2.); n[j] = sj/np.sqrt(2.)
            rows.append(n)
    rows.extend(np.eye(3))
    if name == 'symmetric':
        rows.extend(-np.eye(3))
    A = np.asarray(rows); b = np.full(len(A), .5)
    v = HalfspaceIntersection(np.column_stack((A,-b)), np.zeros(3)).intersections
    v = np.array(sorted(v.tolist(), key=lambda p: tuple(np.round(p,12))))
    # A polytope edge is the intersection of two independent supporting faces.
    tight = abs(v @ A.T - b) < 1e-9
    edges = []
    for i,j in combinations(range(len(v)),2):
        common = A[tight[i] & tight[j]]
        if len(common) >= 2 and np.linalg.matrix_rank(common, tol=1e-9) >= 2:
            edges.append((i,j))
    return Target(A,b,v,np.array(edges,dtype=int),float(ConvexHull(v).volume),name)


def score_matrix(theta: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Eight variables; return four score vectors and their analytic Jacobian.

    theta=(rx,ry,rz, log(U00),log(U11), U01,U02,U12).
    B=Rz Ry Rx U, diag(U)=(exp(a), exp(b), exp(-a-b)); det(B)=1.
    scores[i] = B @ TETRA[i]. Thus no rank-collapse penalty is needed.
    """
    t = np.asarray(theta,dtype=float)
    if t.shape != (8,) or not np.isfinite(t).all():
        raise GeometryError('Expected eight finite plane parameters.')
    x,y,z,a,b,c,d,e = t
    sx,cx,sy,cy,sz,cz = np.sin(x),np.cos(x),np.sin(y),np.cos(y),np.sin(z),np.cos(z)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],float)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],float)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],float)
    dx = np.array([[0,0,0],[0,-sx,-cx],[0,cx,-sx]],float)
    dy = np.array([[-sy,0,cy],[0,0,0],[-cy,0,-sy]],float)
    dz = np.array([[-sz,-cz,0],[cz,-sz,0],[0,0,0]],float)
    U = np.array([[np.exp(a),c,d],[0,np.exp(b),e],[0,0,np.exp(-a-b)]])
    R = Rz@Ry@Rx
    D = np.zeros((8,3,3))
    D[0]=Rz@Ry@dx@U; D[1]=Rz@dy@Rx@U; D[2]=dz@Ry@Rx@U
    for k in (3,4):
        E=np.zeros((3,3)); E[k-3,k-3]=U[k-3,k-3]; E[2,2]=-U[2,2]
        D[k]=R@E
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),start=5):
        E=np.zeros((3,3)); E[i,j]=1.; D[k]=R@E
    scores=TETRA@(R@U).T
    dscores=np.einsum('kij,vj->vik',D,TETRA)
    return scores,dscores


def planes_from_scores(scores: np.ndarray, dscores: np.ndarray | None = None):
    """n_ij points out of cell i: n_ij.x <= 0 for i<j, >=0 for cell j."""
    scores=np.asarray(scores,dtype=float)
    if scores.shape != (4,3) or not np.isfinite(scores).all():
        raise GeometryError('Expected a finite (4,3) score matrix.')
    if dscores is None:
        dscores=np.zeros((4,3,8))
    raw=np.array([scores[j]-scores[i] for i,j in PAIRS])
    dr=np.array([dscores[j]-dscores[i] for i,j in PAIRS])
    lengths=np.linalg.norm(raw,axis=1)
    if min(lengths)<1e-12:
        raise GeometryError('Coincident score vectors.')
    normals=raw/lengths[:,None]
    dn=(dr-normals[:,:,None]*np.einsum('ij,ijk->ik',normals,dr)[:,None,:])/lengths[:,None,None]
    return normals,dn


def cell_planes(normals: np.ndarray, dn: np.ndarray, cell: int):
    ids=[k for k,p in enumerate(PAIRS) if cell in p]
    signs=np.array([1 if PAIRS[k][0]==cell else -1 for k in ids])
    return normals[ids]*signs[:,None],dn[ids]*signs[:,None,None]


def cell_vertices(target: Target, N: np.ndarray, dN: np.ndarray | None=None):
    """Vertices of P intersect {N x<=0}, plus piecewise analytic derivatives.

    Exhaustive types: retained P vertices, intersections of P edges with a
    cut plane, cone extreme rays clipped to P, and the common apex 0.
    No triangulation, sampling, or fixed clipping topology is used.
    """
    if dN is None:
        dN=np.zeros((3,3,8))
    if abs(np.linalg.det(N))<1e-10:
        raise GeometryError('Nearly singular three-plane cone.')
    keep=np.max(target.vertices@N.T,axis=1)<=GEOM_TOL
    points=[target.vertices[keep]]
    derivs=[np.zeros((int(keep.sum()),3,8))]
    u=target.vertices[target.edges[:,0]]
    w=target.vertices[target.edges[:,1]]-u
    for k in range(3):
        den=w@N[k]
        ok=abs(den)>1e-13
        ids=np.flatnonzero(ok)
        t=-(u[ok]@N[k])/den[ok]
        inside=(t>=-GEOM_TOL)&(t<=1+GEOM_TOL)
        ids=ids[inside]; t=t[inside]
        p=u[ids]+t[:,None]*w[ids]
        good=np.max(p@N.T,axis=1)<=GEOM_TOL
        ids=ids[good]; p=p[good]
        # dx = -(edge direction) (dn.x)/(n.edge direction).
        dp=-w[ids,:,None]*(p@dN[k])[:,None,:]/den[ids,None,None]
        points.append(p); derivs.append(dp)
    for i,j in combinations(range(3),2):
        k=3-i-j
        ray=np.cross(N[i],N[j])
        dr=np.cross(dN[i].T,N[j]).T+np.cross(N[i],dN[j].T).T
        if N[k]@ray>0:
            ray=-ray; dr=-dr
        ar=target.A@ray
        h=int(np.argmax(ar/target.b))
        if ar[h]<=1e-15:
            raise GeometryError('Unbounded/ill-conditioned cone ray.')
        t=target.b[h]/ar[h]
        p=t*ray
        dp=t*(dr-ray[:,None]*(target.A[h]@dr)[None,:]/ar[h])
        points.append(p[None,:]); derivs.append(dp[None,:,:])
    points.append(np.zeros((1,3))); derivs.append(np.zeros((1,3,8)))
    p=np.concatenate(points); dp=np.concatenate(derivs)
    # Repeated vertices at clipping transitions are harmless. Keep distinct
    # derivative branches there, but remove exact copies in final export.
    if not np.isfinite(p).all() or max(np.max(p@target.A.T-target.b),np.max(p@N.T))>1e-8:
        raise GeometryError('Numerical halfspace violation.')
    return p,dp


def _unique(p: np.ndarray, tol: float=1e-10):
    result=[]
    for q in p:
        if not result or min(np.linalg.norm(q-r) for r in result)>tol:
            result.append(q.copy())
    return np.array(result)


def check_topology(normals: np.ndarray, target: Target | None=None,
                   tolerance: float=2e-8) -> dict:
    """Test six oriented pairwise planes, INCLUDING independently supplied ones.

    Input order 01,02,03,12,13,23; <= side belongs to the smaller label.
    Cells have disjoint interiors automatically by opposite pair inequalities.
    Four positive volumes + volume sum + all six 2-D interfaces test coverage
    and the intended complete adjacency, numerically (not interval arithmetic).
    Independent normals generally fail the coverage test.
    """
    if target is None: target=make_target()
    try:
        normals=np.asarray(normals,dtype=float)
        if normals.shape!=(6,3) or not np.isfinite(normals).all():
            raise GeometryError('Expected six finite 3-D normal vectors.')
        lengths=np.linalg.norm(normals,axis=1)
        if min(lengths)<1e-12: raise GeometryError('Zero normal.')
        normals=normals/lengths[:,None]
        cells=[]; volumes=[]; diameters=[]; matrices=[]; residual=0.
        for i in range(4):
            N,_=cell_planes(normals,np.zeros((6,3,8)),i)
            # Independent reconstruction by HalfspaceIntersection, not the
            # edge/ray enumerator used by the optimizer.
            direction=np.linalg.solve(N,-np.ones(3)); direction/=np.linalg.norm(direction)
            positive=target.A@direction>0
            radius=min(target.b[positive]/(target.A[positive]@direction))
            interior=.4*radius*direction
            if min(-N@interior)<1e-12: raise GeometryError('No stable strict interior.')
            A=np.vstack((target.A,N)); b=np.r_[target.b,np.zeros(3)]
            p=_unique(HalfspaceIntersection(np.c_[A,-b],interior).intersections)
            hull=ConvexHull(p); p=p[hull.vertices]
            cells.append(p); matrices.append(N); volumes.append(float(hull.volume))
            d2=np.sum((p[:,None,:]-p[None,:,:])**2,axis=2)
            diameters.append(float(np.sqrt(np.max(d2))))
            residual=max(residual,float(np.max(p@A.T-b)))
        interfaces=[]
        for k,(i,j) in enumerate(PAIRS):
            p=cells[i]
            p=p[abs(p@normals[k])<1e-8]
            # Include only points also satisfying the other cell inequalities.
            p=p[np.max(p@matrices[j].T,axis=1)<1e-8]
            singular=np.linalg.svd(p-p.mean(axis=0),compute_uv=False) if len(p)>=3 else np.zeros(3)
            interfaces.append(bool(len(singular)>1 and singular[1]>1e-8))
        error=float(sum(volumes)-target.volume)
        full=all(v>target.volume*1e-10 for v in volumes)
        good=(full and all(interfaces) and abs(error)<=tolerance*target.volume
              and residual<=tolerance)
        return dict(ok=bool(good),cell_volumes=volumes,cell_diameters=diameters,
                    diameter=max(diameters),target_volume=target.volume,
                    volume_sum_error=error,relative_volume_error=abs(error)/target.volume,
                    all_six_interfaces=all(interfaces),interfaces=interfaces,
                    max_halfspace_violation=residual,
                    reason='ok' if good else 'Incomplete cover, negligible cell, or missing interface')
    except (GeometryError,ValueError,np.linalg.LinAlgError,QhullError) as e:
        return dict(ok=False,reason=str(e))


class Objective:
    def __init__(self,target:Target,top_pairs:int=8,seconds:float=0.):
        self.target=target; self.top_pairs=top_pairs
        self.deadline=time.monotonic()+seconds if seconds else float('inf')
        self.last=None; self.best_theta=None; self.best_diameter=float('inf'); self.evals=0

    def evaluate(self,theta):
        if time.monotonic()>self.deadline: raise StopRun('time limit')
        if self.last is not None and np.array_equal(theta,self.last):
            return self.values,self.jacobian
        scores,ds=score_matrix(theta); normals,dn=planes_from_scores(scores,ds)
        vals=[]; grads=[]
        for i in range(4):
            N,dN=cell_planes(normals,dn,i)
            p,dp=cell_vertices(self.target,N,dN)
            ia,ib=np.triu_indices(len(p),1)
            delta=p[ia]-p[ib]; d2=np.einsum('ij,ij->i',delta,delta)
            order=np.argsort(-d2,kind='stable')[:self.top_pairs]
            g=2*np.einsum('ij,ijk->ik',delta[order],dp[ia[order]]-dp[ib[order]])
            val=d2[order]
            if len(val)<self.top_pairs:
                missing=self.top_pairs-len(val)
                val=np.r_[val,np.zeros(missing)]; g=np.vstack((g,np.zeros((missing,8))))
            vals.append(val); grads.append(g)
        self.last=np.array(theta,copy=True)
        self.values=np.concatenate(vals); self.jacobian=np.vstack(grads); self.evals+=1
        diameter=float(np.sqrt(max(self.values)))
        if diameter<self.best_diameter:
            self.best_diameter=diameter; self.best_theta=self.last.copy()
        return self.values,self.jacobian

    def constraints(self,x):
        return x[-1]-self.evaluate(x[:8])[0]

    def jac(self,x):
        return np.c_[-self.evaluate(x[:8])[1],np.ones(4*self.top_pairs)]


def optimize_once(theta0:np.ndarray,target:Target|None=None,*,maxiter:int=250,
                  ftol:float=1e-10,top_pairs:int=8,seconds:float=0.,
                  target_diameter:float|None=None,reject_above:float|None=None,
                  reject_after:int=25,shape_bound:float=1.5,shear_bound:float=3.) -> dict:
    """Rebuild the exact clipped cells at every SLSQP evaluation; no fixed mesh.

    The problem is NONCONVEX and piecewise smooth. Topology changes and ties
    are allowed, but analytic derivatives there are selected branch gradients.
    'success' is solver status, not a global-optimality certificate.
    """
    if target is None: target=make_target()
    theta0=np.asarray(theta0,float).copy()
    if theta0.shape!=(8,): raise ValueError('theta0 must have length eight.')
    theta0[3:5]=np.clip(theta0[3:5],-shape_bound,shape_bound)
    theta0[5:]=np.clip(theta0[5:],-shear_bound,shear_bound)
    initial=check_topology(planes_from_scores(score_matrix(theta0)[0])[0],target)
    if not initial['ok']:
        return dict(valid=False,reason=initial['reason'])
    obj=Objective(target,top_pairs)
    d0=obj.evaluate(theta0)[0].max()
    # Always retain a usable initial partition, even with a tiny time budget.
    obj.deadline=time.monotonic()+seconds if seconds else float('inf')
    x0=np.r_[theta0,d0+1e-9]
    bounds=[(None,None)]*3+[(-shape_bound,shape_bound)]*2+[(-shear_bound,shear_bound)]*3+[(0,None)]
    iterations=0; start=time.monotonic(); success=False; status=-1
    def callback(x):
        nonlocal iterations
        iterations+=1
        obj.evaluate(x[:8])
        if target_diameter is not None and obj.best_diameter<=target_diameter:
            raise StopRun('target reached')
        if reject_above is not None and iterations>=reject_after and obj.best_diameter>reject_above:
            raise StopRun('heuristic early rejection')
    try:
        if target_diameter is not None and obj.best_diameter<=target_diameter:
            raise StopRun('target reached at initialization')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Values in x were outside bounds')
            result=minimize(lambda x:x[-1],x0,jac=lambda x:np.r_[np.zeros(8),1.],
                            method='SLSQP',bounds=bounds,
                            constraints={'type':'ineq','fun':obj.constraints,'jac':obj.jac},
                            callback=callback,options={'maxiter':maxiter,'ftol':ftol,'disp':False})
        message=str(result.message); success=bool(result.success); status=int(result.status)
    except (StopRun,GeometryError,np.linalg.LinAlgError) as e:
        message=str(e)
    theta=obj.best_theta
    scores,_=score_matrix(theta); normals,_=planes_from_scores(scores)
    check=check_topology(normals,target)
    valid=bool(check['ok'] and abs(check['diameter']-obj.best_diameter)<2e-8)
    return dict(valid=valid,parameters=theta.tolist(),scores=scores.tolist(),
                plane_normals=normals.tolist(),plane_pairs=[list(p) for p in PAIRS],
                diameter=float(check.get('diameter',obj.best_diameter)),
                initial_diameter=float(np.sqrt(d0)),verification=check,
                success=success,status=status,message=message,iterations=iterations,
                evaluations=obj.evals,seconds=time.monotonic()-start,target=target.name,
                shape_bound=shape_bound,shear_bound=shear_bound,
                at_shape_bound=bool(np.any(abs(theta[3:5])>shape_bound-1e-6)
                                    or np.any(abs(theta[5:])>shear_bound-1e-6)))


def random_start(rng:np.random.Generator,shape_spread:float=.35) -> np.ndarray:
    """Haar-random rotation plus random anisotropy/shear of a regular fan."""
    theta=np.zeros(8)
    theta[:3]=Rotation.random(random_state=rng).as_euler('xyz')
    theta[3:]=rng.normal(0,shape_spread,5)
    return theta


def export_partition(result:dict,target:Target) -> tuple[np.ndarray,list[list[int]],float]:
    """Place all target vertices FIRST (23 by default), then new cell vertices."""
    p=list(target.vertices.copy()); groups=[]
    normals=np.array(result['plane_normals'])
    for i in range(4):
        N,_=cell_planes(normals,np.zeros((6,3,8)),i)
        v,_=cell_vertices(target,N)
        group=[]
        for q in v:
            distances=np.linalg.norm(np.asarray(p)-q,axis=1)
            k=int(np.argmin(distances))
            if distances[k]>2e-10:
                k=len(p); p.append(q.copy())
            if k not in group: group.append(k)
        groups.append(group)
    points=np.asarray(p)
    actual=max(float(np.max(np.linalg.norm(points[g,None,:]-points[None,g,:],axis=2))) for g in groups)
    if abs(actual-result['diameter'])>2e-8:
        raise GeometryError('Export diameter disagrees with independently checked hulls.')
    return points,groups,actual


def _atomic(path:Path,text:str):
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(text,encoding='utf-8'); os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def save_record(result:dict,target:Target,output:Path):
    output.mkdir(parents=True,exist_ok=True)
    p,g,d=export_partition(result,target)
    result=dict(result,diameter=d,fixed_count=len(target.vertices),point_count=len(p),
                format_version=1,parameter_order=['rx','ry','rz','log_u00','log_u11','u01','u02','u12'])
    txt='\n'.join((str(4),repr(d),str(len(p)),json.dumps(g),json.dumps(p.tolist())))+'\n'
    js=json.dumps(result,indent=2,allow_nan=False)+'\n'
    for ext in ('txt','json'):
        old=output/f'best.{ext}'
        if old.exists(): _atomic(output/f'previous_best.{ext}',old.read_text())
    # Each file is replaced atomically; the pair is not a filesystem transaction.
    _atomic(output/'best.txt',txt); _atomic(output/'best.json',js)


def _worker(job):
    start_id,seed,target_name,theta,options=job
    rng=np.random.default_rng(seed)
    spread=options.pop('shape_spread')
    if theta is None: theta=random_start(rng,spread)
    result=optimize_once(np.asarray(theta),make_target(target_name),**options)
    result.update(start_id=start_id,random_seed=int(seed))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--starts',type=int,default=100,help='Number of local searches; 0 = no count limit')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--target',choices=('positive','symmetric'),default='positive')
    parser.add_argument('--output',type=Path,default=Path('six_plane_results'))
    parser.add_argument('--resume',action='store_true',help='Load best.json; use a new --seed for fresh random starts')
    parser.add_argument('--target-diameter',type=float,help='Stop the whole search at a verified diameter <= this value')
    parser.add_argument('--seconds-per-start',type=float,default=0.)
    parser.add_argument('--hours',type=float,default=0.)
    parser.add_argument('--maxiter',type=int,default=250)
    parser.add_argument('--ftol',type=float,default=1e-10)
    parser.add_argument('--top-pairs',type=int,default=8)
    parser.add_argument('--reject-above',type=float,help='Optional heuristic, not a lower-bound certificate')
    parser.add_argument('--reject-after',type=int,default=25)
    parser.add_argument('--shape-spread',type=float,default=.35)
    parser.add_argument('--shape-bound',type=float,default=1.5)
    parser.add_argument('--shear-bound',type=float,default=3.)
    parser.add_argument('--mutation-probability',type=float,default=0.,help='Optional perturbations of the current record')
    parser.add_argument('--mutation-scale',type=float,default=.15)
    parser.add_argument('--check',type=Path,help='Check saved JSON or a JSON array of six oriented normals and exit')
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args(argv)
    if args.starts<0 or args.workers<1 or args.maxiter<1 or args.top_pairs<1 or args.seed<0:
        parser.error('Invalid count or seed.')
    if min(args.seconds_per_start,args.hours,args.shape_spread,args.mutation_scale)<0 or min(args.ftol,args.shape_bound,args.shear_bound)<=0:
        parser.error('Invalid tolerance, time, or parameter range.')
    if not 0<=args.mutation_probability<=1: parser.error('Mutation probability must be in [0,1].')
    if args.reject_after<1: parser.error('--reject-after must be positive.')
    if args.check:
        data=json.loads(args.check.read_text())
        name=data.get('target',args.target) if isinstance(data,dict) else args.target
        normals=data['plane_normals'] if isinstance(data,dict) else data
        checked=check_topology(np.array(normals),make_target(name))
        print(json.dumps(checked,indent=2)); return 0 if checked['ok'] else 2
    target=make_target(args.target)
    incumbent=None
    if (args.output/'best.json').exists():
        if not args.resume: parser.error('Output already has a record; use --resume or a different --output.')
        incumbent=json.loads((args.output/'best.json').read_text())
        if incumbent['target']!=args.target: parser.error('Target differs from saved record.')
        check=check_topology(np.array(incumbent['plane_normals']),target)
        if not check['ok']: parser.error('Saved record failed independent verification.')
        incumbent['diameter']=check['diameter']
    if incumbent and args.target_diameter is not None and incumbent['diameter']<=args.target_diameter:
        if not args.quiet: print(f"Existing best D={incumbent['diameter']:.12f} already meets target.")
        return 0
    rng=np.random.default_rng(args.seed)
    deadline=time.monotonic()+3600*args.hours if args.hours else float('inf')
    options={k:getattr(args,k) for k in ('maxiter','ftol','top_pairs','target_diameter',
             'reject_above','reject_after','shape_bound','shear_bound','shape_spread')}
    options['seconds']=args.seconds_per_start
    submitted=completed=invalid=0; stop=False
    def job():
        nonlocal submitted
        seed=int(rng.integers(0,2**63-1)); theta=None
        if args.mutation_probability>0 and incumbent and rng.random()<args.mutation_probability:
            theta=(np.array(incumbent['parameters'])+rng.normal(0,args.mutation_scale,8)).tolist()
        j=(submitted,seed,args.target,theta,options.copy()); submitted+=1; return j
    def allowed():
        return not stop and (args.starts==0 or submitted<args.starts) and time.monotonic()<deadline
    def accept(r):
        nonlocal incumbent,completed,invalid,stop
        completed+=1
        if not r['valid']: invalid+=1; return
        if incumbent is None or r['diameter']<incumbent['diameter']-1e-10:
            save_record(r,target,args.output); incumbent=r
            if not args.quiet:
                print(f"BEST D={r['diameter']:.12f}  start={r['start_id']}  n={len(export_partition(r,target)[0])}",flush=True)
        if incumbent and args.target_diameter is not None and incumbent['diameter']<=args.target_diameter:
            stop=True
    try:
        if args.workers==1:
            while allowed(): accept(_worker(job()))
        else:
            # No run logs or task database. At most one pending task per worker.
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending=set()
                while allowed() and len(pending)<args.workers:
                    pending.add(pool.submit(_worker,job()))
                while pending:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for f in done: accept(f.result())
                    if allowed():
                        while allowed() and len(pending)<args.workers:
                            pending.add(pool.submit(_worker,job()))
                    else:
                        # Running solves finish; no new jobs are launched.
                        for f in list(pending):
                            if f.cancel(): pending.remove(f)
    except KeyboardInterrupt:
        if not args.quiet: print('Interrupted; completed record checkpoints are retained.',flush=True)
    if not args.quiet:
        best=f"{incumbent['diameter']:.12f}" if incumbent else 'none'
        print(f'Completed={completed}; rejected geometry={invalid}; best={best}',flush=True)
    return 0 if incumbent else 1


if __name__=='__main__':
    raise SystemExit(main())
