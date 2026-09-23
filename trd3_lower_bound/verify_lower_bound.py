#!/usr/bin/env python3
"""Exact finite-point lower bound for four-set covers of the positive 3-TRD.

Only Python's standard library is used. Coordinates are integers divided by
one common positive denominator; no floating-point decisions are made.

Usage:
  python verify_lower_bound.py
  python verify_lower_bound.py witness.json --proof refutation.json
  python verify_lower_bound.py witness.json --write-proof refutation.json
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time


def make_graph(record: dict) -> tuple[list[list[int]], dict]:
    """Check the exact geometry and construct ALL pairs at distance >= L."""
    if record.get('target') != 'positive_three_truncated_rhombic_dodecahedron':
        raise ValueError('Unrecognized target.')
    Q = record['coordinate_denominator']
    pts = record['point_numerators']
    L = Fraction(record['lower_bound'])
    if type(Q) is not int or Q <= 0 or L <= 0 or not pts:
        raise ValueError('Invalid denominator, lower bound, or empty witness.')
    if len({tuple(p) for p in pts}) != len(pts):
        raise ValueError('Duplicate points in the certificate.')
    for i, p in enumerate(pts):
        if len(p) != 3 or any(type(a) is not int for a in p):
            raise ValueError(f'Point {i}: need exactly three integers.')
        # x,y,z <= 1/2. There are NO extra lower axial cuts.
        if any(2*a > Q for a in p):
            raise ValueError(f'Point {i} violates a positive axial truncation.')
        # |x|+|y| <= 1/sqrt(2), etc. Squaring is safe: both sides >= 0.
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if 2*(abs(p[a])+abs(p[b]))**2 > Q**2:
                raise ValueError(f'Point {i} is outside a rhombic face.')
    adj = [[] for _ in pts]
    rhs = L.numerator**2 * Q**2
    den2 = L.denominator**2
    minimum = None
    edge_count = 0
    for i, p in enumerate(pts):
        for j in range(i):
            sq = sum((p[k]-pts[j][k])**2 for k in range(3))
            if sq*den2 >= rhs:
                adj[i].append(j)
                adj[j].append(i)
                edge_count += 1
                minimum = sq if minimum is None else min(minimum, sq)
    info = dict(points=len(pts), edges=edge_count, lower_bound=str(L),
                minimum_edge_squared_numerator=minimum,
                squared_distance_denominator=Q**2,
                all_points_inside_exact_target=True)
    return adj, info


def propagate(adj: list[list[int]], domains: list[int]) -> bool:
    """Remove the color of each forced vertex from all its neighbors."""
    queue = [v for v, m in enumerate(domains) if m.bit_count() == 1]
    done = [False]*len(adj)
    for v in queue:
        if done[v]:
            continue
        done[v] = True
        bit = domains[v]
        for w in adj[v]:
            if domains[w] & bit:
                domains[w] ^= bit
                if not domains[w]:
                    return False
                if domains[w].bit_count() == 1:
                    queue.append(w)
    return True


def refute_or_color(adj: list[list[int]], k: int = 4) -> tuple[dict, dict]:
    """Exhaustive coloring search. An UNSAT result carries a decision tree."""
    if not adj or not 1 <= k <= 8:
        raise ValueError('Need a nonempty graph and 1 <= k <= 8.')
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2*len(adj)+100))
    root = max(range(len(adj)), key=lambda v: len(adj[v]))
    domains = [(1 << k)-1]*len(adj)
    domains[root] = 1  # without loss of generality, permute color names
    stats = {'nodes': 0, 'contradiction_leaves': 0, 'branch_nodes': 0}
    coloring = None

    def search(dom: list[int]):
        nonlocal coloring
        stats['nodes'] += 1
        if not propagate(adj, dom):
            stats['contradiction_leaves'] += 1
            return 0
        candidates = [v for v, m in enumerate(dom) if m.bit_count() > 1]
        if not candidates:
            coloring = [m.bit_length()-1 for m in dom]
            return None
        v = min(candidates, key=lambda j: (dom[j].bit_count(), -len(adj[j]), j))
        stats['branch_nodes'] += 1
        branches = []
        for c in range(k):
            if dom[v] & (1 << c):
                child = dom.copy()
                child[v] = 1 << c
                proof = search(child)
                if proof is None:
                    return None
                branches.append([c, proof])
        return [v, branches]

    tree = search(domains)
    if tree is None:
        assert coloring is not None
        if any(coloring[u] == coloring[v] for u in range(len(adj)) for v in adj[u]):
            raise RuntimeError('Internal error: purported coloring is invalid.')
        return {'status': 'COLORABLE', 'coloring': coloring}, stats
    return {'status': 'NOT_COLORABLE', 'colors': k,
            'root_fixed_to_color_zero': root, 'tree': tree}, stats


def check_refutation(adj: list[list[int]], proof: dict) -> dict:
    """Check the supplied exhaustive proof, without doing coloring search.

    This deliberately uses a second propagation implementation: repeated sweeps
    rather than the searcher's singleton queue. Every permitted branch must
    occur, and every leaf must derive an empty color domain.
    """
    k = proof['colors']
    root = proof['root_fixed_to_color_zero']
    if proof['status'] != 'NOT_COLORABLE' or k != 4:
        raise ValueError('Need a four-color refutation.')
    if type(root) is not int or not 0 <= root < len(adj):
        raise ValueError('Invalid root vertex.')
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2*len(adj)+100))
    stats = {'nodes': 0, 'contradiction_leaves': 0, 'branch_nodes': 0}
    domains = [15]*len(adj)
    domains[root] = 1

    def verify(dom, node):
        stats['nodes'] += 1
        changed = True
        conflict = False
        while changed and not conflict:
            changed = False
            for v, mask in enumerate(dom):
                if not mask:
                    conflict = True
                    break
                if mask & (mask-1):
                    continue
                for w in adj[v]:
                    new = dom[w] & ~mask
                    if new != dom[w]:
                        changed = True
                        dom[w] = new
                        if not new:
                            conflict = True
                            break
                if conflict:
                    break
        if node == 0:
            if not conflict:
                raise ValueError('Proof leaf has no contradiction.')
            stats['contradiction_leaves'] += 1
            return
        if conflict:
            raise ValueError('Nonterminal proof node after a contradiction.')
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError('Invalid proof node.')
        v, branches = node
        if type(v) is not int or not 0 <= v < len(adj) or dom[v].bit_count() < 2:
            raise ValueError('Invalid branching vertex.')
        required = [c for c in range(4) if dom[v] & (1 << c)]
        if [b[0] for b in branches] != required:
            raise ValueError('A color branch is missing, duplicated, or invalid.')
        stats['branch_nodes'] += 1
        for c, child in branches:
            d = dom.copy()
            d[v] = 1 << c
            verify(d, child)

    verify(domains, proof['tree'])
    return stats


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('witness', nargs='?', type=Path, default=here/'witness.json')
    p.add_argument('--proof', type=Path, help='Check an existing refutation instead of searching.')
    p.add_argument('--write-proof', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args(argv)
    start = time.monotonic()
    try:
        record = json.loads(args.witness.read_text())
        adj, info = make_graph(record)
        if args.proof:
            proof = json.loads(args.proof.read_text())
            stats = check_refutation(adj, proof)
        else:
            proof, stats = refute_or_color(adj)
            if proof['status'] == 'COLORABLE':
                print('NOT CERTIFIED: the witness graph is four-colorable.')
                return 1
            second = check_refutation(adj, proof)
            if second != stats:
                raise RuntimeError('The independent proof checker disagrees.')
        if args.write_proof:
            args.write_proof.write_text(json.dumps(proof, separators=(',', ':'))+'\n')
        info.update(status='EXACT_LOWER_BOUND_VERIFIED', lower_bound_decimal=record['lower_bound'],
                    coloring_proof=stats, elapsed_seconds=time.monotonic()-start,
                    arithmetic='Integer/rational geometry and exhaustive finite graph proof; no floating point decisions.')
        if args.report:
            args.report.write_text(json.dumps(info, indent=2)+'\n')
        print(f'All {len(adj)} rational points are in the exact positive 3-TRD.')
        print(f'Graph: {info["edges"]} edges, each of length >= {record["lower_bound"]}.')
        print(f'Four-color refutation verified: {stats}.')
        print(f'PROVED: beta_4(P3) >= {record["lower_bound"]}.')
        print(f'Time: {info["elapsed_seconds"]:.3f} s')
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
