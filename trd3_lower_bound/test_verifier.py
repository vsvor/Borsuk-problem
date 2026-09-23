#!/usr/bin/env python3
"""Regression tests for the exact geometric and finite-graph verifier."""
from __future__ import annotations
import copy
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import random
import unittest
from verify_lower_bound import make_graph, refute_or_color, check_refutation
HERE = Path(__file__).resolve().parent

def graph(n, edges):
    a=[[] for _ in range(n)]
    for u,v in edges:a[u].append(v);a[v].append(u)
    return a

def naive_colorable(adj, k=4):
    """Independent vertex-at-a-time backtracking, without propagation."""
    order=sorted(range(len(adj)),key=lambda v:-len(adj[v]))
    labels=[-1]*len(adj)
    def visit(i):
        if i==len(adj):return True
        v=order[i]
        forbidden={labels[w] for w in adj[v]}
        for c in range(k):
            if c not in forbidden:
                labels[v]=c
                if visit(i+1):return True
        labels[v]=-1
        return False
    return visit(0)

# Exact arithmetic A+B*sqrt(2), for the 16-point hand-proof regression test.
def add(x,y):return (x[0]+y[0],x[1]+y[1])
def neg(x):return (-x[0],-x[1])
def mul(x,y):return (x[0]*y[0]+2*x[1]*y[1],x[0]*y[1]+x[1]*y[0])
def nonnegative(x):
    a,b=x
    if b==0:return a>=0
    if a>=0 and b>=0:return True
    if a<=0 and b<=0:return False
    return a*a>=2*b*b if a>=0 else 2*b*b>=a*a

def elementary_graph():
    zero=(F(0),F(0));u=(F(0),F(1,4));minus_u=neg(u)
    pts=[[minus_u]*3]
    for i in range(3):pts.append([minus_u if j==i else u for j in range(3)])
    for i in range(3):pts.append([u if j==i else minus_u for j in range(3)])
    for i in range(3):pts.append([(F(0),F(-1,2)) if j==i else zero for j in range(3)])
    for i in range(3):
        for j in range(3):
            if i==j:continue
            p=[zero]*3;p[i]=(F(1,2),F(0));p[j]=(F(-1,2),F(1,2));pts.append(p)
    edges=[]
    for i,j in combinations(range(16),2):
        sq=zero
        for a,b in zip(pts[i],pts[j]):
            d=add(a,neg(b));sq=add(sq,mul(d,d))
        if nonnegative(add(sq,(F(-7,8),F(0)))):edges.append((i,j))
    return graph(16,edges)

class Tests(unittest.TestCase):
    def test_k4(self):self.assertEqual(refute_or_color(graph(4,combinations(range(4),2)))[0]['status'],'COLORABLE')
    def test_k5(self):
        adj=graph(5,combinations(range(5),2));pr,st=refute_or_color(adj)
        self.assertEqual(st,check_refutation(adj,pr))
    def test_random_graphs_against_independent_solver(self):
        rng=random.Random(91029)
        for _ in range(80):
            n=rng.randrange(1,9);p=rng.random()
            adj=graph(n,[e for e in combinations(range(n),2) if rng.random()<p])
            pr,st=refute_or_color(adj)
            self.assertEqual(pr['status']=='COLORABLE',naive_colorable(adj))
            if pr['status']!='COLORABLE':self.assertEqual(st,check_refutation(adj,pr))
    def test_geometry_and_certificate(self):
        r=json.loads((HERE/'witness.json').read_text());adj,info=make_graph(r)
        self.assertEqual(info['points'],439);self.assertEqual(info['edges'],27264)
        pr=json.loads((HERE/'refutation.json').read_text());st=check_refutation(adj,pr)
        self.assertEqual(st['nodes'],142)
    def test_forged_leaf_rejected(self):
        pr={'status':'NOT_COLORABLE','colors':4,'root_fixed_to_color_zero':0,'tree':0}
        with self.assertRaises(ValueError):check_refutation(graph(4,[]),pr)
    def test_missing_branch_rejected(self):
        adj,_=make_graph(json.loads((HERE/'witness.json').read_text()))
        pr=json.loads((HERE/'refutation.json').read_text());pr['tree'][1].pop()
        with self.assertRaises(ValueError):check_refutation(adj,pr)
    def test_outside_point_rejected(self):
        r=json.loads((HERE/'witness.json').read_text());r['point_numerators'][0]=[r['coordinate_denominator'],0,0]
        with self.assertRaises(ValueError):make_graph(r)
    def test_rhombic_violation_rejected(self):
        r=json.loads((HERE/'witness.json').read_text());Q=r['coordinate_denominator'];r['point_numerators'][0]=[-Q,-Q,0]
        with self.assertRaises(ValueError):make_graph(r)
    def test_ideal_16_point_bound(self):
        adj=elementary_graph();pr,st=refute_or_color(adj)
        self.assertEqual(pr['status'],'NOT_COLORABLE')
        self.assertEqual(st,check_refutation(adj,pr))
        self.assertFalse(naive_colorable(adj))
    def test_five_color_extension(self):
        adj,_=make_graph(json.loads((HERE/'witness.json').read_text()))
        pr,_=refute_or_color(adj,k=5)
        self.assertEqual(pr['status'],'COLORABLE')

if __name__=='__main__':unittest.main()
