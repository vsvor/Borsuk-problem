# A global lower bound for four-part coverings of the 3-TRD

Let

\[
P_3=\{(x,y,z):x,y,z\le 1/2,\quad
 |x|+|y|,|x|+|z|,|y|+|z|\le 1/\sqrt2\}.
\]

The three negative axial vertices are **not** truncated. Distances are in these
physical coordinates, with the facet planes of the target at distance 1/2.
Write beta_4(P_3) for the infimum of the largest diameter among four sets whose
union contains P_3. Parts need not be convex, connected, measurable, or disjoint.

## Result

**The finite certificate in this directory proves exactly**

\[
\boxed{\beta_4(P_3)\ge 0.9695977535
       =1939195507/2000000000.}
\]

Together with the previously certified padded-cover upper bound in the project
repository, this gives

\[
0.9695977535\le\beta_4(P_3)\le 0.969767761865705.
\]

The gap is 0.000170008365705, about 0.017531% of the upper bound. The upper bound
is quoted from the existing project, not recomputed by this directory. It refers
to the explicitly enlarged cover, not its unmodified decimal coordinates.
The global optimum has **not** been identified.

## Finite-point proof

`witness.json` gives 439 distinct rational points, with coordinates of the form
(A,B,C)/10^12. Every point is checked to lie in the *ideal* target by exact
integer inequalities:

- 2A, 2B, 2C <= 10^12;
- 2(|A|+|B|)^2, 2(|A|+|C|)^2, 2(|B|+|C|)^2 <= 10^24.

Join two points whenever their squared distance is at least L^2, where
L=0.9695977535. Since 10^12 L = 969597753500 is an integer, this can equivalently
be checked by comparing the sum of three squared numerator differences with
969597753500^2. The resulting graph has 27,264 edges and is not four-colorable.
A separate five-coloring is also supplied, so its chromatic number is exactly 5.

If four sets of diameter strictly less than L covered P_3, assigning each witness
point to one of its covering sets would properly four-color this graph. This is
a contradiction. The argument is independent of all optimization models,
incidences, convexity, common-point assumptions, and initial topology.

## Recheck

Python 3.10+; no third-party packages are required.

```bash
# Rebuild the graph exactly, find a refutation, then independently check it:
python3 verify_lower_bound.py

# Just check the supplied proof tree, without running a coloring search:
python3 verify_lower_bound.py witness.json --proof refutation.json

# Regenerate the proof/report:
python3 verify_lower_bound.py witness.json \
    --write-proof new_refutation.json --report new_verification.json

python3 -m unittest -v test_verifier.py
```

The saved refutation has 142 nodes: 70 branching nodes and 72 contradiction
leaves. The first vertex may be fixed to color zero because color names may be
permuted. At every subsequent node the checker verifies that *all* remaining
colors are represented. At each leaf, forced-color propagation must derive an
empty color domain. The independent proof checker uses repeated sweeps, whereas
the proof searcher uses a singleton queue.

Rebuilding the graph, regenerating the proof, and checking it took about 2.6
seconds in the computation environment. Ten regression tests pass, including
random small graphs compared with an independent backtracker, malicious-proof
rejection, exact target containment, and the 16-point elementary construction.

## Elementary lower bound

`ELEMENTARY_PROOF.md` gives a completely noncomputational proof of
beta_4(P_3) >= sqrt(7/8), using 16 explicitly described points and just two cases.
The stronger 0.9695977535 result relies on the finite computer-checked certificate.

## Discovery versus verification

The witness was discovered by sampling the target edges and faces, testing
four-colorability of far-distance graphs, and adding points that obstruct the
current coloring. A conflict core reduced an intermediate 2,323-point witness
to 442 points. The minimum prescribed-edge length was then increased by a
constrained numerical optimization, keeping every point in the target.

Finally coordinates were rounded toward zero to the grid 10^-12, three resulting
duplicate points were removed, and **everything relevant to the lower-bound
proof was recomputed exactly**. The numerical discovery stage is not trusted by
the verifier and is not needed to reproduce the theorem. No claim is made that
the witness has the smallest possible number of points or is optimally placed.

An exact finite verification is still a program-based proof, not a
proof-assistant formalization. All mathematical decisions in the verifier use
integers or rational numbers; elapsed-time reporting is the only floating-point
output. The positive five-coloring is supplementary and not required for the
lower bound.

## Scope

In particular, this specific fixed P_3 cannot be covered by four sets of diameter
0.966. This is **not** a lower bound for the universal Borsuk constant over all
sets of diameter one: P_3 itself has diameter greater than one. The balanced
multi-target construction concerns a different covering problem.
