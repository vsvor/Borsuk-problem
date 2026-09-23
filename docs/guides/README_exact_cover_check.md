# Independent exact verification of convex covers in 3D

`exact_cover_check.py` reads only the final point coordinates, hull membership
lists, and a target definition. It does not import an optimizer or read an
initial topology, reference mesh, plane-incidence model, or solver status.
Python 3.10 or newer is sufficient. No third-party packages are required.

## Recommended commands for the two balanced covers

```bash
python3 exact_cover_check.py delta_star_dummy.txt \
  --a 0.5 --b 0.008811423368752003 --case adjacent \
  --pad 1e-12 --diameter-bound 0.96539917866 \
  --full-volume --overlaps --report dummy_check.json

python3 exact_cover_check.py delta_star_upper.txt \
  --a 0.008811423368752003 --b 0 --case adjacent \
  --pad 1e-12 --diameter-bound 0.96539917866 \
  --full-volume --overlaps --report upper_check.json
```

These commands check explicitly enlarged hulls, NOT the unmodified input hulls.
The enlargement is explained below. The original input files are never changed.
Both checks pass; exact upper bounds on the enlarged maximum diameters are
0.965399178654095 and 0.965399178654163, respectively. Full 3D inclusion-exclusion
has EXACTLY zero missing volume, not a small floating-point residual.

Omit `--full-volume` for the faster boundary/common-point certificate. Omit
`--overlaps` when only a cover and a diameter bound are needed. Both reductions
are valid for these examples. Use `--pad 0` (the default) to check the literal
input hulls. The unmodified decimal hulls have small genuine gaps in the ideal
analytic targets, for which exact rational counterexample points are saved.

## 1. Two independent coverage tests

### General test: exact inclusion-exclusion (`--full-volume`)

Write P for the target and K_i for the checked closed convex hulls. Calculate

    missing = volume(P) - sum over nonempty I of
              (-1)^(|I|+1) volume(P intersect intersection_{i in I} K_i).

With four cells this requires 15 convex intersections. All halfspaces,
intersection vertices, volumes, additions and comparisons are rational/exact.
In particular, `missing == 0` is an exact comparison.

For a full-dimensional compact convex P, equality to zero proves *all* of P is
covered, including its boundary: if x in P is missing from the finite closed
union, a small ball around x misses the union, and its intersection with P has
positive 3D volume. Thus there cannot be an isolated or boundary-only missing
point when the exact missing volume is zero.

No common point or original topology is needed. The program also runs the
boundary check and refuses a verdict if the two methods contradict each other.
The number of intersections grows as 2^k-1, so use this for a small number of
cells, not thousands of cells.

### Faster test: final boundary subtraction + a checked common point

For each target face F, start with the polygon F, and subtract each checked
hull in turn using its exact halfspaces. A convex polygon minus a convex body
is represented as a list of convex polygons. Intersections are computed with
rational arithmetic; no thin positive-area polygon is ignored.

If no polygons remain on any target face, the whole boundary is covered. A
boundary point outside the closed union would have a relatively open uncovered
neighborhood on that face, hence positive planar area. Only exactly zero-area
pieces may be discarded. A plane containing the whole target face does NOT
count as an outside halfspace; this avoids false gaps along coincident planes.

The program also checks c in P and c in every K_i. It tries shared listed
vertices, the mean of all input points, and the other input points. An explicit
candidate may be supplied as `--common x y z`; it is checked, never trusted.
The existence of a common point is not inferred from the initial configuration.

For x in P, extend the ray from c through x to b on the target boundary. One
checked hull contains b and c, so convexity puts the entire segment cb, including
x, in that hull. Therefore the boundary certificate covers the entire solid.

If the boundary passes but no common point is verified, the default result is
INCONCLUSIVE, not PASS. Add `--full-volume` to settle the general case. The test
suite includes a hollow cubical shell that covers its entire boundary but has
an interior cavity; the fast test refuses to certify it and the volume test
computes the exact missing volume 1/8.

## 2. Exact arithmetic and the literal data

The file format is the existing five-line format:

1. Number of cells.
2. Stored diameter (not trusted).
3. Number of points.
4. Lists of zero-based point indices, one per cell.
5. Point coordinate triples.

JSON decimal tokens are read directly into rational numbers without an
intermediate `float`. Thus 0.1 means exactly 1/10. Homogeneous integer coordinates
(X,Y,Z,W), W>0, represent the rational point (X/W,Y/W,Z/W). Integers have arbitrary
precision. Floats are used only in fields explicitly marked as approximate and
for elapsed wall-clock time; they cannot decide any geometry or bound comparison.

The convex hull is constructed by exhaustive enumeration of triples of input
points. A triple contributes a facet precisely when all points lie on one side
of its plane. This identifies *all* exact supporting facets, including ones
created by tiny rounding perturbations. Non-full-dimensional cells are rejected.
The code does not use Qhull, SciPy, Shapely, floating-point coplanarity tests, or
an optimization termination status.

Polytope vertices are found by exact enumeration of all nonsingular triples of
halfspaces, followed by exact feasibility checks. Volume is a sum of oriented
facet-triangle determinant contributions divided by six. Lower-dimensional
intersections have exactly zero volume.

This is a mathematical certificate by an independently executable exact
arithmetic algorithm. It is not a proof-assistant-verified implementation; the
source and tests are supplied for inspection.

## 3. Target choices

Choose exactly one:

* `--fixed-count N`: target = convex hull of the first N input coordinate
  triples, interpreted as exact decimals. This checks the literal saved target,
  not an ideal polyhedron whose vertices merely round to those numbers.
* `--target vertices.json`: a JSON vertex array or object with a `vertices`
  array. Those decimals are used exactly.
* `--a A --b B [--case adjacent|opposite]`: the ideal parametric rhombic target.

For the analytic mode the base is

    |x|+|y| <= 1/sqrt(2), |x|+|z| <= 1/sqrt(2), |y|+|z| <= 1/sqrt(2).

The x slab is -1/2+b <= x <= 1/2+a. In the adjacent case the other cuts are
y>=-1/2 and z>=-1/2. In the opposite case they are -1/2<=y<=1/2. A redundant cut,
such as the x upper plane when a=0.5, is harmless. The constraints a+b>=0 and a
nonempty slab are checked. Empty/lower-dimensional targets are rejected.

No float approximation to 1/sqrt(2) is silently declared exact. Instead integer
square-root arithmetic constructs rational t_- and t_+ with

    t_- < 1/sqrt(2) < t_+,  t_+ - t_- = 10^-30.

Coverage of the rational OUTER target with base constant t_+ proves coverage of
the exact intended target. This changes the base offset, not the diameter unit.
The a,b values themselves are the exact supplied decimal/rational numbers.

If the outer target is not covered, that alone does not disprove coverage of the
ideal target. The program then tries to find an uncovered boundary point of the
rational INNER target with base constant t_-. Such a point is a valid exact
counterexample inside the ideal target. If no such point is found, the ideal
target result remains INCONCLUSIVE. In full-volume mode an interior-only gap
without a boundary witness may also leave this *ideal-target* verdict unresolved;
it does not get silently classified as a proved failure for the ideal body.

## 4. Geometric padding is a change of the covering sets

If an original hull has the exact inequalities n_f.x <= h_f, `--pad e` checks

    E_i = intersection_f {x: n_f.x <= h_f + e*||n_f||_1}.

e is nonnegative and rational. At e=0 this is the original hull. At e>0, these
are different, explicitly specified polytopes, not original sets checked with
a looser acceptance tolerance. They contain the original hulls.

The offset is scale-invariant under positive rescaling of a plane equation.
Each plane moves outward by at most sqrt(3)*e in Euclidean distance. This does
NOT imply a Hausdorff-distance bound for the entire hull or a universal
D+2*sqrt(3)*e diameter formula: intersections near corners need not have that
bound. The program enumerates the actual expanded vertices and recomputes their
maximum squared pair distance EXACTLY. It then compares with `--diameter-bound`
squared, without taking an approximate square root. Printed diameter upper
bounds are rounded upward by integer-square-root arithmetic.

For the two examples, e=10^-12 proves D <= 0.96539917866, and therefore also
D < 0.966. This does not prove that the raw files were already exact covers.
If covering pieces must lie inside the true target, use E_i intersect P. This
preserves convexity and coverage and cannot increase a diameter. Exact integer
halfspace descriptions of all E_i are included in the reports. Do not round
these halfspaces/vertices back to decimals and silently assume the certificate
survives unchanged.

## 5. Covering versus partition

`--overlaps` constructs P intersect E_i intersect E_j for each pair. Exact affine
rank of its enumerated vertices decides whether that intersection has nonempty
3D interior. Every positive overlap has a saved strict-interior rational witness.
A positive-volume overlap means the checked clipped convex cells are not a
partition with disjoint interiors. Shared boundary faces alone are allowed.

In analytic mode, overlap checks refer to the rational OUTER target. A negative
overlap test is therefore also valid for the ideal target. A positive overlap
in that tiny outer collar alone would not automatically establish an overlap
inside the ideal target; the rational witness can be checked separately. For
the packaged examples all six positive overlap witnesses also lie strictly
inside the ideal target, as recorded in verification_summary.json.

For a partition into arbitrary subsets (without a convexity requirement), any
finite cover can be converted to a partition by giving each point to the first
cell containing it. Diameters cannot increase. No universal-containment theorem
for arbitrary unit-diameter sets is being checked here.

## 6. Results and exit codes

The input is never overwritten. `--report FILE` selects the result JSON; otherwise
it is INPUT_STEM_exact_check.json. Exit codes: 0 = coverage proved and the optional
diameter bound proved; 1 = failed or inconclusive coverage, or failed diameter
bound; 2 = malformed/unsupported input or a computation error. Positive overlaps
are reported separately and do not make a successful *cover* exit with failure.

Important report fields:

* `covers_checked_target`, `covers_ideal_target` (where applicable).
* `checked_max_diameter_upper`, `diameter_bound_proved`.
* `boundary`, including exact uncovered-point witnesses when found.
* `full_volume_check`, including an exact zero test and exact missing-volume
  numerator/denominator encoded as hexadecimal integers.
* `pairwise_overlaps`, `original_hulls_contained_in_checked_target`.
* `checked_cell_halfspaces_exact` and `checked_target_halfspaces_exact`.

An exact rational point witness is given as four integer strings X,Y,Z,W.
A float display may round a tiny gap away; the integer strings are authoritative.

## 7. Tests

```bash
python3 -m unittest -v test_exact_cover_check.py
```

All 16 regression tests pass, including exact touching cells, overlapping cells,
coincident planes, a 10^-40 gap, explicit geometric expansion, a hollow shell,
exact volume calculations, input parsing, and both real balanced examples.
The general verifier requires full-dimensional input cells and a bounded,
full-dimensional target; it is designed for the small 3D problems in this project.
