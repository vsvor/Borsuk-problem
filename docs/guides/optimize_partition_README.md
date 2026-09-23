# Diameter minimization for the supplied four-hull cover

## Run

Requirements: Python 3.10 or later, NumPy, SciPy, and Shapely.

```bash
python3 -m pip install numpy scipy shapely
python3 optimize_partition.py r_dod_4_0966.txt \
    --fixed-count 23 \
    --output r_dod_4_optimized.txt \
    --report r_dod_4_optimized.json \
    --check-gradients
```

The output has the same five-line format as the input. Point indices and the
four membership lists are unchanged. The second line is replaced by the ACTUAL
maximum diameter computed from all pairs in every hull. The input's stored
objective value is not trusted or used as a diameter constraint. The input is
never overwritten. Existing output/report files at explicitly chosen output
paths may be replaced after successful verification.

Tested with NumPy 2.3.5, SciPy 1.17.0, and Shapely 2.1.2. No CVXPY, commercial
solver, GPU, or SAT solver is required. The optimizer uses analytic derivatives.

## Result on the supplied file

The supplied coordinates give these diameters:

| Hull | Input diameter | Optimized diameter |
|---|---:|---:|
| K1 | 1.030201890436 | 0.969767761861449 |
| K2 | 0.965967320248 | 0.969767761861448 |
| K3 | 1.044200534577 | 0.969767761861449 |
| K4 | 0.965967558049 | 0.969767761861449 |

Maximum diameter decreases from 1.044200534576963 to 0.969767761861449,
an improvement of about 7.1282%. K2 and K4 become slightly larger: the objective
is the maximum over the four diameters, not monotone improvement of each one.

The run used 26 movable scalar coordinates, one epigraph variable, and 496
distinct co-hull pair inequalities. SLSQP converged in 9 iterations. These are
measurements from the supplied example, not iteration guarantees for other data.

A separate tangent-relaxation LP gives the numerical diameter interval

    [0.969767761856292, 0.969767761861449].

The lower endpoint includes an explicit 1e-11 safety allowance in squared
objective units and a correction for nonzero dual stationarity residuals. This
is numerical evidence of the GLOBAL minimum in the fixed-incidence model below,
not an interval-arithmetic certificate or the minimum over all four-part covers.

## What is fixed and what moves

The target is the convex hull of input points 0,...,22. Its 15 planes are inferred
from these fixed points; the program does not accidentally introduce three
additional negative axial truncations. For this file the target is

    x, y, z <= 1/2,
    |x|+|y|, |x|+|z|, |y|+|z| <= 1/sqrt(2).

Points 0,...,22 are kept exactly as stored. Each remaining point is classified
by the target face planes within --incidence-tol, initially 1e-6. A point on an
edge moves on that same edge; a point in a face moves in that same face; an
interior point moves inside the target. Every point is constrained to remain
inside the target, so edge points cannot run past the target edge endpoints.
Point-to-hull memberships are fixed.

Before optimization, approximate face/edge coordinates are orthogonally
projected to their identified planes. For this input the largest adjustment is
3.70004e-7. This removes the previously measured rounding-scale coverage errors.
The fixed target vertices are not modified.

Changing --incidence-tol changes the model and should not be used casually as an
optimization tuning parameter. It is for classifying noisy geometric data.

## Why the optimization is convex

Use local affine coordinates

    p_i(z) = a_i + B_i z,

where the columns of B_i span the intersection of the prescribed face planes.
These equalities are therefore eliminated exactly in the mathematical model.
The remaining target-containment constraints are linear inequalities in z.

For every pair (i,j) listed together in at least one covering hull, impose

    ||p_i(z) - p_j(z)||^2 <= s.

Minimize s; the final diameter is sqrt(s). These are convex quadratic epigraph
constraints. All repeated pair constraints across hulls are removed. The diameter
of a finite convex hull equals the largest distance between its generating
points, so no diameter pairs are omitted.

SLSQP is used to find a candidate, not as an unsupported assertion of global
optimality. The separate lower-bound computation tests how close that candidate
is to the global optimum of the convex model.

## Why the constraints preserve coverage

Merely minimizing pair distances without coverage constraints would be invalid.
The program checks the following stronger combinatorial certificate BEFORE any
optimization.

The projected reference boundary admits a conforming triangulation with 42
vertices, 120 edges, and 80 triangles. Each triangle's three point indices belong
together to at least one of the four hulls. Every edge has exactly two incident
triangles on the closed boundary. On each target face, polygon unions verify
that its reference triangles tile it without positive-area gaps or overlaps.
Every boundary segment of a target face lies on an original target edge.

Consider ANY feasible new positions, and map every reference triangle affinely
to the triangle with the corresponding new point positions. The maps agree on
shared edges and therefore give a continuous map on the target boundary.

On a target face F, this map takes F into F. Every edge of F maps into itself and
its endpoints are fixed. On the boundary of F the map is homotopic to the identity
through the straight-line homotopy, staying on the same boundary edges. Hence
its winding number about any interior point of F is one. The degree argument
implies that the map is onto F. The mapped triangulation may fold: this does NOT
invalidate surjectivity or the covering argument, and no triangle-orientation
constraints are needed.

Each mapped triangle lies in its owner hull because its vertices belong to that
hull. Thus the four hulls cover the entire target boundary. Finally point 27 is
listed in all four hulls and remains inside the target. For any boundary point b
covered by K_i, the segment from point 27 to b lies in K_i by convexity. These
segments cover the entire target polyhedron. This proves coverage for every
exactly feasible point set in this incidence model.

For other input files the automatic Delaunay triangulation might not respect
memberships even when another triangulation would. The program then stops with
a diagnostic rather than optimizing an unverified cover. It also rejects input
without a common point, invalid fixed vertices, or inconsistent face incidences.

## What 'partition' means here

The output is a cover by four convex hulls. Their interiors are allowed to
overlap; this is the cover condition used in the preceding checks. For a disjoint
partition without a convexity requirement, assign each point to the first hull
containing it. Each resulting part is a subset of its hull, so its diameter cannot
increase. The program does NOT enforce an interior-disjoint partition into four
convex cells. That additional requirement is a different optimization problem.

## Independent numerical checks

The optimizer recomputes each output diameter from all co-hull pairs. It also
checks containment and prescribed plane residuals, and forms the actual union
of projected hull patches on EACH of the 15 target faces. It does not infer
coverage from the sum of hull volumes.

For the delivered result, the maximum prescribed-plane residual is 1.11022e-16;
the largest missing face area is 5.74837e-18, consistent with roundoff. Face areas
are not distances; the report deliberately distinguishes these quantities.

The supplied independent verifier can additionally calculate all 15 nonempty
hull intersections, inclusion-exclusion volumes, and a global coverage-distance
bound:

```bash
python3 verify_trd_cover.py r_dod_4_optimized.txt \
    --json r_dod_4_optimized_coverage.json \
    --distance-tolerance 1e-13
```

Its measured missing-volume result is -2.22e-16 (roundoff around zero, not a
negative geometric volume), and its numerical coverage-distance estimate is
about 1.00e-16. Neither verifier uses outward-rounded interval arithmetic.

## The lower bound

For each convex squared-distance function f_e and candidate z0, the tangent

    f_e(z0) + grad(f_e)(z0) . (z-z0)

is a global lower bound on f_e(z). Replacing all f_e by these affine functions
and retaining the linear feasible domain yields a linear program whose optimum
is a lower bound on the original minimum squared diameter.

The JSON also contains nonnegative distance and containment multipliers. The
distance multipliers are normalized to sum to one. Their weighted affine lower
function is minimized over a deliberately loose, explicit coordinate box; this
corrects for approximate, rather than exact, dual stationarity. A small disclosed
numerical allowance is then subtracted. This is a practical numerical diagnostic,
not rigorous directed-rounding verification.

If the solver stops early, the program retains its best numerically feasible
iterate, reports the solver status and the lower-bound gap, and does not label
an unconverged point as a proven optimum. Ctrl-C during SLSQP similarly retains
and validates the best feasible incumbent before saving.

## Restart with the SAME incidence model

An optimized point can reach an additional target face. Reclassifying it from
scratch could unnecessarily freeze additional coordinates. To avoid that, reuse
the saved model:

```bash
python3 optimize_partition.py r_dod_4_optimized.txt \
    --model r_dod_4_optimized.json \
    --output r_dod_4_polished.txt \
    --report r_dod_4_polished.json \
    --maxiter 5000 --ftol 1e-14
```

Use distinct output/report paths so the original model is not overwritten.
Random restarts are not needed to escape local minima in this convex model.
To substantially improve below 0.96976776, a larger family must be considered,
such as changing hull memberships or prescribed boundary incidences. The
present numerical lower bound does not rule out better covers in those families.

## Tests and visualization

```bash
python3 test_optimize_partition.py
python3 show_hull_diameters.py r_dod_4_optimized.txt \
    --matplotlib --target-vertices 23
```

The viewer is the separate program from the preceding response; it is not a
runtime dependency of this optimizer. Nine regression tests cover parsing,
derivatives, the reference mesh, numerical optimization and its lower bound,
fixed vertices, file round-trips/model restarts, twelve large random
face-preserving deformations, and three distant optimization starts.

The JSON reports include all hull diameters and witness pairs, coverage
residuals, solver status, numerical lower-bound data, and the reusable reference
mesh/incidence model. Point numbering remains zero-based throughout; displayed
hull numbers are one-based.
