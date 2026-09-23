# Balancing two parameterized truncation covers

`balance_truncations.py` continues saved partition models as the parameter changes,
reoptimizes their coordinates with SLSQP, and brackets a crossing using numerical
primal/dual bounds. It does not run a new random search at each parameter value.

## Families and scope

Let R be the original rhombic dodecahedron:

    |x|+|y| <= 1/sqrt(2)
    |x|+|z| <= 1/sqrt(2)
    |y|+|z| <= 1/sqrt(2).

The default `--case adjacent` retains y >= -1/2 and z >= -1/2. The two families are:

* A, selected with `--dummy`: x >= -1/2 + delta. The upper x cut is redundant;
  `--dummy-a 0.5` places it at x=1. This is a three-effective-truncation target.
* B, selected with `--upper`: -1/2 <= x <= 1/2 + delta. This has four effective
  truncations throughout the example's bracket.

`--case opposite` instead retains -1/2 <= y <= 1/2 for both families. Input seeds
must be in the matching coordinate orientation. A nonredundant `--dummy-a` is
rejected; dummy_a must exceed 1/sqrt(2)-1/2 with a small numerical margin.

For every supplied reference model, its point count, memberships, and original
boundary-face incidences stay fixed. The target vertices themselves move to the
new target vertices. All other coordinates allowed by those incidences are
reoptimized. This is not a six-plane-constrained optimization.

With several seeds per family the program evaluates every model and takes the
minimum of their optimized diameters. The crossing is for these two finite model
pools. It is NOT asserted to be the crossing of globally optimal covers among all
possible partitions. The sets produced are convex-hull covers; interiors may
intersect. The program does not prove a universal containment theorem for all
unit-diameter sets.

## Installation and first run

Use Python 3.10 or later. Keep these three scripts together:

    balance_truncations.py
    partitions_slsqp.py
    parametric_truncation_search.py

The latter two are included unchanged.

    python3 -m pip install numpy scipy shapely
    python3 balance_truncations.py \
        --dummy /path/to/a_dummy_b_001/best.json \
        --upper /path/to/a_001_b_0/best.json \
        --lo 0 --hi 0.02 \
        --delta-tol 1e-11 \
        --output-dir delta_balance

A directory is accepted instead of its `best.json`. The source delta is inferred
from the source target, so the two seeds need not have the same initial delta.
The source target is checked against the analytic halfspaces and independently
constructed vertices. The program validates the original source coverage model.

Prefer the JSON checkpoints: their original reference incidences are retained.
A .txt file with a matching adjacent .json is also accepted. A bare five-line
.txt file requires `--seed-delta 0.01`, and must pass the original input fallback's
boundary-triangulation/common-point check. That fallback may reject a genuine
cover whose optimized coordinates no longer supply the expected reference mesh.
It never treats a missing reference certificate as success. If this happens, use
the engine's original `best.json` instead. Local `optimize_partition.py` reports
are also accepted together with their matching .txt file.

There is no automatic rotation of arbitrarily oriented input targets. Seeds from
the previous parameterized wrapper already have the required orientation.

## Multiple initial models

Repeat either argument:

    python3 balance_truncations.py \
        --dummy run_A/best.json --dummy run_A/previous_best.json \
        --upper run_B/best.json --upper run_B/previous_best.json \
        --lo 0 --hi 0.02 --output-dir delta_balance_pool

Every selected model must remain valid on the bracket. A model is not silently
dropped because it becomes incompatible or its solver result is inconvenient.
Additional memberships discovered by a global search can change the crossing.

## The numerical result in this package

The two included models reproduce, at delta=0.01:

    A = 0.9649655753222...
    B = 0.9656571899538...

The A seed was obtained by reflecting an earlier saved three-truncation cover;
it starts at delta=0. The B seed was obtained from a 150-start search at delta=0.01.
They were not taken from the user's unuploaded local search directories.

A run from [0, 0.02] gave the numerical bracket:

    0.008811423364095392 <= delta* <= 0.008811423373408615.

Both endpoints round to 0.0088114234 at eight significant digits. At the midpoint,

    delta = 0.008811423368752004
    A     = 0.9653991786498369
    B     = 0.9653991786503513.

The small discrepancy is smaller than the requested root-bracket uncertainty.
The separate run starting at [0.0088, 0.0089] gave a consistent bracket. Numerical
results can differ in the last few digits across library versions.

Reproduce the included example, from the package directory:

    python3 balance_truncations.py \
        --dummy examples/dummy_seed/best.json \
        --upper examples/upper_seed/best.json \
        --lo 0 --hi 0.02 --output-dir my_balance

`examples/balanced/result.json` records the bracket, source paths, primal/dual
bounds, and the selected solution at the midpoint. `validation.json` records the
additional tests. All length values are physical lengths; there is no changing
normalization in the comparison.

## Precision and stopping

The default `--delta-tol 1e-11` is the requested FULL bracket width. It is not
SLSQP's objective tolerance. At convergence the midpoint's bracket error is at
most half that width, conditional on the numerical bounds and the model pool.
The report also checks whether both endpoints round identically to eight decimal
places and to eight significant digits. These are different requests.

At each delta, let [LA,UA] and [LB,UB] be the numerical lower/upper bounds for the
two model pools. The sign of A-B is resolved only when

    LA - UB > 0        or        UA - LB < 0.

An unresolved midpoint never determines which half to discard. The program
tries quarter-point probes. If these cannot narrow the bracket, it returns
`objective_uncertainty_or_equality_plateau`, retaining the current interval. It
does not print eight digits as certified merely because two approximate values
look equal. A flat interval of equality need not determine a unique delta.

Possible status values include:

* `converged`: requested width achieved with resolved endpoint signs;
* `not_bracketed`: initial endpoint signs are unresolved or do not differ;
* `objective_uncertainty_or_equality_plateau`: the remaining interval cannot be
  resolved at the present numerical precision;
* `max_iterations` or `floating_point_limit`.

The exit code is 0 for convergence, 2 for an unresolved search or input error,
and 130 for interruption. An endpoint-zero case is reported as unbracketed rather
than pretending a strict sign was obtained. A new interval around that endpoint
can resolve a crossing; exact equality or a plateau may remain nonunique.

Relevant controls:

    --ftol 1e-13             initial SLSQP tolerance
    --maxiter 1200           per polishing attempt
    --polish-attempts 3      progressively tighter attempts when needed
    --diameter-tol 3e-12     desired primal/dual diameter gap
    --bound-guard 1e-12      squared-diameter arithmetic allowance
    --upper-guard 5e-13      diameter arithmetic allowance
    --max-bisections 80

Do not reduce the guards solely to force the requested digits to print. They
are disclosed numerical allowances, not proven roundoff bounds. If a lower bound
is weak, the program can stop with a wider interval even after SLSQP reports
success. The final whole-root bracket, not SLSQP's status alone, determines the
reported root precision. No comparison with an incumbent at a different delta
is used to prune a continuation solve.

All bounds and coverage checks use floating-point arithmetic, including the
quadric coefficients. They are not outward-rounded interval or exact-arithmetic
certificates. They measure numerical precision within the supplied models, not
uncertainty caused by untested membership/incidence patterns.

## Why changing delta preserves the model

The target halfspace normals do not change and their offsets are affine in delta.
As long as the target face lattice stays unchanged, every target vertex is an
affine function of delta. The source vertices are identified by their incident
planes, not by a newly sorted coordinate order.

For every original reference point, the program finds nonnegative barycentric
weights on the target vertices of its prescribed face. Keeping those weights
fixed gives an affine moving anchor inside that face. A nullspace basis supplies
all additional face/edge motions. Thus each optimized point has the form

    p_i(delta,z) = anchor_i(delta) + B_i z,

with constant B_i. All target-containment constraints are linear, and the
squared co-hull distance inequalities are convex quadratics for fixed delta.
SLSQP uses analytic Jacobians, all co-hull pair constraints, and repaired feasible
iterates. The coordinate box has half-width 2, larger than the base body's
sqrt(2) diameter; it does not eliminate a geometrically feasible position.

The original certified reference complex, not a fresh Voronoi reconstruction at
the optimized coordinates, supplies the coverage argument. Transporting its
vertices face-preservingly to the combinatorially identical target and extending
over a subdivision gives a map onto the target. Each mapped simplex lies in an
owner hull. Folded simplices and overlapping hulls are allowed. At every solve,
independent boundary-patch unions, affine residuals, and containment residuals
are checked. The analytic target is independently reconstructed at the bracket
endpoints and when a solution is saved.

If a target vertex acquires or loses an incidence, the script rejects the
continuation. It does not silently freeze the wrong topology. Choose a seed and
bracket within one target-combinatorial regime, or obtain new reference models
on the other side of a transition.

## Numerical lower bounds

The code solves a linear relaxation formed by tangent planes to the squared
distance functions. Its nonnegative multipliers supply a convex quadratic
Lagrangian Q(z). For any feasible z,

    maximum squared distance >= Q(z).

A trial minimizer of Q is computed on its positive eigenspace. Convexity gives
an affine minorant there, and its remaining gradient is minimized over the
explicit coordinate box. Therefore approximate stationarity or a singular
Hessian does not have to be treated as exact. Tiny LP weights can also be
removed: each resulting nonnegative normalized weight vector is checked as a
separate lower bound. The maximum valid computed lower bound is used, less the
arithmetic guard.

For a pool, the lower and upper bounds are respectively the minima of its branch
lower and upper bounds. A lower bound from only the currently winning branch
would not bound the minimum over a larger model pool; this implementation does
not make that error.

## Output and reusing results

Only the current selected pair, the preceding saved pair, and compact bracket
state are written. There is no full iteration history or run log:

    dummy.txt                 upper.txt
    dummy.json                upper.json
    result.json               bracket.json
    previous_dummy.txt        previous_upper.txt
    previous_dummy.json       previous_upper.json
    previous_result.json

Intermediate solution records are replaced when a better balanced upper bound
is found; the final selected pair is evaluated at the bracket midpoint. Thus the
last saved pair is the root estimate, not necessarily the smallest objective
encountered away from the root. On normal termination each current text file
matches its JSON. Updates are atomic per file, not a transactional multi-file
snapshot. On an interruption during writes, matching JSON files are authoritative;
their coordinates can regenerate the text exports.

`--quiet` suppresses record-improvement messages and the final terminal summary.
No seed is overwritten. Use a fresh output directory when reading these exported
continuation checkpoints as new seeds:

    python3 balance_truncations.py \
        --dummy delta_balance/dummy.json --upper delta_balance/upper.json \
        --lo 0.0088 --hi 0.0089 --output-dir delta_balance_refined

The new JSON format retains its original source certificate and current warm
start. It is understood by this script; it is not a native global-search or
`optimize_partition.py --model` checkpoint. The five-line .txt exports are the
usual coordinates and memberships and can be displayed with the previous viewer.
Use the saved `fixed_count` (23 for the example dummy family, 26 for the upper
family) instead of the earlier default of 23 for both.

## Tests

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest -v test_balance_truncations

The 15 tests cover exact synthetic roots, ambiguous midpoint signs, invalid
brackets, dual lower bounds away from stationarity, parameter signs, source
validation, target transport and topology changes, analytic derivatives,
reproduction of both seed values, restart records, text/JSON matching, multiple
models, and eight-significant-digit root bracketing.
