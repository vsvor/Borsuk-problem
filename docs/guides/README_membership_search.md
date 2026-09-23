# Coverage-preserving membership search

`search_memberships.py` extends `optimize_partition.py`. It changes the vertex
membership lists, solves the convex coordinate problem for each promising
candidate, and accepts a change only after an independent coverage check.

Python >= 3.10. Keep both scripts in the same directory.

```bash
python3 -m pip install numpy scipy shapely
python3 search_memberships.py r_dod_4_optimized.txt \
    --model r_dod_4_optimized.json \
    --patch-size 4 --split-star-limit 4 --rounds 3 \
    --max-candidates 20000 \
    --output r_dod_membership_search.txt \
    --report r_dod_membership_search.json
```

The input is never overwritten. The output has the original five-line format;
point numbering is unchanged. The membership lists are sorted but their hull
order is unchanged. Reports can be reused as `--model` on a restart. Supplying
`--model` is important: it retains the original face/edge constraints rather
than inventing additional constraints from an optimized point's new position.
The output remains compatible with `show_hull_diameters.py` and the previous
optimizer. No chart or graphical backend is required for the search.

## Main function

```python
import json
from pathlib import Path
import optimize_partition as op
from search_memberships import check_membership_changes

data = op.read_data(Path("r_dod_4_optimized.txt"))
saved = json.loads(Path("r_dod_4_optimized.json").read_text())["model"]
model = op.build_model(data, fixed=23, incidence_tol=1e-6, saved_model=saved)
z = model.coordinates(data.points)

result = check_membership_changes(
    model, z,
    max_rounds=3,
    max_patch_size=4,
    split_star_limit=4,
    random_trials=0,
    max_candidates=20000,
    improvement_tol=1e-8,
)
print(result.diameter, result.improved, result.report["status"])
op.write_data(Path("better.txt"), result.points, result.groups, result.diameter)
op.atomic_text(Path("better.json"), json.dumps(result.report, indent=2) + "\n")
```

When already inside the original optimizer, pass its existing `model` and `z`;
there is no need to rebuild them. Use `result.model` and `result.z` for subsequent
coordinate optimization, because the co-hull pair constraints have changed.
Do not keep using the old `model.pairs` or the old membership lists.

The input `model`, its arrays, and `z` are not modified by the search. Set
`polish_start=False` to skip the initial fixed-membership coordinate solve.
Otherwise this initial improvement is recorded separately from subsequent
membership-search improvements.

## Test a particular proposed membership change

```python
from search_memberships import try_membership_change

# proposed_groups must be four complete lists of zero-based point indices.
trial = try_membership_change(model, z, proposed_groups)
print(trial.status, trial.report)
if trial.status == "improved":
    model, z = trial.model, trial.z
    points = model.positions(z)
```

This function first verifies the retained boundary-mesh certificate, then
rebuilds every co-hull pair constraint. It can reject a candidate using an
immutable-vertex or tangent-LP lower bound. Otherwise it reoptimizes its
coordinates, recomputes the actual diameter, and checks coverage. A feasible
result is retained even when SLSQP's termination status is unsuccessful; solver
status and objective bounds are reported separately. An unresolved lower bound
is not reported as proof that improvement is impossible.

`prune=False` (CLI `--no-prune`) disables bound/dominance screening. This is useful
for independently checking screening or for extremely small improvement scales.
The ordinary floating-point calculation is not an interval certificate.

## Membership neighborhood

At a fixed point of the coordinate optimizer, the program generates:

1. Every single membership deletion except deletion of the protected common
   point, plus simultaneous trimming of memberships unnecessary for the chosen
   triangle labels.
2. Transfers of every connected same-owner boundary-triangle patch of size at
   most `max_patch_size`. Adjacency means sharing a mesh edge; patches may cross
   a target polyhedron edge.
3. Transfers of an entire source-owned vertex star, regardless of its size. This
   means all triangles incident to one point and currently assigned to one hull.
4. Split-star transfers for stars of size at most `split_star_limit`, assigning
   their triangles to possibly different destination hulls.
5. Optional randomized simultaneous patch transfers. Several intermediate
   changes are combined before evaluating the result; an individual intermediate
   change need not improve the diameter.

For a triangle labeling, hull k receives the vertices of all triangles labeled
k and the protected common point. Original memberships of points not used in
the boundary mesh are retained. Consequently, changing a triangle owner makes
coordinated additions and deletions in the membership lists. This differs from
simply moving a point between lists without checking what portion of the
boundary was lost.

By default `max_patch_size=3`, `split_star_limit=4`, and no random trials are
used. No-op and duplicate patterns are removed. Four hulls with at least four
listed points each are retained. The coordinates of the first 23 points are
fixed, but their memberships are not explicitly frozen.

### Wider randomized search

```bash
python3 search_memberships.py r_dod_4_optimized.txt \
    --model r_dod_4_optimized.json \
    --patch-size 4 --split-star-limit 4 \
    --random-trials 5000 --random-moves 4 --seed 7 \
    --rounds 3 --max-candidates 30000 \
    --output r_dod_membership_random.txt \
    --report r_dod_membership_random.json
```

`random_trials` is the number of raw random proposals **per round**, not the
number of distinct candidates or nonlinear solves. `random_moves` bounds the
number of simultaneous patch mutations (at least two). A new seed generates a
different finite sample. Membership neighborhoods are rebuilt after an accepted
round-best move. The search is greedy between rounds; it does not claim to
explore every sequence of neutral or temporarily worsening optimized states.

## Why coverage survives coordinate optimization

The original optimizer validates a reference triangulation of all target faces.
A candidate must assign every reference triangle to at least one hull containing
its three vertex indices. The common point must still belong to all four hulls.

Every new point remains in its prescribed target face or edge. Affine maps of
the reference triangles therefore form a continuous map of each target face
into itself; on its boundary they preserve each target edge and fix its ends.
Such a boundary map is homotopic to the identity and has degree one about every
interior point. The face map is onto, even if triangles fold or overlap. Each
mapped triangle lies in its owner convex hull, so the hulls cover the boundary.
Their common point and convexity extend coverage to the interior.

This argument keeps the coordinate constraints convex; triangle orientation or
non-overlap constraints are not needed for a **cover**. Interior-disjoint convex
cells are not enforced. Boundary unions and affine residuals are also checked
numerically after optimization. Face-area errors are not Euclidean distance
errors; the report labels them separately.

A candidate rejected because one reference triangle has no owner might still
cover the target by a different triangulation. The retained-mesh condition is
sufficient, not necessary. This program does not flip mesh edges, change point
incidences, add points, or remove the protected common point.

## Pruning and objective comparisons

For candidate hull lists J, the subproblem remains

    minimize s,
    subject to ||p_i(z) - p_j(z)||^2 <= s for every co-hull pair,
               z in the original affine/linear target domain.

Only **new optimized coordinates** determine acceptance. In particular, a
candidate whose initial diameter is worse is not discarded for that reason.
Adding memberships alone cannot improve the exact optimized value because it
adds pair constraints. Superset-constraint pruning is enabled only when the
incumbent lower bound is already within `improvement_tol` of its upper bound.
Otherwise those candidates are not discarded by dominance.

For a co-hull pair of fixed target vertices, their immutable distance is a lower
bound for that candidate. A tangent-LP bound may also exclude improvement:
convex squared distances lie above their affine tangent functions. The existing
optimizer computes dual residual corrections and a disclosed numerical safety
allowance. Bounds below the incumbent do not guarantee attainability and are
not themselves reported as improved partitions.

The returned maximum diameter is recomputed from point pairs. It is not merely
the solver's epigraph variable. Improvements must exceed the positive absolute
`improvement_tol`, default 1e-8. Set `--improvement-tol` to change this resolution.

## Limits, checkpoints, and status

`max_candidates` is a whole-call limit on distinct nonincumbent proposals,
including screened/rejected ones. Candidate uniqueness is per round, because
reoptimization and the current target for improvement change across rounds.
`max_rounds` limits the number of best-improvement rounds. `time_limit` / CLI
`--time-limit` is checked between candidates; it is not a hard interruption of an
individual LP/SLSQP call. `maxiter` limits each coordinate solve.

The CLI writes a verified baseline checkpoint and then writes a checkpoint for
every new best verified candidate. A checkpoint report is labeled as such and
contains a reusable incidence model. The final report replaces it with complete
diagnostics. Ctrl-C normally retains the current best; input files are untouched.

Search statuses include `no_improvement_found`, `candidate_limit`, `time_limit`,
`round_limit`, and `interrupted`. The latter four indicate a truncated search.
Candidate statuses separately distinguish certificate failure, each pruning
reason, an improvement, a bounded non-improvement, and an unresolved numerical
case. No negative local-search result is a global membership-optimality proof.

## Measured result on the supplied optimized configuration

Starting value: 0.9697677618614492. No improved membership pattern was found at
absolute improvement tolerance 1e-8.

Deterministic search (patch size 4, split-star limit 4):

- 2,774 distinct proposed patterns; 65 failed the retained-mesh certificate.
- 734 were supersets of the incumbent co-hull constraints.
- 1,949 were excluded by fixed-target-vertex distance bounds.
- 25 were excluded by tangent-LP bounds.
- 1 was reoptimized, giving 0.9776563472603288, with a matching numerical lower
  bound sufficient to exclude improvement at the requested tolerance.

Adding 5,000 randomized compound proposals (up to four mutations, seed 7)
produced **6,597 distinct patterns in total**, including the deterministic ones.
Of these, 65 failed the mesh certificate, 2,133 were dominated constraint
supersets, 4,373 failed the immutable-vertex bound, 25 failed the tangent-LP
bound, and the same one remaining candidate required reoptimization. There were
no numerically unresolved candidates in these runs. Neither run reached its
candidate limit. The search ended after the first round because no improvement
was found.

The final maximum missing target-face area was about 1.73e-17 and the maximum
affine-plane residual about 1.11e-16. These are numerical checks, not exact zeros
or newly computed Hausdorff-distance errors. The search did not establish that
all membership changes are unhelpful; different meshes/incidences or larger
coordinated changes remain outside this test.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python3 -m unittest -v test_search_memberships.py
```

All 11 tests passed in the delivered run. They cover unchanged inputs, common
point protection, ownerless-triangle rejection, dynamic co-hull constraints,
positive improvement from removing a deliberately redundant membership,
automatic detection of that improvement, pruned/unpruned agreement, randomized
triangle relabeling with large face-preserving deformations, saved-model
restarts, argument validation, and budget limits. The positive improvement test
uses a deliberately worsened configuration; it is not a claimed improvement on
the user's already optimized partition.

Implementation reference: SciPy official documentation for SLSQP and linprog:
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
