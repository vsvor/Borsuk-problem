# Parameterized four-truncation search

Keep `parametric_truncation_search.py` and `partitions_slsqp.py` together.
Python 3.10 or later is required.

```bash
python3 -m pip install numpy scipy shapely
python3 parametric_truncation_search.py \
    --a 0.02 --b 0.01 --case adjacent \
    --starts 5000 --workers 6 --seed 17 \
    --seconds-per-start 5 --output-dir search_ab
```

The wrapper generates the target and calls the existing **unchanged** SLSQP
multistart engine. The engine uses clipped Voronoi/power diagrams for starting
memberships, then optimizes their point coordinates with hard target-face/edge
constraints. This is NOT a six-plane-through-the-origin search. The parameters
`a,b` are fixed throughout one invocation; the program does not optimize them.

## Geometry and signs

The original, unscaled rhombic dodecahedron is

```
|x|+|y| <= 1/sqrt(2)
|x|+|z| <= 1/sqrt(2)
|y|+|z| <= 1/sqrt(2).
```

Both target families retain the slab

```
-1/2+b <= x <= 1/2+a.
```

Thus the lower halfspace in matrix form is **-x <= 1/2-b**. There is no sign
change in the user's parameter b. The slab has width `1+a-b` and midpoint
`(a+b)/2`. In particular `a=b` translates a width-one slab, whereas `b=-a`
changes its width symmetrically. The restriction `a+b >= 0` is checked on the
specified decimal parameters, not rounded away as a numerical violation.

The two planes not specified by a,b are selected explicitly:

| `--case` | Other inequalities |
|---|---|
| `adjacent` (default) | `y >= -1/2`, `z >= -1/2` |
| `opposite` | `-1/2 <= y <= 1/2` |

At a=b=0, the adjacent representative has untruncated +y,+z vertices. It is
obtained from the earlier adjacent target (untruncated +x,+y) by a coordinate
permutation. The opposite representative leaves +z,-z untruncated.

Each run prints the actual retained inequalities. All saved coordinates,
diameters, and stopping thresholds are in the ORIGINAL physical units. The
old solver's internal recentering/normalization is reversed on output.

## Valid parameter values and degeneracies

`a+b >= 0` alone is not a nonemptiness condition. We separately require

```
max(-1/2+b, -1/sqrt(2)) < min(1/2+a, 1/sqrt(2)).
```

For these two choices of other planes this is the exact full-dimensionality
condition: y=z=0 is admissible and the x-projection of the base body is
[-1/sqrt(2),1/sqrt(2)]. Numerically, very thin targets are rejected as well.
The current conservative radius threshold is 2e-8 in physical units.

The target need not contain the origin. For example a=0.2,b=0.6 retains
0.1 <= x <= 0.7 and is a valid test case.

A requested cut can stop producing a facet. For example a > 1/sqrt(2)-1/2
puts the upper x plane entirely beyond the original target. By default the
program reports a warning and solves the actual intersection. Add

```bash
--require-four-cuts
```

to reject the parameter pair unless all four requested cuts define
positive-area two-dimensional target faces (with numerical tolerances).
A supporting plane that only touches a vertex is not counted as a facet.

Vertices are rebuilt from feasible intersections of triples of supporting
planes, rather than deforming a hard-coded 26-vertex list. Their number and
the number of remaining rhombic faces can therefore change. `target.json`
records the actual `vertex_count`, `fixed_count`, plane equations, face areas,
volume, diameter, an interior point, and numerical warnings.

## Generate only

```bash
python3 parametric_truncation_search.py \
    --a 0.02 --b 0.01 --case adjacent \
    --generate-only --output-dir geometry_ab
```

This writes `geometry_ab/target.json`, accepted by the original program:

```bash
python3 partitions_slsqp.py --vertices geometry_ab/target.json \
    --parts 4 --starts 5000 --workers 6 --output-dir independent_search
```

Geometry generation uses NumPy and SciPy; Shapely and the backend file are
only needed when starting the search.

## Search controls

All existing SEARCH flags are forwarded to the original program. Examples:

* `--starts 10000 --workers 6`: attempt count and worker count.
* `--target-diameter 0.96`: stop the whole search after a verified successful
  result at or below this value. This is not a claim that 0.96 is attainable.
* `--run-target 0.967`: stop a local solve at this value, continuing multistart.
* `--seconds-per-start 5`: cooperative budget per attempt.
* `--starts 0 --hours 6`: time-limited search without an attempt-count limit.
* `--reject-above 1.02 --reject-after 25`: optional heuristic rejection of poor
  runs. It can miss improvements; disabled unless explicitly requested.
* `--power-weight-scale 0.02`: enable weighted starting diagrams.
* `--no-bound-screen`, `--no-deduplicate`: turn off the corresponding screening.
* `--quiet`: suppress normal terminal summaries (geometry warnings remain).

For the backend's complete CLI documentation:

```bash
python3 parametric_truncation_search.py --search-help
```

Do not forward `--vertices`, `--fixed-count`, `--parts`, `--initial-model`, or
an input partition path. This wrapper constructs its own target and searches
exactly FOUR hulls. Use the original engine separately for other workflows.

## Saved files and restart safety

The output directory has:

```
target.json
best.txt
best.json
previous_best.txt
previous_best.json
state.json
```

The previous-best files appear after the second accepted record. `state.json`
is compact restart state, not a detailed per-run log. The original engine also
creates its advisory lock file. Existing best/previous-best writing, feasible
iterate retention, and pruning mechanisms are unchanged.

```bash
python3 parametric_truncation_search.py \
    --a 0.02 --b 0.01 --case adjacent \
    --output-dir search_ab --resume \
    --starts 10000 --workers 6
```

Use **the same a,b and case** on resume. The wrapper refuses to replace target
metadata in a directory belonging to different parameters. The backend also
checks the exact target hash. Use a NEW output directory for a new pair a,b;
an old best.json is not automatically a valid cover of the new body.

Without `--output-dir`, a directory name is generated from the case, a, b, and
a short parameter hash to avoid collisions between rounded decimal names.

The first `fixed_count` points of each exported partition are target vertices.
Read that count from target.json when using the prior viewer or local optimizer:
do not assume it remains 26 for all parameter values. The `.txt` format is the
same five-line format as in the previous scripts.

## Scope of the result

For each generated membership/incidence model the coordinate optimization is
convex. Searching across models remains heuristic, not a global certificate.
The output is a COVER by four convex hulls; their interiors may overlap.
The four target truncation planes must not be confused with internal partition
interfaces: no origin-passing condition is imposed on the latter.

Coverage is inherited from the backend's reference-complex deformation
construction and its numerical geometry checks. Geometry, feasibility checks,
and lower-bound screening use floating-point arithmetic, not outward-rounded
interval arithmetic. Numerical area residuals are not Hausdorff distance bounds.

## Validation

Run the included fast tests:

```bash
python3 -m unittest test_parametric_truncation -v
```

Twelve tests cover both unshifted targets, the sign of b, decimal a+b validation,
nonfinite/empty/thin cases, missing facets, topology changes, targets excluding
the origin, independent Qhull halfspace intersections on 18 shifted targets,
CLI handling, and refusal to overwrite a mismatched target.

The included validation JSON also records actual two-worker SLSQP searches,
checkpoint resume, and target-based early stopping. Example outputs are
functional tests, not claimed optima for their parameter pairs.

## Relevant implementation documentation

The backend and its original README are included. Official numerical interfaces:

* https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
* https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
