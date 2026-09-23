# Multistart polyhedral-cover search with SLSQP

`partitions_slsqp.py` is a standalone replacement search engine for the **three-dimensional** problem in `partitions3d.py`. It uses NumPy, SciPy and Shapely; it does not import PyTorch. Python 3.10 or later is required. The test environment used NumPy 2.3.5, SciPy 1.17.0 and Shapely 2.1.2.

The program is a heuristic global search over reference partitions, with convex coordinate optimization for each reference's fixed memberships and target-face incidences. It does **not** certify a global optimum over all covers.

## Installation and first run

```bash
python3 -m pip install numpy scipy shapely

python3 partitions_slsqp.py r_dod_4_0966.txt \
    --fixed-count 23 --starts 10000 --workers 6 \
    --output-dir search --target-diameter 0.966
```

The first 23 points define the fixed target polyhedron. The supplied four membership lists are polished once, then the program generates new Voronoi partitions with new points, memberships and boundary incidences. The input's stored diameter is **not** trusted; every saved diameter is recomputed from all co-hull point pairs.

`--starts` counts **additional random attempts** in this invocation; the initial input polish does not consume a start. A start rejected because of bad/degenerate geometry, a bound, or a duplicate also counts toward this budget. Use `--no-initial` to start from scratch while still obtaining the target from the input file.

To search a different target, put its distinct extreme vertices in a JSON array:

```bash
python3 partitions_slsqp.py --vertices target_vertices.json \
    --parts 4 --starts 10000 --workers 6 --output-dir other_search
```

Only full-dimensional convex **3-D** targets are supported. The number of cells is configurable. General-dimensional and spherical target problems from possible other variants of the original code are not implemented here.

## Output: best and previous best, not run logs

The search directory contains:

- `best.txt`, `best.json`: the current best verified cover and its original reference model.
- `previous_best.txt`, `previous_best.json`: the incumbent immediately before the latest improvement. These do not exist until there has been a second record.
- `state.json`: small cumulative counters, the base random seed and the next submitted start ID. It does not contain individual run histories.
- `.search.lock`: an advisory directory lock, not a log.

There are no per-iteration files and no per-run trace files. The terminal prints only new records and a final summary (plus important errors). `--quiet` suppresses records and the final summary. Per-run candidates and a small elite seed pool exist in memory only.

Both `.txt` files retain the original five-line format: cell count, actual diameter, point count, membership lists, coordinates. Their first 23 points remain the target vertices for the supplied example. They can be read by the previous hull viewer.

Writes use temporary files followed by atomic replacement. The JSON is authoritative, and the redundant text export is regenerated on resume. Two controllers may not write the same directory at once. Do not manually remove `.search.lock` while a search is using it. The OS releases the advisory lock when a process exits.

## Restart and time budgets

```bash
python3 partitions_slsqp.py r_dod_4_0966.txt \
    --fixed-count 23 --resume --output-dir search \
    --starts 0 --hours 6 --workers 6 --target-diameter 0.966
```

`--starts 0` means there is no attempt-count limit. `--hours` applies to the current invocation. Resume restores the best cover, its **original** reference geometry, the base random seed and the next start ID. It never re-infers incidences from optimized points. Those points can have reached additional faces, and re-inferring constraints would incorrectly shrink the search family.

Other settings come from the current command; repeat options such as `--power-weight-scale` when resuming a weighted search. An explicitly different `--seed` is rejected on resume. In-flight IDs from a stopped/crashed invocation are skipped, not replayed. The in-memory topology cache and elite pool are not saved; the elite pool is rebuilt starting with the incumbent.

`--seconds-per-start 5` places a cooperative wall-time budget on seed generation, diagram construction and local optimization. It is checked at numerical-iteration boundaries, not by forcibly killing SLSQP/Qhull/LP calls, so it is not a hard real-time limit. The whole-search time budget is also cooperative.

Ctrl-C preserves completed checkpoint records. During an active SLSQP solve the program tries to retain its best feasible iterate; during seed generation or geometry construction there may be no completed cover from that attempt to save. Spawned workers handle a shared stop request while the parent handles Ctrl-C. A force-kill cannot save unfinished work, but previously completed JSON checkpoints remain usable.

## Early stopping: three logically different cases

### A good-enough feasible result

```bash
# Stop the entire search, but only after the resulting cover is verified:
--target-diameter 0.966

# Stop each local solve at a feasible D <= 0.967, but continue the global search:
--run-target 0.967
```

For minimization, a value **below** a threshold is good. `--run-target` intentionally sacrifices further polishing once that goal is met. Choose it below the current incumbent when the goal is improvement; choosing a loose threshold can terminate useful runs prematurely. Both thresholds refer to the **actual** maximum pairwise diameter of the candidate coordinates, not the epigraph variable or a penalty loss.

### A numerical lower bound rules out improvement

This is the default rejection mechanism for poor topologies. First, a fixed-fixed pair gives an elementary lower bound. During SLSQP, a tangent linear-programming relaxation gives a stronger lower bound for the fixed-incidence convex problem. If the bound is at least `current_best - improvement_tol`, the attempt is stopped.

The tangent bound uses nonnegative dual weights, normalization, explicit correction for stationarity residuals over a coordinate box, and a numerical safety allowance. This is **not** an outward-rounded interval-arithmetic certificate. LP failure does not cause a promising topology to be rejected.

Control the frequency with `--bound-start 6 --bound-every 8`. To turn off bound-based early screening use `--no-bound-screen`. To rerun repeated topologies as well, add `--no-deduplicate`. Final lower-bound diagnostics may still be computed even when screening is disabled.

### Optional aggressive heuristic

```bash
--reject-above 1.02 --reject-after 25
```

This stops a run if its best feasible diameter is still above 1.02 after 25 coordinate-optimizer iterations. **It can discard a topology that would ultimately improve the incumbent.** It is disabled by default. A large current objective is an upper bound, not a proof that a run is hopeless.

## Pipeline and departures from the original implementation

The original `multiple_runs()` repeatedly obtains a spherical seed configuration, constructs clipped Voronoi cells, and uses Adam to move their vertices. Its local loss is a nonsmooth maximum diameter plus a penalty on selected boundary-plane residuals.

The replacement preserves that broad pipeline, with the following explicit changes:

1. **Direct halfspace construction.** The cell of site `a_i` is intersected directly with the target using
   `2(a_j-a_i) . x <= |a_j|^2 - |a_i|^2`.
   No artificial faraway points are needed. A Chebyshev-center LP supplies a strict interior point when the site itself is unsuitable. Empty and numerically unresolved cells are rejected.
2. **Hard geometry constraints.** A boundary point moves within its original target face/edge; target vertices stay fixed, and every movable point remains inside the target. Face equalities are eliminated with orthonormal null-space bases instead of penalized. This is stricter and geometrically safer than permitting motion along an unbounded supporting plane.
3. **Smooth epigraph formulation.** Minimize squared diameter `s`, subject to every relevant squared pair distance being at most `s`. Pair and linear-constraint Jacobians are analytic. No differentiation of a `max()` and no learning-rate schedule are needed.
4. **Coarse solve followed by polishing.** Defaults are `ftol=1e-8`, at most 120 iterations, then `ftol=1e-12`, at most 500 iterations for candidates within a physical diameter window of 0.003 of the incumbent. SLSQP failure or an iteration limit does not imply optimality; the best feasible iterate is retained and all actual distances are recomputed.
5. **New topologies, not just coordinate restarts.** Independent seed configurations and perturbed elite seed configurations create new point sets, memberships and target-face incidences. A shared topology cache avoids re-solving reference diagrams defining an already-excluded convex model. Cell-label permutations are canonicalized for at most six cells; for larger cell counts deduplication remains correct but is less complete.
6. **Process parallelism.** Workers have independent deterministic task seeds and share the incumbent bound, a stop flag and a small cache. BLAS/OpenMP threads default to one per process to avoid oversubscription. Set `PARTITION_BLAS_THREADS` before launching to override this deliberately.

`ftol` is a solver stopping tolerance, not a guaranteed number of correct digits in the diameter. Numerical containment, affine residuals and actual pair distances are checked separately.

## Seed generation and an enlarged search family

Default `--seed-method mixed` uses approximately 55% spherical starts, 25% interior random starts and 20% ball-packing starts for independent attempts. A separate default mutation probability of 0.45 substitutes perturbations of good seeds when an elite is available.

The spherical generator uses unit-norm constraints and SLSQP for repulsive energy. For four sites it uses a randomly rotated regular tetrahedron directly, avoiding repeated optimization of the same spherical arrangement. Mixed starts sometimes perturb radii and translation. `--seed-method sphere --mutation-probability 0` is closer to the original centered-sphere initialization.

The packing generator maximizes a common ball radius with SLSQP, subject to pair separation and distances to target planes. This seed-generation problem is **nonconvex**; only the later fixed-reference diameter subproblem is convex.

To include weighted Voronoi (power) diagrams:

```bash
python3 partitions_slsqp.py r_dod_4_0966.txt \
    --fixed-count 23 --starts 10000 --workers 6 \
    --power-weight-scale 0.02 --output-dir power_search
```

Their halfspaces are

```
2(a_j-a_i) . x <= |a_j|^2 - |a_i|^2 + w_i - w_j.
```

Weights are used only to **generate the reference partition**. They are not constrained to describe the final deformed hulls. The default weight scale is zero, i.e. ordinary Voronoi diagrams. Positive weight scales broaden the initialization family and can produce empty cells, which are discarded.

All internal coordinates are centered and scaled by the target diameter. Diameter thresholds, printed values and saved coordinates are in the original physical units. Mutation scales are dimensionless fractions of the target diameter; power-weight scales are in normalized squared-distance units. `--mutation-min 0.015 --mutation-max 0.18` draws logarithmically distributed coordinate-perturbation scales. `--elite-size 12` controls the in-memory elite archive.

Parallel execution and adaptive elite selection make the complete search trajectory scheduling-dependent. Each task's base random stream is reproducible from `(seed, start_id)`, but that alone does not reproduce a mutated task without its parent seeds. Saved record files include the generating reference sites/weights. Use one worker for deterministic sequential experiments.

## Optimization problem

For a fixed reference diagram, let `J_k` be its membership lists. Write each point as `p_i(z) = a_i + B_i z` in its recorded target-face affine space. The model is

```
minimize s
subject to
    |p_i(z) - p_j(z)|^2 <= s    for each pair occurring in some J_k,
    A p_i(z) <= b              for all points,
    fixed target vertices remain unchanged.
```

This is a convex quadratic-constraint epigraph problem. The minimum diameter is `sqrt(s)`. The objective stored in output is recomputed directly from the resulting hull-generating points.

The search does not perturb coordinates repeatedly within an already optimized convex model and call that a global search. Its outer loop changes the reference geometry and the discrete incidence model. There is no guarantee that finitely many attempts visit every possible model.

## Why the generated covers remain covers

A reference clipped power diagram is a face-to-face convex polyhedral decomposition of the target. Its full-dimensional cells and all their intersections form a finite polyhedral complex. The generator checks each cell against its defining halfspaces, verifies the required vertices and shared memberships, and checks total reference volume.

Take the barycentric subdivision of this **reference complex**. Map each original vertex to its optimized position and map the barycenter of each reference face to the average of its moved vertices. Extend affinely over every simplex. This gives a continuous map from the target to itself, agreeing on shared faces. A simplex originally in cell `k` maps into the convex hull of that cell's moved vertices.

A vertex on a target face stays in that face. Thus the map preserves every target boundary face. Its straight-line homotopy to the identity also preserves the boundary. The map has degree one at every interior target point, so it is onto. The union of the deformed hulls therefore covers the target. This argument does not require a point shared by every cell and also applies to five or more cells.

For the supplied arbitrary input cover, which need not still be a power diagram, a separate fallback verifies an owner-labeled boundary triangulation and a listed point common to all hulls. The reference boundary mesh and shared point then provide the face-surjection/cone argument used in the previous optimizer. If that input certificate fails, the input is not adopted as an incumbent; new reference partitions may still be searched. The warning is not a claim that the arbitrary cover is impossible.

The output is a **cover by convex hulls**, not a required interior-disjoint convex partition. The deformed reference complex can fold; overlap is allowed. Enforcing a nonfolding, interior-disjoint convex partition would be a different optimization problem.

The mathematical argument concerns exact geometry. Implementation uses floating-point reconstruction, not exact predicates or interval arithmetic. Reference validation, affine and containment residuals, and independent unions of hull patches on every target face are checked. These area diagnostics are not a computed Hausdorff distance. The regression tests additionally check full union volume by inclusion-exclusion for four and five cells, including a large deformation with no common vertex.

## Python API

```python
from pathlib import Path
from partitions_slsqp import SearchConfig, read_partition, run_search

if __name__ == '__main__':
    points, groups, _ = read_partition('r_dod_4_0966.txt')
    cfg = SearchConfig(
        starts=10000,
        workers=6,
        seed=7,
        parts=4,
        output_dir='search',
        target_diameter=0.966,
        # run_target=0.967,          # optional local good-enough stop
        # power_weight_scale=0.02, # optional enlarged family
    )
    summary = run_search(points[:23], cfg, initial_partition=(points, groups))
    print(summary['best_diameter'])
```

With multiprocessing, use a proper script and the `if __name__ == '__main__':` guard. For an interactive notebook use one worker or launch the CLI subprocess.

Low-level functions include `Target.from_vertices`, `build_diagram`, `Model`, `solve_model`, `tangent_lower_bound` and `model_from_record`. `solve_model` returns the best feasible parameter vector and a compact status dictionary. `run_search` returns a summary, not an in-memory collection of all attempts.

To polish an existing optimized file using a reference from the previous optimizer:

```bash
python3 partitions_slsqp.py r_dod_4_optimized.txt \
    --fixed-count 23 --initial-model r_dod_4_optimized.json \
    --starts 1000 --workers 4 --output-dir from_old_model
```

For this new program's own checkpoints, use `--resume` instead. Its checkpoint JSON format is not the same as the old optimizer's JSON report format.

## Validation performed

```bash
python3 -m unittest -v test_partitions_slsqp.py
```

All 15 regression tests passed. They include analytic gradients, the known fixed-incidence optimum, lower-bound checks, weighted cells, five-cell coverage without a common vertex under a large deformation, actual union-volume checks, saved-reference round trips, label-invariant signatures, good/poor/time early exits, spherical and packing seed stages, directory locks, restart ID preservation, a two-process CLI run, and the one-cell zero-variable case.

Polishing the supplied 43-point file reproduces maximum diameter approximately `0.969767761861449`. A four-worker run of 1,000 ordinary-Voronoi attempts, and another of 800 weighted attempts with scale 0.02, did not improve it. This is an empirical result, not an exclusion of better topologies.

A separate 60-attempt, two-worker test with `--no-initial` found a different 41-point cover with diameter approximately `0.969995198424`. It tested genuinely new reference models and record rotation, but is slightly worse than the supplied configuration after polishing.

The package's `example_result` contains the incumbent and compact restart state from the 1,000-start ordinary-Voronoi test. Since no random start improved its polished input, that directory has no previous-best record. `from_scratch_example` contains the best and previous-best records from the independent 60-start test plus a four-start resume test. `validation_summary.json` distinguishes the tests and their results.

## References for the numerical interfaces

- SciPy SLSQP: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
- SciPy HalfspaceIntersection: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
- SciPy linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

The algorithmic baseline is the user's attached `partitions3d.py`; the direct-clipping, hard-constraint, lower-bound, cache and parallel-search machinery here are additions, not claims about features of that file.
