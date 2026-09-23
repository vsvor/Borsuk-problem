# Four parts defined by six central interface planes

A standalone NumPy/SciPy program, without PyTorch, Shapely, or the previous
partition optimizer. Python 3.10 or newer.

## Run

```bash
python3 -m pip install numpy scipy
python3 six_plane_search.py --starts 1000 --workers 4 --output six_search
```

The default target is the **positive-axial truncation** from the later
`r_dod_4_0966.txt` data, not the earlier symmetric truncation:

```
x <= 1/2, y <= 1/2, z <= 1/2,
|x|+|y| <= 1/sqrt(2), |x|+|z| <= 1/sqrt(2), |y|+|z| <= 1/sqrt(2).
```

It has 23 vertices, 36 edges, 15 faces, volume `3.5 - 2*sqrt(2)`;
all supporting planes are at distance 1/2 from the origin. The code constructs
these data itself; no input partition file is required. `--target symmetric`
also imposes x,y,z >= -1/2, producing the earlier 32-vertex target.

The default search uses independent random starts. Optional perturbations of
the current record add a small basin-hopping-style component:

```bash
python3 six_plane_search.py --starts 10000 --workers 6 --seed 137 \
    --mutation-probability 0.4 --mutation-scale 0.12 \
    --target-diameter 0.966 --output six_search_mutation
```

A time-limited search without an attempt-count limit:

```bash
python3 six_plane_search.py --starts 0 --hours 6 --workers 6 \
    --seconds-per-start 10 --output six_search_long
```

`--target-diameter D` stops scheduling new runs after a verified result <= D;
it also ends a local solve when that target is reached. Already-running
parallel jobs finish. `--seconds-per-start` limits the SLSQP phase
cooperatively: it cannot interrupt a numerical subroutine in progress.

An optional, potentially unsafe rejection heuristic:

```bash
python3 six_plane_search.py --starts 10000 --workers 6 \
    --reject-above 1.02 --reject-after 25 --output six_search_fast
```

This ends a local solve if its best actual diameter is still above 1.02 after
25 SLSQP iterations. It is **not a lower bound** on what that run could achieve.
It is disabled by default. For minimization, falling *below* the desired
threshold is success, not a reason to reject a partition.

## Why not six independent random planes?

Six generic complete planes through the origin make 32 full-dimensional
regions, not four. The six planes in this program support the six **pairwise
interfaces** of four cells. Only an interface wedge of each plane is used as a
cell boundary; the rest of that plane is not a global cut.

Six independently chosen pairwise normals usually leave gaps. For example,
three pairwise interfaces of a triple of cells must meet along a nonzero ray;
therefore their three normals must be linearly dependent. Independent random
normals satisfy such conditions with probability zero.

The construction instead chooses four affinely independent score vectors a_i
and defines

    K_i = P intersect { x : (a_j-a_i).x <= 0 for every j != i }.

Every x has a maximal score a_i.x, so these four sets cover P. Different
interiors are disjoint because they take opposite sides of their pairwise
plane. Affine independence gives four full-dimensional cones and all six
interfaces. The origin is common to the four cells.

This is a zero-intercept linear-score diagram, **not** an ordinary Voronoi
diagram of arbitrary sites: ordinary Voronoi bisectors need not pass through
the origin.

## Eight geometric parameters

Write Q for the four vertices of a regular tetrahedron centered at 0. Set

    a_i = B Q_i,  B = Rz(rz) Ry(ry) Rx(rx) U,
    U = [[exp(a), c, d], [0, exp(b), e], [0, 0, exp(-a-b)]].

The parameter vector is `(rx, ry, rz, a, b, c, d, e)`. Det(B)=1 identically,
so the four score vectors cannot become affinely dependent in exact arithmetic.
This removes the irrelevant common scale rather than constraining six
arbitrary normals with compatibility equations. Translation of every score
vector by the same vector is also irrelevant and has already been removed.

Rotations are unrestricted. The default local boxes are |a|,|b| <= 1.5 and
|c|,|d|,|e| <= 3. They avoid very thin, poorly conditioned cones. These are
**search restrictions**, adjustable with `--shape-bound` and `--shear-bound`,
not proven bounds on a global optimizer. A record reports whether it hits
these bounds. A large bound may worsen numerical conditioning. Euler angles
also have coordinate singularities, which multistart can mitigate but not
eliminate as an optimization concern.

Starts have a uniform random rotation and normally distributed shape
parameters (`--shape-spread`, default 0.35). Thus planes are randomized
jointly, with compatibility built in. Shape parameters are clipped to the
chosen boxes before optimization. With mutation probability zero, a fixed
seed defines the same sequence of starts independently of worker count.
With mutations enabled, arrival order of parallel results affects proposals.

## Local minimax problem

The optimizer moves the **planes themselves**. It does not move their clipped
vertices independently or permit a boundary point to drift off an interface.
At every trial theta, it recomputes all four clipped polyhedra. Their vertex
types are enumerated exhaustively:

1. original target vertices retained in the cell;
2. target-edge intersections with a cut plane;
3. cone extreme rays clipped to the target boundary;
4. the common origin.

It calculates all pairwise squared distances in each cell. The objective is
an epigraph variable s, with constraints that the largest squared distances
are <= s; the diameter is sqrt(s). By default the eight largest pairs per
cell are supplied to SLSQP (`--top-pairs`). Including the largest pair makes
this equivalent in feasible set to including every pair. Several near-active
pairs give SLSQP more information at a minimax tie.

The code supplies analytic branch derivatives. For a retained plane
intersection M(theta)x(theta)=b, they are the implicit derivatives of that
linear system (implemented through simpler edge/ray formulas). In particular,

    d ||u-v||^2 / d theta = 2 (u-v).(du/dtheta-dv/dtheta).

Clipping topology and pair order are rebuilt each evaluation. There is no
frozen target-boundary triangulation. At clipping transitions or equal
diameters the functions are only piecewise smooth; the Jacobian chooses
current branches. **This problem is nonconvex. SLSQP success is neither a
local-minimum certificate at nonsmooth points nor a global-minimum proof.**
In particular, a diameter determined by two unchanged target vertices can
have a flat region; another start may be needed to cross that region.

The previous convex coordinate optimizer's tangent-LP lower bounds do not
apply to this plane-parameter problem. They are deliberately not used here.
Every parameter vector still defines a geometrically valid partition, so the
best geometry evaluated in a line search can be kept even when its epigraph
variable is infeasible or SLSQP terminates unsuccessfully. Before saving, the
actual diameter and topology are independently reconstructed and checked.

## Independent topology checking

```python
import numpy as np
from six_plane_search import make_target, check_topology

P = make_target('positive')
normals = np.array(...)  # shape (6,3)
report = check_topology(normals, P)
print(report)
```

Normal order is `01, 02, 03, 12, 13, 23`, using **zero-based cell labels**.
For a pair i<j, n_ij.x <= 0 is the side of cell i, and >= 0 the side of cell j.
Positive rescaling of a row has no effect; changing its sign swaps its sides.
The JSON records also store the order explicitly.

The checker accepts arbitrary supplied normals, including incompatible ones.
It uses a separate HalfspaceIntersection reconstruction, checks positive
volume for all four cells, computes the four convex-hull diameters, verifies
that the sum of volumes equals the target volume, and checks that all six
shared interfaces have dimension two. Opposite halfspace assignments already
guarantee disjoint interiors, which is why the volume sum tests missing
coverage here (unlike a generic cover that may have overlaps).

By default, a relative volume discrepancy up to 2e-8 and a halfspace residual
up to 2e-8 are allowed. Cells below 1e-10 of the target volume and numerically
negligible interfaces are rejected. **These are floating-point checks**, not
exact arithmetic or outward-rounded interval certificates. In particular,
an extremely small gap in independently supplied normals could be below the
checking tolerance; compatible score-generated planes have exact structural
coverage before rounding.

```bash
python3 six_plane_search.py --check six_search/best.json
```

The same command accepts a JSON array of six oriented normals.

## Calling one solve from Python

```python
import numpy as np
from six_plane_search import make_target, random_start, optimize_once

P = make_target()
theta0 = random_start(np.random.default_rng(7))
result = optimize_once(theta0, P, maxiter=300, seconds=10.0)
print(result['valid'], result.get('diameter'), result.get('message'))
# result['parameters']: eight parameters
# result['plane_normals']: six unit normals (all offsets are exactly zero)
# result['verification']: independent volume / interface / diameter checks
```

## Checkpoints and the earlier viewer

Only records are printed, plus a final count summary. `--quiet` suppresses
those messages. Only these files are written by the search:

```
best.txt
best.json
previous_best.txt
previous_best.json
```

The previous-best files appear after a second record. The text format is the
same five-line format used previously: part count, actual maximum diameter,
point count, membership lists, coordinates. The target vertices come first;
their order is the program's deterministic sorted order, **not the original
input point numbering**. All saved membership indices refer to the saved
coordinates. The JSON stores the six plane equations, generating scores,
eight parameters, solver termination, and independent geometric checks.
The best and previous-best filenames are updated, not appended to a run log.

Each file is atomically replaced, but the text/JSON pair is not a filesystem
transaction. JSON is authoritative on restart. Do not launch two controllers
writing the same output directory. Parallel workers within one controller
never write directly.

Reuse a directory with its existing best and start a fresh random sequence:

```bash
python3 six_plane_search.py --resume --seed 298 \
    --starts 10000 --workers 6 --output six_search
```

This resumes from the **record**, not from a stored history of failed starts;
use a new seed to avoid repeating the same random proposals. Optional mutations
can start from the loaded record. Re-running without `--resume` refuses to
overwrite an existing best.json.

The previous viewer can read the new best.txt:

```bash
python3 show_hull_diameters.py six_search/best.txt \
    --matplotlib --target-vertices 23
```

`show_hull_diameters.py` is not a dependency of this search and is not included
in this package. Use `--target-vertices 32` for the symmetric target.

## Supplied example and validation

A 100-start test and a separate 1,000-start test on the default positive
truncation both found D approximately **0.972881319605**. The supplied example
has 42 points and cell diameters

    0.9728813196049223
    0.9728813196049976
    0.9728813196051979
    0.9728813196052167

The independent cell-volume sum differs from the target by -1.11e-16.
All six interfaces are two-dimensional. This is a feasible example, not a
proof of the optimum in the six-plane family. It is larger than the earlier
0.969767761861449 cover; that earlier model was less restricted.

Run the 12 regression tests:

```bash
python3 -m unittest -v test_six_plane_search.py
```

They check the two targets, analytic score/normal/diameter Jacobians at generic
points, random valid fans, rejection of independent incompatible normals,
agreement of vertex enumeration with HalfspaceIntersection, plane rescaling,
local optimization, compatible export, record rotation, and early stopping.
The package's validation JSON records the numerical experiments. No detailed
individual run histories are required by the program.
