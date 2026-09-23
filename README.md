*What is the largest constant $`d_{3,4}`$ such that any set of unit diameter in three-dimensional Euclidean space can be covered by four sets of diameter at most $`d_{3,4}`$?*

This repository presents numerical results for this problem. The main method consists in partitioning a universal cover or the polyhedra forming a universal covering system. As the universal cover, the cover found by V.V. Makeev is used, i.e., a rhombic dodecahedron truncated by three planes with inradius $`1/2`$.

In the paper by Alexander Tolmachev, Dmitry Protasov and Vsevolod Voronov,
*Coverings of planar and three-dimensional sets with subsets of smaller diameter*,
Discrete Applied Mathematics 320 (2022), 270–281,
[arXiv:2210.12394](https://arxiv.org/abs/2210.12394),
[DOI:10.1016/j.dam.2022.06.016](https://doi.org/10.1016/j.dam.2022.06.016) in the numerical results on 3D partitions some errors were made: partitioning the truncated rhombic dodecahedron using optimization of random Voronoi cells actually provides only the bound $`d_{3,4} \leq 0.9697\ldots`$. However, it is possible to prove even better estimate with the same algorithm. To obtain the bound $`d_{3,4} \leq 0.965399\ldots`$, one needs to consider partitions of two polyhedra (A and B, see below), at least one of which certainly covers any set of unit diameter. Moreover, a partition by six planes can be performed somewhat better than stated in the paper, $`d_{3,4} \leq 0.9728\ldots`$.

New versions of the programs, additional optimization algorithms, optimized partitions, and this repository were prepared with GPT-6 Astra. The original version is provided in the legacy folder.

# Convex coverings of truncated rhombic dodecahedra

Reproducible Python searches, saved configurations, and independent verification
for four-part coverings of axially truncated rhombic dodecahedra, with a comparison of the five Platonic solids. This repository
assembles the computations from this project; it is not an assertion that the
saved records are globally optimal or the best results in the literature.

**The numerical data and the exact certificates are deliberately separate.**
Literal decimal-coordinate hulls can have tiny coverage gaps. The certified
cover bounds below refer to explicitly enlarged hulls, not to those raw hulls.
A separate rational six-plane construction is an exact convex partition of the
ideal target and requires no enlargement.

## Results

Distances are ordinary Euclidean distances in the original scale. Every row
uses four cells. Point counts include fixed target vertices and shared points.
The 3-truncation rows have 23 fixed target vertices.

| Construction | Points | Saved numerical maximum diameter | Proved upper bound for the explicitly enlarged cover |
|---|---:|---:|---:|
| [3-truncated: six-plane construction](data/three_truncations/six_planes/) | 42 | 0.9728813196052167 | 0.972881319609817 |
| [3-truncated: locally relaxed six-plane model](data/three_truncations/six_planes_relaxed/) | 42 | 0.9699951984240177 | 0.969995198428698 |
| [3-truncated: best retained unrestricted cover](data/three_truncations/best_cover/) | 43 | 0.9697677618614492 | 0.969767761865705 |
| [Cover A](data/balanced/dummy/) | 43 | 0.9653991786498369 | 0.965399178654095 |
| [Cover B](data/balanced/upper/) | 47 | 0.9653991786503513 | 0.965399178654163 |
| [Symmetric 6-TRD](data/symmetric/six_trd/) | 53 | 0.9635512424574932 | 0.963551242461420 |

The last column is obtained by replacing each exact raw hull inequality
$`n·x \leq h`$ with $`n·x <= h + 10^{-12} \|n\|_1`$, enumerating the resulting vertices,
and checking their actual diameters. These are changed sets, not an acceptance
tolerance. All registered padded-cover certificates have **exactly zero missing volume** in
full three-dimensional inclusion–exclusion. See [certificates/current](certificates/current/)
and the [machine-readable result registry](results.json).

The four enlarged hulls may overlap. Intersecting each with the ideal target
keeps them convex and does not increase a diameter. Assigning each point to
the first covering hull gives a disjoint partition without increasing diameter,
but those assigned pieces need not be convex.

The bound for the rhombic dodecahedron truncated by six planes (6-TRD) actually means the best result that can be obtained by this method, by cutting something off from the rhombic dodecahedron with planes perpendicular to the axes. At least one of the covers will necessarily contain 6-TRD.

We also note that the optimality of the presented partitions (if they are optimal) requires a separate study. It is only asserted that these partitions can be found using the presented programs in a few minutes (or a few hours in the original version of the code), and that with a number of restarts of $`\sim 10^6`$ we had no further improvements. Note that GPT-6 quickly proved that 0.9697... is close to the optimum for 3-TRD, namely, the lower bound is 0.96959... See [3-TRD lower bound](trd3_lower_bound/)


### Exact rational/symbolic six-plane versions

The four score vectors are rationalized jointly, retaining six compatible
pairwise interface planes. Vertices, squared distances and volumes are computed
symbolically in `Q(sqrt(2))`, with exact comparisons.

| Score grid | Proved diameter interval | Status |
|---|---|---|
| denominator `10^6` | [0.972881404297868, 0.972881404297869] | Exact four-cell convex partition of the ideal target |
| denominator `10^12` | [0.972881319605256, 0.972881319605257] | Exact four-cell convex partition of the ideal target |

These are **new rational approximations of the saved numerical construction**,
not exact symbolic minimizers of the six-plane optimization problem. The compact
version even has a rational squared diameter. The [formulas, integer plane
normals and exact diameter expressions](symbolic/six_planes/FORMULAS.md) and
[executable symbolic calculation](scripts/six_plane_symbolic.py) are included.
Decimal display exports have their own point numbering and are not certificates.

### Symmetric comparison (four cells, inradius 1/2)

Every supporting facet plane is distance **1/2 from the origin**. This matches
6-TRD and gives minimum width 1 for the centrally symmetric solids. The regular
tetrahedron is the exception: its altitude is 2 and its minimum width is sqrt(3).
`--normalization min-width` rescales the tetrahedron by 1/sqrt(3); the other rows
are unchanged. Unit tetrahedral altitude would instead divide all lengths by 2.

| Solid | Vertices / faces | Best numerical diameter | Independently proved upper bound | Global optimum proved here? |
|---|---:|---:|---|---|
| 6-TRD | 32 / 18 | 0.963551242457493 | 0.963551242461420 (padded cover) | No |
| Regular tetrahedron | 4 / 4 | 1.500000000000000 | **3/2**, exact construction | **Yes** |
| Cube | 8 / 6 | 1.224744871391589 | **sqrt(3/2)**, four rectangular boxes | No claim |
| Regular octahedron | 6 / 8 | 1.224744871391589 | **sqrt(3/2)**, exact construction | **Yes** |
| Regular dodecahedron | 20 / 12 | 1.044923649675676 | 1.044923649680310 (padded cover) | No |
| Regular icosahedron | 12 / 20 | 0.927050983124843 | **3(sqrt(5)-1)/4**, exact cover | No claim |

These use 2,000 starts per target, followed by tighter polishing. Numerical
searches need not preserve a solid's symmetries. All six numerical records also
have full exact 15-intersection volume checks with $`10^{-12}`$ facet padding. The
separate elementary/algebraic constructions require **no padding**.

The icosahedral result simplifies to **edge midpoints, face centroids, and the
origin** with a four-color assignment of the twelve original vertices. Its
certificate checks the complete final boundary subdivision in Q(sqrt(5)). It is
an exact upper bound, not a proved global minimum. It does not by itself make
the icosahedron a universal container for all unit-diameter sets.


See [the normalization, proofs, and formulas](docs/SYMMETRIC_COMPARISON.md),
[the numerical summary](reports/symmetric_comparison.json), and
[exact Platonic certificates](symbolic/platonic/).



### Balanced truncation parameter

The saved continuation balances two fixed-model diameter functions at

```text
delta = 0.008811423368752003
numerical root bracket = [0.008811423364095392, 0.008811423373408615]
```

The bracket concerns the lower envelopes over the supplied reference models.
It is a floating-point primal–dual result, not an interval-arithmetic proof of
global optimality. The exact coverage certificates instead use the displayed
`delta` as an **exact rational decimal**. Both certified covers have diameter
strictly below `0.96539917866`. See [the saved balancing result](data/balanced/result.json).

## Geometry and conventions

Let

```text
R = { (x,y,z): |x|+|y| <= 1/sqrt(2),
               |x|+|z| <= 1/sqrt(2),
               |y|+|z| <= 1/sqrt(2) }.
P3 = R intersect {x <= 1/2, y <= 1/2, z <= 1/2}.
```

There are no additional $`x,y,z >= -1/2`$ inequalities in P3.
Its volume is exactly $`7/2 - 2*sqrt(2)`$.

The balanced targets use a different orientation:

```text
A_delta = R intersect {x >= -1/2+delta, y >= -1/2, z >= -1/2}
B_delta = R intersect {-1/2 <= x <= 1/2+delta, y >= -1/2, z >= -1/2}.
```

In the parametric script, these are $`(a,b)=(0.5,delta)` and `(delta,0)`$.
For A, $`x<=1`$ is a redundant (dummy) inequality. The literal sign convention is
$`-1/2+b <= x <= 1/2+a`$. The two cases are not obtained merely by changing a file
label: their target vertices and incidence models differ.

The six planes are **pairwise interfaces**, not six complete cuts through every
cell. For four score vectors $`a_i`$, cell i is
P3 intersect $`{(a_j-a_i)·x <= 0: j != i}`$. Maximizing a score proves coverage;
opposite pairwise inequalities separate the interiors.

## Quick start

Use Python **3.10 or newer**, from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 tools/check_results.py --checksums
python3 tools/run_tests.py
```

The core numerical dependencies are NumPy, SciPy and Shapely. Exact coverage
verification and the six-plane symbolic calculation use only the standard
library. Optional plotting and the supplementary triangle calculation:

```bash
python3 -m pip install -r requirements-optional.txt
```

No PyTorch is needed by the active search programs. The user-supplied historical
Adam implementation is retained separately under `data/original/` and is not
imported or executed by the test suite.

## Verify the saved results independently

```bash
# Exact final-boundary subtraction plus a checked common point:
python3 tools/verify_results.py

# Additionally recompute all 15 convex intersections per cover and test overlaps:
python3 tools/verify_results.py --full --overlaps

# Check the raw decimal hulls, without enlargement (failures are expected):
python3 tools/verify_results.py --raw
```

Reports go to `runs/verification/`; checked-in certificates are not overwritten.
Use `--only three_best balanced_dummy balanced_upper` to choose a subset.
The checker constructs rational inner/outer enclosures of the irrational target;
a successful outer-cover check proves coverage of the ideal target. Each of the original five raw
records has an exact uncovered witness inside the inner target. New comparison
records are not assumed to have gaps; the independent raw check decides.

For a generic final partition file:

```bash
python3 scripts/exact_cover_check.py best.txt --fixed-count 23 \
  --pad 1e-12 --diameter-bound 0.970 --full-volume --overlaps
```

That command uses the literal decimal hull of the prefix as target. It is not
a substitute for the analytic-target checks in `tools/verify_results.py`.
See [verification details](docs/VERIFICATION.md).

## Reproduce the computations

All related numerical scripts are kept together in `scripts/` so their local
imports work without installation. Put new outputs under `runs/`.

### 6-TRD and Platonic comparison

```bash
# Generate and search the six target bodies, keeping only best/previous best:
python3 scripts/symmetric_polyhedra.py --run --starts 10000 --workers 6 \
    --seed 23 --output runs/symmetric

# Search only 6-TRD (omit --run to generate geometry without a search):
python3 scripts/symmetric_polyhedra.py --solids six_trd --run \
    --starts 10000 --workers 6 --output runs/six_trd

# Tight polishing and explicit tetrahedral-score candidates, into a new folder:
python3 tools/polish_symmetric_results.py --input runs/symmetric \
    --output runs/symmetric_polished

# Exact final-data checks of the new registered numerical covers:
python3 tools/verify_results.py --only six_trd tetrahedron cube octahedron \
    dodecahedron icosahedron --full --overlaps

# Reconstruct the four simple exact Platonic covers (standard library only):
python3 scripts/platonic_exact.py --output runs/platonic_exact --export-displays
```

To verify a newly searched cover against its ideal (not decimal-vertex) target:

```bash
python3 scripts/verify_symmetric_cover.py runs/symmetric/icosahedron/best.txt \
    --solid icosahedron --pad 1e-12 --diameter-bound 0.93 \
    --full-volume --report runs/icosahedron_check.json
```

The `*_display.txt` files exported by `platonic_exact.py` are for visualization;
the exact JSON constructions are authoritative.

`--normalization min-width` is available in the target generator; it only
changes the regular tetrahedron. Best/previous checkpoints from a different
normalization must not be reused in the same output directory.

### Six-plane multistart

```bash
python3 scripts/six_plane_search.py --starts 10000 --workers 6 \
  --seed 7 --output runs/six_planes
```

### Release the six-plane restriction

```bash
python3 scripts/optimize_partition.py \
 data/three_truncations/six_planes/partition.txt --fixed-count 23 \
  --output runs/relaxed/partition.txt --report runs/relaxed/model.json \
  --print-every 0
```

The first relaxation omits `--model`: a six-plane JSON is not a local-optimizer
incidence model. On subsequent local restarts, pass the matching local report.

### Membership search on the best retained cover

```bash
python3 scripts/search_memberships.py \
 data/three_truncations/best_cover/partition.txt \
  --model data/three_truncations/best_cover/model.json --fixed-count 23 \
  --patch-size 4 --split-star-limit 4 --max-candidates 20000 --rounds 3 \
  --output runs/memberships/partition.txt --report runs/memberships/model.json --quiet
```

### Unrestricted multistart

```bash
python3 scripts/partitions_slsqp.py \
 data/three_truncations/best_cover/partition.txt --fixed-count 23 \
  --starts 10000 --workers 6 --seed 17 --output-dir runs/global
```

For a fresh diagram-only search on the same target add `--no-initial`. The
SLSQP stage is convex within a fixed membership/incidence model; exploration
across models is heuristic. `fixed_pair_pruned` rejects a model whose fixed
co-hull pair already prevents improvement; it is not a solver failure.

### Parametric truncations and balance

```bash
python3 scripts/parametric_truncation_search.py --a 0.02 --b 0.01 \
  --case adjacent --starts 5000 --workers 6 --output-dir runs/parametric

python3 scripts/delta_balance.py \
  --dummy data/balanced/seeds/dummy_seed/best.json \
  --upper data/balanced/seeds/upper_seed/best.json \
  --lo 0 --hi 0.02 --delta-tol 1e-11 --output-dir runs/balance
```

`delta_balance.py` is a compatibility alias; the canonical implementation is
`balance_truncations.py`. The seeds, continued model checkpoints and final
coordinate files are all retained. Use a new directory when changing target
parameters. A checkpoint from another target is not automatically a valid cover.

### Rational approximation and symbolic calculation

```bash
python3 scripts/six_plane_symbolic.py \
 data/three_truncations/six_planes/model.json \
  --denominator 1000000000000 --bound 0.97288131961 \
  --output runs/symbolic/fine.json --export runs/symbolic/display.txt
```

### View the saved cells and their diameter pairs

```bash
python3 scripts/show_hull_diameters.py \
 data/three_truncations/best_cover/partition.txt \
  --matplotlib --target-vertices 23
```

For balanced B use 26 target vertices; for balanced A use 23.
Detailed original CLI guides are in [docs/guides](docs/guides/).

## Repository layout

```text
scripts/                  searches, continuation, viewers and verifiers
data/three_truncations/   the three requested 3-truncation records and models
data/balanced/            two balanced covers, seeds, models and root bracket
data/intermediate/        five-cut, four-adjacent and four-opposite records
data/symmetric/           6-TRD and five Platonic-solid records and targets
data/original/            historical inputs, including known inconsistent data
certificates/current/     freshly recomputed exact cover checks and raw witnesses
symbolic/six_planes/       rational scores, exact vertices, volumes and formulas
supplementary/             intermediate-target comparison and triangle certificate
tests/                    portable regression tests
tools/                    repository audits and batch verification
docs/                     verification, reproducibility and original CLI guides
reports/                  source provenance, historical reports and current tests
results.json              authoritative index of the five primary records
```



