# What the checks prove

## Three levels of evidence

1. `partition.txt` contains literal decimal coordinates and memberships. Its
   stored diameter is a numerical record, not a coverage proof.
2. `certificates/current/*_padded.json` proves coverage by explicitly enlarged
   rational polytopes. These are different from the raw hulls.
3. `symbolic/six_planes/*_certificate.json` defines exact score-derived cells of
   the ideal irrational target. This is an exact convex partition with shared
   boundary faces and no interior overlaps. No padding is used in that model.

Never substitute a rounded coordinate export for either exact definition.

## Independent final-geometry procedure

`exact_cover_check.py` reads decimals directly as rational numbers. Supporting
planes are found by exhaustive triples of generators, with exact side tests.
Intersections use integer determinants. No input reference topology, NumPy,
Qhull, Shapely, SLSQP termination flag, or numerical feasibility tolerance is used.

For a four-cell cover the full test computes

    V(P) - sum(nonempty I subset {1,2,3,4})
           (-1)^(|I|+1) V(P intersect intersection(i in I) E_i).

All 15 intersections are convex. Their volumes and alternating sum are exact.
A finite union of closed hulls is closed. An uncovered point of a full-dimensional
compact convex target has a missing relative neighborhood of positive volume.
Thus exact zero missing volume proves coverage of the entire target, including
its boundary. This argument does not apply to an approximate numerical zero.

The independent faster test subtracts the four final hulls from every target
face and checks that one actual point belongs to all cells and the target.
Convexity then fills every radial segment to the boundary. The common-point
assumption is checked, not imported from the initial diagram. The general full
volume route needs no common point.

## Analytic target enclosure

The stored target vertices are approximate. The batch verifier instead builds
halfspaces using a rational upper enclosure of `1/sqrt(2)` obtained by integer
square roots. Its excess is less than `10^-30`. Axial offsets, including the
saved delta, are exact rationals. Coverage of this outer body implies coverage
of the ideal body. An inner enclosure supplies exact uncovered witnesses for
the raw decimal files. An outer failure alone is not a counterexample to ideal
coverage; the script reports that distinction.

## Explicit padding

With e = 10^-12, replace every supporting halfspace `n·x <= h` by
`n·x <= h + e*||n||_1`. This is invariant under positive rescaling of a plane.
There is no assumption of a universal diameter-increase formula: the new
vertices and their pairwise distances are recomputed exactly.

For all five primary records, both exact coverage methods pass. The full
inclusion–exclusion checks have exact zero missing volume. All raw files instead
have an uncovered witness in the ideal target. Original files remain unchanged.

The padded bodies extend slightly outside the target and may overlap. Intersect
them with the target to obtain contained convex covering sets. Pairwise overlap
checks distinguish a cover from an interior-disjoint convex partition. A cover
can be made into a disjoint partition of arbitrary subsets by first-owner
assignment, with no diameter increase.

## Certificate contents

Each current report records the exact integer halfspaces of the target and all
checked cells, input/target SHA-256 values, padding, exact squared diameters,
upward-rounded decimal bounds, volume zero tests, raw counterexample points
when appropriate, and overlap witnesses. Homogeneous integer coordinates and
rational strings are authoritative; approximate displays are only displays.

`tools/check_results.py --checksums` checks provenance and basic numeric data.
It does **not** repeat coverage arithmetic. Run `tools/verify_results.py --full`
to repeat that arithmetic. The code and tests are inspectable but are not a
proof-assistant formalization.

The complete original algorithm guide is [README_exact_cover_check.md](guides/README_exact_cover_check.md).
