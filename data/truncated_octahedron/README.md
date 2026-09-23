# Four-part covering of the Grünbaum universal cover

Target normalization:

```text
|x|+|y|+|z| <= sqrt(3)/2,
x <= 1/2, y <= 1/2, z <= 1/2.
```

This is the regular octahedron with distance 1 between opposite faces, with the
three positive axial vertices truncated at distance 1/2 from the origin. It has
15 vertices, diameter sqrt(2), and volume `5/2-sqrt(3)`.

Results:

- Classical Grünbaum four-part value: `0.988732000585888`.
- Best numerical four-convex-hull cover found here: `0.987727022161782`.
- Exact verification after explicit `1e-12` facet enlargement: `< 0.987727022165961`.
- 6,877 multistart attempts were completed.
- Independent fixed-topology SLSQP returned the same value, with a numerical
  fixed-model interval about `[0.987727022156719, 0.987727022161782]`.
- 5,573 local membership candidates (deterministic plus randomized) found no improvement.

The raw decimal coordinates have tiny rounding gaps against the ideal analytic
target; `exact_check_raw.json` records that failure. `exact_check_padded.json`
proves coverage of a rational outer enclosure of the ideal target after the
explicit enlargement, and recomputes the enlarged-cell diameters exactly.

## Search

```bash
python grunbaum_octahedron_search.py --run --starts 5000 --workers 6 --output-dir run
```

## Exact verification

```bash
python verify_grunbaum_cover.py relaxed.txt --pad 1e-12 \
  --diameter-bound 0.98772702217 --full-volume --report check.json
```

No global optimality claim is made. The convex hulls may overlap; a disjoint
(non-convex) partition follows by assigning each point to the first hull that contains it.
