# Symmetric comparison data

Four-set covers, with inradius **1/2**. The six directories are `six_trd`,
`tetrahedron`, `cube`, `octahedron`, `dodecahedron`, and `icosahedron`.

Each directory contains:

* `target.json`: ideal geometry identifier, floating search vertices, rational
  outer-enclosure halfspaces, facet count, and normalization metadata.
* `partition.txt`: selected numerical record in the original five-line format.
* `model.json`: matching SLSQP reference model and final coordinates.
* `search_best.json` and `search_state.json`: the pre-polishing multistart record
  and compact counts for 2,000 completed starts. These are historical provenance;
  use the saved source target and appropriate search checkpoint to resume.
* `active_pairs.json`: recomputed co-hull pairs within 1e-8 of the maximum diameter.

For tetrahedron and octahedron an explicit regular-tetrahedral score candidate
was also checked and selected when smaller than the rounded solver output.
The numerical generation and the final exact coverage check are distinct.

All selected records have exact rational padded-cover certificates under
`certificates/current` from the repository root. Padding is 1e-12
in the facet L1 convention, not an ignored geometric tolerance.

Separate exact constructions without padding for tetrahedron, cube, octahedron,
and icosahedron are in `symbolic/platonic` from the root. The latter's JSON is
authoritative; its `*_display.txt` exports are only for visualization.

See `docs/SYMMETRIC_COMPARISON.md` from the root for the proofs and the distinction
between a numerical upper bound for 6-TRD and the true method barrier beta4(P6).
