# Five-cut and four-adjacent record comparison

This compares the best records retained in the previously supplied
`intermediate_truncation_package.zip`. It does not compare later local runs or
claim an optimum over all four-part covers.

## Input records and indexing

- `five_best.txt` / `.json`: 29 target vertices, 50 total points;
  maximum diameter 0.9635512424576009.
- `four_adjacent_best.txt` / `.json`: 26 target vertices, 47 total points;
  maximum diameter 0.9635512424576053.

All point indices are zero-based, and cells are numbered 1 through 4.
The text and JSON point arrays and memberships were checked for equality.

## Reproduce

Install numpy and scipy, then run:

```
python compare_records.py
```

For the exact three-point lower/upper bound, additionally install sympy and run:

```
python triangle_certificate.py
```

The latter uses exact arithmetic in Q(sqrt(2)), not a floating-point optimizer.
It proves that the minimum maximum distance for the specified canonical
three-point edge problem is strictly between 0.96355124245749 and
0.96355124245750. It is not a global lower bound for arbitrary covers.

## Conclusions

The coordinate symmetry is T(x,y,z)=(-x,-z,y). Cell numbers map from the five-cut
record to the four-adjacent record as [3,4,2,1]. Under this symmetry:

    T(P_five) = P_four_adjacent intersect {x <= 1/2}.

There are 25 exactly matching fixed vertices and 21 corresponding movable
vertices. Their reference target-face incidences and all 189 distinct
movable-to-movable co-hull pair constraints agree under this mapping. The four
cut-off face corners (five-cut indices 0,1,2,3) are replaced by the untruncated
axial vertex (four-adjacent index 25). Other optimized coordinates are not
identical: the maximum difference over all matched points is about 0.123.

The diameter-determining core consists of four disjoint three-point triangles:

| Five-cut indices | Four-adjacent indices, corresponding order |
| --- | --- |
| 33,41,47 | 34,41,36 |
| 34,36,43 | 46,42,43 |
| 38,39,45 | 32,44,37 |
| 42,46,49 | 40,38,31 |

The maximum aligned coordinate discrepancy on these 12 vertices is
3.5719600658e-7. All twelve triangle-edge lengths are within 1.2e-13 of their
record's maximum diameter. All are convex diameter constraints because each
pair is contained in a common cell, not because an entire triangle necessarily
belongs to a single cell.

At tolerance 1e-8, the five-cut record has one additional tight pair (37,39),
and the four-adjacent record has two near-tight pairs (30,38), (30,43).
Their deficits from the respective record maximum are about 5.53e-14,
2.40304e-11 and 4.38924e-10. The four triangular bottlenecks already furnish the
lower bound without those additional pairs.

## Extra cap

The extra four-adjacent axial vertex e=(1/sqrt(2),0,0) can be added to the image
of five-cut cell 3, which is four-adjacent cell 2. All four base corners of the
removed cap already belong to that cell. The largest distance from e to its
old generators is 0.8444039831230539, attained at five-cut point 40. It is
strictly below the old diameter by 0.119147259334547.

Thus this cap extension gives a four-adjacent cover of exactly the same
computed maximum diameter as the saved five-cut cover. Conversely, clipping
the saved four-adjacent cover at x=1/2 and applying T inverse produces a
five-cut cover without changing its diameter-determining triangles.

Derived covers are supplied in the original five-line format:

- `four_adjacent_extended_from_five.txt`: 51 points, target vertices first 26.
- `five_clipped_from_four_adjacent.txt`: 50 points, target vertices first 29.

These files have their own remapped point indices, recorded in
`comparison_report.json`. The raw input membership lists in the report always
refer to the unmodified original records, not the derived files.

## Verification scope

`independent_boundary_checks.json` was produced using the earlier
`optimize_partition.py`: build a boundary mesh/common-point certificate and
check target-face coverage. All four covers (the two originals and two derived
covers) passed; maximum missing-face areas were below 3e-17. These numerical
geometric checks are not exact or interval certificates, and area residuals
are not Hausdorff distances. Hull interiors are allowed to overlap.

The exact rational/algebraic bottleneck certificate is separate from those
floating-point geometric checks.

For a diameter strictly below the three-point bottleneck value, each of the
four surviving triangles must lose at least one co-hull pair constraint or
one prescribed target-edge incidence. This is necessary, not sufficient;
coverage still has to be preserved.
