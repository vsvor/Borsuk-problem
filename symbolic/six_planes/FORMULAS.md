# Exact rational six-plane construction

New reproducible calculation added during repository assembly. This is a rational approximation of the saved numerical scores, not a symbolic solution of the nonlinear minimization problem.

Let `t = sqrt(2)/2` and `P = {x: |x_i|+|x_j| <= t (i<j), x_i <= 1/2}`. All coordinates below are in the original Euclidean scale. Define `K_i = P intersect {x: (a_j-a_i) dot x <= 0 for j != i}`. The common score denominator cancels from these inequalities.

## Compact version: grid denominator 10^6

The score numerators (one row per cell, cells 1–4) are:

```text
0 0 0
719189 -931694 1113708
-653806 -67933 1494572
772707 720064 1160152
```

Six oriented plane normals, `n_ij dot x = 0` (cell i uses <=):

| Cells | Normal |
|---|---|
| 1,2 | `719189, -931694, 1113708` |
| 1,3 | `-653806, -67933, 1494572` |
| 1,4 | `772707, 720064, 1160152` |
| 2,3 | `-1372995, 863761, 380864` |
| 2,4 | `26759, 825879, 23222` |
| 3,4 | `1426513, 787997, -334420` |

The maximum squared diameter is exactly

```text
(678436141956363256939712946118793/716785433639513889617381096906896) + (0)*sqrt(2)
```

A proved decimal enclosure of the maximum diameter is

`0.972881404297868 <= D <= 0.972881404297869`.

The attaining pair uses the **new exact-export numbering**: cell 3, vertices [26, 34].

| Cell | Vertices | Proved diameter interval |
|---|---:|---|
| 1 | 15 | `0.972881347529311, 0.972881347529312` |
| 2 | 17 | `0.972881273325583, 0.972881273325584` |
| 3 | 15 | `0.972881404297868, 0.972881404297869` |
| 4 | 20 | `0.972881404038494, 0.972881404038495` |

Exact cell volumes, exact coordinates `r+s*sqrt(2)`, cell membership lists and all six two-dimensional interfaces are in [`coarse_certificate.json`](coarse_certificate.json).

An integer polynomial annihilating D, coefficients in descending degree, is:

```json
["716785433639513889617381096906896", "0", "-678436141956363256939712946118793"]
```

## High-accuracy version: grid denominator 10^12

The score numerators (one row per cell, cells 1–4) are:

```text
0 0 0
719189193313 -931694443313 1113708243569
-653805899011 -67933266183 1494572102340
772706813605 720063706289 1160151649023
```

Six oriented plane normals, `n_ij dot x = 0` (cell i uses <=):

| Cells | Normal |
|---|---|
| 1,2 | `719189193313, -931694443313, 1113708243569` |
| 1,3 | `-653805899011, -67933266183, 1494572102340` |
| 1,4 | `772706813605, 720063706289, 1160151649023` |
| 2,3 | `-1372995092324, 863761177130, 380863858771` |
| 2,4 | `26758810146, 825879074801, 23221702727` |
| 3,4 | `1426512712616, 787996972472, -334420453317` |

The maximum squared diameter is exactly

```text
(2579873147304380782228001043059414499695208037638325098972904563282259005/4187589481787010842927080652891223201981877379357756479813584253217292544) + (478119175303771859243760693940674265/2046360056731710848638574063057220112)*sqrt(2)
```

A proved decimal enclosure of the maximum diameter is

`0.972881319605256 <= D <= 0.972881319605257`.

The attaining pair uses the **new exact-export numbering**: cell 4, vertices [36, 32].

| Cell | Vertices | Proved diameter interval |
|---|---:|---|
| 1 | 15 | `0.972881319604887, 0.972881319604888` |
| 2 | 17 | `0.972881319605142, 0.972881319605143` |
| 3 | 15 | `0.972881319605091, 0.972881319605092` |
| 4 | 20 | `0.972881319605256, 0.972881319605257` |

Exact cell volumes, exact coordinates `r+s*sqrt(2)`, cell membership lists and all six two-dimensional interfaces are in [`fine_certificate.json`](fine_certificate.json).

An integer polynomial annihilating D, coefficients in descending degree, is:

```json
["17535905667973206016168158451056591051012037283029378590528163154006949862564334908217129948170784124372169246421569693991924745232635313677991936", "0", "-21606899311993153218362980286123980277818956025240068823487539391364443367437662097836402158849078526393207111836285197616011070720518628526717440", "0", "4741196749459112052957744544677012365537601951567042908709420020648631494833351169586537077137061123022474774099712994535613867096246564142425225"]
```

## What is proved

Every point maximizes at least one score. For distinct cells the pairwise inequalities have opposite signs, so their intersection lies in the corresponding interface plane and their interiors are disjoint. The program additionally enumerates every triple-plane intersection, checks every halfspace exactly, verifies that each cell is full-dimensional, checks all six interfaces, and proves the exact volume identity `sum volume(K_i) = 7/2 - 2*sqrt(2)`.

The field arithmetic stores `(r,s)` for `r+s*sqrt(2)`. A comparison with zero reduces to rational signs and a comparison of `r^2` with `2*s^2`. No numerical epsilon is used. Decimal displays are not used in any geometric decision. The proof is an independently executable arithmetic computation, not a proof-assistant-verified implementation.

Rationalizing the six normals independently would generally destroy their compatibility. This script rationalizes the four score vectors jointly after subtracting the first score.

## Reproduce

```bash
python scripts/six_plane_symbolic.py data/three_truncations/six_planes/model.json \
  --denominator 1000000 --bound 0.972883 \
  --output runs/symbolic/coarse.json
python scripts/six_plane_symbolic.py data/three_truncations/six_planes/model.json \
  --denominator 1000000000000 --bound 0.97288131961 \
  --output runs/symbolic/fine.json --export runs/symbolic/display.txt
```

Run from the repository root. Standard library only. The five-line display file is rounded and must **not** be used as an exact certificate. Original numerical and new symbolic vertex numberings differ.
