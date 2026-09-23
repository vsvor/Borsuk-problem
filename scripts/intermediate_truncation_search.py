#!/usr/bin/env python3
"""Generate and optionally search the inequivalent 4/5-axis truncations of a rhombic dodecahedron.

The base rhombic dodecahedron is
    |x|+|y| <= 1/sqrt(2),
    |x|+|z| <= 1/sqrt(2),
    |y|+|z| <= 1/sqrt(2).
A truncation labelled +x means x <= 1/2; -x means -x <= 1/2, etc.

Up to signed coordinate permutations there are only three cases with 4 or 5
truncations:
  five          : five cuts, one untruncated axial vertex;
  four_opposite : four cuts, the two untruncated axial vertices are opposite;
  four_adjacent : four cuts, the two untruncated axial vertices are non-opposite.

By default the script writes target JSON files. With --run it invokes
partitions_slsqp.py once for each symmetry class and writes summary.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import ConvexHull

AXES = {"x": 0, "y": 1, "z": 2}
ALL_CUTS = ("+x", "-x", "+y", "-y", "+z", "-z")

# Representatives of the three signed-permutation symmetry classes.
CASES = {
    # Omit -z. Any single omitted cut is equivalent.
    "five": ("+x", "-x", "+y", "-y", "+z"),
    # Omit +z and -z: the two untruncated axial vertices are opposite.
    "four_opposite": ("+x", "-x", "+y", "-y"),
    # Omit +x and +y: the two untruncated axial vertices are non-opposite.
    "four_adjacent": ("-x", "-y", "+z", "-z"),
}


def halfspaces(cuts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return A,b for A x <= b."""
    rows, rhs = [], []
    q = 1.0 / math.sqrt(2.0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si, sj in itertools.product((-1.0, 1.0), repeat=2):
            a = np.zeros(3)
            a[i], a[j] = si, sj
            rows.append(a)
            rhs.append(q)
    for cut in cuts:
        sign = 1.0 if cut[0] == "+" else -1.0
        a = np.zeros(3)
        a[AXES[cut[1]]] = sign
        rows.append(a)
        rhs.append(0.5)
    return np.asarray(rows, float), np.asarray(rhs, float)


def vertices_from_halfspaces(A: np.ndarray, b: np.ndarray, tol: float = 2e-10) -> np.ndarray:
    """Enumerate all 3-plane intersections and keep extreme feasible vertices."""
    pts = []
    for idx in itertools.combinations(range(len(A)), 3):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-11:
            continue
        x = np.linalg.solve(M, b[list(idx)])
        if np.max(A @ x - b) <= tol:
            if not any(np.linalg.norm(x - y) <= 2e-9 for y in pts):
                pts.append(x)
    if len(pts) < 4:
        raise RuntimeError("Failed to construct a full-dimensional target.")
    V = np.asarray(pts)
    hull = ConvexHull(V)
    V = V[np.unique(hull.simplices)]
    # Deterministic lexicographic ordering.
    order = np.lexsort((V[:, 2], V[:, 1], V[:, 0]))
    return V[order]


def target_record(name: str) -> dict:
    cuts = CASES[name]
    A, b = halfspaces(cuts)
    V = vertices_from_halfspaces(A, b)
    hull = ConvexHull(V)
    omitted = [c for c in ALL_CUTS if c not in cuts]
    return {
        "case": name,
        "cuts": list(cuts),
        "untruncated_axial_vertices": omitted,
        "vertices": V.tolist(),
        "vertex_count": len(V),
        "volume": float(hull.volume),
        "target_diameter": float(max(np.linalg.norm(V[i] - V[j])
                                     for i in range(len(V)) for j in range(i))),
        "note": "All base rhombic-dodecahedron support planes and all selected truncation planes are at the original scale; truncation planes have distance 1/2 from the origin."
    }


def read_best(path: Path) -> dict | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    return {
        "diameter": obj.get("diameter"),
        "points": len(obj.get("points", [])),
        "run_id": obj.get("run_id"),
        "source": obj.get("source"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("intermediate_truncations"))
    ap.add_argument("--run", action="store_true", help="Run partitions_slsqp.py on all three targets.")
    ap.add_argument("--search-script", type=Path, default=Path(__file__).with_name("partitions_slsqp.py"))
    ap.add_argument("--starts", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--seconds-per-start", type=float)
    ap.add_argument("--target-diameter", type=float)
    ap.add_argument("--power-weight-scale", type=float, default=0.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"symmetry_classes": {}, "search": {}}
    for name in CASES:
        rec = target_record(name)
        summary["symmetry_classes"][name] = {k: rec[k] for k in
            ("cuts", "untruncated_axial_vertices", "vertex_count", "volume", "target_diameter")}
        target_path = args.output / f"target_{name}.json"
        target_path.write_text(json.dumps(rec, indent=2) + "\n")
        if not args.quiet:
            print(f"{name:14s}: vertices={rec['vertex_count']:2d}, volume={rec['volume']:.12f}, "
                  f"diam(P)={rec['target_diameter']:.12f}, omitted={rec['untruncated_axial_vertices']}")

        if args.run:
            outdir = args.output / f"search_{name}"
            cmd = [sys.executable, str(args.search_script), "--vertices", str(target_path),
                   "--parts", "4", "--starts", str(args.starts), "--workers", str(args.workers),
                   "--seed", str(args.seed), "--output-dir", str(outdir),
                   "--power-weight-scale", str(args.power_weight_scale)]
            if args.hours is not None:
                cmd += ["--hours", str(args.hours)]
            if args.seconds_per_start is not None:
                cmd += ["--seconds-per-start", str(args.seconds_per_start)]
            if args.target_diameter is not None:
                cmd += ["--target-diameter", str(args.target_diameter)]
            if args.quiet:
                cmd += ["--quiet"]
            if not args.quiet:
                print("  running:", " ".join(cmd))
            proc = subprocess.run(cmd)
            summary["search"][name] = {"returncode": proc.returncode,
                                        "best": read_best(outdir / "best.json")}
            if proc.returncode != 0:
                print(f"WARNING: search for {name} exited with code {proc.returncode}", file=sys.stderr)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.run:
        print("\nBest verified diameters:")
        for name in CASES:
            best = summary["search"].get(name, {}).get("best")
            d = None if best is None else best.get("diameter")
            print(f"  {name:14s}: {'none' if d is None else f'{d:.15f}'}")
    print(f"\nSaved targets and summary under: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
