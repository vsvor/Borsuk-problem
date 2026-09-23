#!/usr/bin/env python3
"""Four-part SLSQP search with two parameterized axial truncation planes.

Base: |x|+|y|, |x|+|z|, |y|+|z| <= 1/sqrt(2).
Always retain -1/2+b <= x <= 1/2+a; require a+b >= 0.
  --case adjacent (default): also y >= -1/2, z >= -1/2.
  --case opposite:           also -1/2 <= y <= 1/2.

The parameters a,b are FIXED during a search; only the four covering hulls
are optimized. Physical lengths are not rescaled in the reported output.

Keep partitions_slsqp.py beside this file. All its search options, including
--starts, --workers, --resume and --target-diameter, are forwarded unchanged.
Geometry-changing/input options of that program cannot be forwarded.

Examples:
  python parametric_truncation_search.py --a 0.02 --b 0.01 --starts 5000 --workers 6
  python parametric_truncation_search.py --a 0.02 --b 0.01 --case opposite --starts 5000
  python parametric_truncation_search.py --a 0 --b 0 --generate-only

Python >= 3.10. numpy, scipy, shapely (shapely is only needed for the search).
The global search is heuristic; geometric checks are floating-point.
"""
from __future__ import annotations

import os
# Avoid importing NumPy with an oversubscribed BLAS before the backend starts.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = os.environ.get("PARTITION_BLAS_THREADS", "1")

import argparse
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

T = 1.0 / math.sqrt(2.0)
GEOMETRY_TOL = 2e-10
MIN_RADIUS = 2e-8  # Avoid near-degenerate geometries at the backend's tolerances.
FORMAT_VERSION = 1


class ParameterError(ValueError):
    """Invalid parameters, degenerate geometry, or inconsistent checkpoint."""


def _decimal(value: str | float | Decimal, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ParameterError(f"{name} must be a finite real number.") from exc
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ParameterError(f"{name} must be finite and representable in double precision.")
    return result


def _canonical(value: Decimal) -> str:
    return str(value.normalize()) if value else "0"


def make_halfspaces(a: str | float | Decimal, b: str | float | Decimal,
                    case: str = "adjacent") -> tuple[np.ndarray, np.ndarray, dict]:
    """Return unit-normal inequalities A@x <= rhs and parameter metadata.

    In particular the second x inequality is -x <= 1/2-b, NOT 1/2+b.
    Decimal arithmetic enforces a+b>=0 on the specified decimal inputs.
    """
    da, db = _decimal(a, "a"), _decimal(b, "b")
    if case not in ("adjacent", "opposite"):
        raise ParameterError("case must be 'adjacent' or 'opposite'.")
    with localcontext() as ctx:
        ctx.prec = max(40, len(da.as_tuple().digits) + len(db.as_tuple().digits) + 10)
        if da + db < 0:
            raise ParameterError("The requested restriction a+b >= 0 is violated.")
        width = Decimal(1) + da - db
        if width <= 0:
            raise ParameterError("The x slab is empty or flat: need 1+a-b > 0.")
        lower = float(-Decimal('0.5') + db)
        upper = float(Decimal('0.5') + da)
    clipped_lower, clipped_upper = max(lower, -T), min(upper, T)
    if clipped_lower >= clipped_upper:
        raise ParameterError("The x slab does not meet the interior of the rhombic dodecahedron.")
    if clipped_upper - clipped_lower < 4 * MIN_RADIUS:
        raise ParameterError("The retained slab is too thin for reliable double-precision search.")

    rows: list[list[float]] = []
    rhs: list[float] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si, sj in itertools.product((-1.0, 1.0), repeat=2):
            row = [0.0, 0.0, 0.0]
            row[i], row[j] = si / math.sqrt(2.0), sj / math.sqrt(2.0)
            rows.append(row)
            rhs.append(0.5)
    rows += [[1., 0., 0.], [-1., 0., 0.]]
    rhs += [upper, -lower]
    cut_names = ["x_upper", "x_lower"]
    if case == "adjacent":
        rows += [[0., -1., 0.], [0., 0., -1.]]
        rhs += [0.5, 0.5]
        cut_names += ["y_lower", "z_lower"]
    else:
        rows += [[0., 1., 0.], [0., -1., 0.]]
        rhs += [0.5, 0.5]
        cut_names += ["y_upper", "y_lower"]
    info = dict(a=float(da), b=float(db), a_decimal=_canonical(da), b_decimal=_canonical(db),
                case=case, x_lower=lower, x_upper=upper,
                x_slab_width=float(width),
                x_slab_center=0.5 * lower + 0.5 * upper,
                retained_x_range=[clipped_lower, clipped_upper], cut_names=cut_names)
    return np.asarray(rows), np.asarray(rhs), info


def _vertices(A: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Enumerate all nonsingular triple intersections; no assumed vertex count."""
    # An optional axis cut with rhs>T is already redundant on the base body.
    # Exclude it computationally to avoid huge constants in LPs/intersections.
    keep = np.ones(len(rhs), dtype=bool)
    keep[12:] = rhs[12:] <= T
    AA, bb = A[keep], rhs[keep]
    lp = linprog([0., 0., 0., -1.], A_ub=np.column_stack((AA, np.ones(len(AA)))),
                 b_ub=bb, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not lp.success or lp.x[3] <= MIN_RADIUS:
        raise ParameterError("Empty, lower-dimensional, or numerically thin target.")
    candidates = []
    for triple in itertools.combinations(range(len(AA)), 3):
        M = AA[list(triple)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, bb[list(triple)])
        if np.max(AA @ p - bb) > GEOMETRY_TOL:
            continue
        if not any(np.linalg.norm(p - q) < GEOMETRY_TOL for q in candidates):
            candidates.append(p)
    if len(candidates) < 4:
        raise ParameterError("Fewer than four distinct feasible target vertices.")
    V = np.asarray(candidates)
    hull = ConvexHull(V)
    V = V[hull.vertices]
    # Quantized keys stabilize equal-coordinate groups without rounding vertices.
    V = np.asarray(sorted(V, key=lambda v: tuple(np.round(v, 12)) + tuple(v)))
    if np.max(V @ A.T - rhs) > 2 * GEOMETRY_TOL:
        raise ParameterError("Constructed target failed its halfspace check.")
    return V, lp.x[:3], float(lp.x[3])


def _facet_record(V: np.ndarray, n: np.ndarray, c: float, name: str) -> dict:
    gaps = V @ n - c
    ids = np.flatnonzero(np.abs(gaps) <= 5 * GEOMETRY_TOL)
    dimension = -1
    area = 0.0
    if len(ids):
        singular = np.linalg.svd(V[ids] - V[ids][0], compute_uv=False)
        dimension = int(np.sum(singular > 5 * GEOMETRY_TOL))
        if dimension >= 2:
            # Axial normals: dropping their nonzero coordinate is an isometry.
            coords = np.delete(V[ids], int(np.argmax(abs(n))), axis=1)
            area = float(ConvexHull(coords).volume)
    status = "facet" if dimension >= 2 else "touching_only" if len(ids) else "redundant"
    return dict(name=name, normal=n.tolist(), rhs=c, status=status,
                intersection_dimension=dimension, vertex_indices=ids.tolist(), area=area,
                max_plane_residual=float(np.max(gaps)))


def make_target(a: str | float | Decimal, b: str | float | Decimal,
                case: str = "adjacent", require_four_cuts: bool = False) -> dict[str, Any]:
    """Build a target JSON record accepted by partitions_slsqp.py --vertices."""
    A, rhs, parameters = make_halfspaces(a, b, case)
    V, center, radius = _vertices(A, rhs)
    hull = ConvexHull(V)
    cuts = [_facet_record(V, A[12+i], float(rhs[12+i]), name)
            for i, name in enumerate(parameters.pop("cut_names"))]
    inactive = [c["name"] for c in cuts if c["status"] != "facet"]
    if require_four_cuts and inactive:
        raise ParameterError("Not all four requested cuts produce facets: " + ", ".join(inactive))
    distances = np.linalg.norm(V[:, None] - V[None, :], axis=2)
    planes = []
    for row in hull.equations:
        if not any(np.linalg.norm(row - other) < 1e-9 for other in planes):
            planes.append(row)
    key = {k: parameters[k] for k in ("case", "a_decimal", "b_decimal")}
    fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return dict(format_version=FORMAT_VERSION, family="parametric_four_truncation",
                parameters=parameters, geometry_key=key, geometry_sha256=fingerprint,
                vertices=V.tolist(), vertex_count=len(V), fixed_count=len(V),
                A=A.tolist(), rhs=rhs.tolist(), halfspace_convention="A @ point <= rhs",
                target_volume=float(hull.volume), target_diameter=float(np.max(distances)),
                facet_count=len(planes), requested_truncation_count=4,
                active_truncation_facets=4-len(inactive), truncation_planes=cuts,
                interior_point=center.tolist(), inscribed_ball_radius=radius,
                origin_in_target=bool(np.all(rhs >= -GEOMETRY_TOL)),
                max_vertex_halfspace_violation=float(max(0., np.max(V @ A.T - rhs))),
                warnings=(["These requested cuts are redundant or only touch the target: "
                           + ", ".join(inactive)] if inactive else []),
                coordinates="Original physical coordinates; no change of length scale.",
                note="a,b stay fixed. Convex-hull covers may overlap. Numerical, not interval-certified.")


def _write_target_guarded(path: Path, record: dict) -> None:
    """Never replace geometry metadata belonging to a different parameter pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("geometry_key") != record["geometry_key"]:
            raise ParameterError("Output directory belongs to different a,b or --case. Use a new directory.")
        old_v = np.asarray(previous.get("vertices", []))
        new_v = np.asarray(record["vertices"])
        if old_v.shape != new_v.shape or not np.array_equal(old_v, new_v):
            raise ParameterError("Target vertices differ from those already saved; use a new directory.")
        return
    if any((path.parent / name).exists() for name in ("best.json", "previous_best.json", "state.json")):
        raise ParameterError("Checkpoints exist without this wrapper's target.json. Use a new directory.")
    # Exclusive creation prevents silently overwriting another process's target.
    # A partial file after abrupt termination is an error, not a reason to replace it.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        _write_target_guarded(path, record)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Upper x plane is x=1/2+a.")
    p.add_argument("--b", required=True, help="Lower x plane is x=-1/2+b. Require a+b>=0.")
    p.add_argument("--case", choices=("adjacent", "opposite"), default="adjacent")
    p.add_argument("--output-dir", type=Path, help="Default: directory named by case and parameter values.")
    p.add_argument("--generate-only", action="store_true", help="Write target.json without starting optimization.")
    p.add_argument("--require-four-cuts", action="store_true", help="Reject unless all four cuts give 2-D target facets.")
    p.add_argument("--quiet", action="store_true", help="Suppress geometry/record summaries; errors and geometry warnings remain.")
    p.add_argument("--search-help", action="store_true", help="Print partitions_slsqp.py's full search options, then exit.")
    p.epilog = ("Search options are forwarded, for example: --starts 10000 --workers 6 "
                "--seed 17 --seconds-per-start 5 --target-diameter 0.96 --resume.\n"
                "Use --search-help without --a/--b to see all search options.")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Import the old solver only when it is needed; geometry generation does not
    # depend on Shapely or on the backend file.
    if argv == ["--search-help"]:
        import partitions_slsqp as engine
        return engine.main(["--help"])
    parser = _parser()
    args, search_args = parser.parse_known_args(argv)
    if args.search_help:
        import partitions_slsqp as engine
        return engine.main(["--help"])
    reserved = {"--vertices", "--input", "--fixed-count", "--parts", "--initial-model", "--no-initial"}
    if any(s.split("=", 1)[0] in reserved for s in search_args):
        parser.error("Do not supply --vertices, --input, --fixed-count, --parts, --initial-model, or --no-initial. "
                     "The wrapper constructs a target and searches exactly four hulls.")
    if args.generate_only and search_args:
        parser.error("Search options have no effect with --generate-only; remove them.")
    try:
        record = make_target(args.a, args.b, args.case, args.require_four_cuts)
        if args.output_dir is None:
            def slug(value: float) -> str:
                return f"{value:.10g}".replace("-", "m").replace("+", "").replace(".", "p")
            args.output_dir = Path(f"truncation_{args.case}_a{slug(float(args.a))}_b{slug(float(args.b))}_"
                                   + record["geometry_sha256"][:8])
        target_path = args.output_dir / "target.json"
        _write_target_guarded(target_path, record)
        for message in record["warnings"]:
            print("WARNING: " + message, file=sys.stderr, flush=True)
        if not args.quiet:
            params = record["parameters"]
            print(f"Target: {args.case}; {params['x_lower']:.12g} <= x <= {params['x_upper']:.12g}", flush=True)
            print("Other cuts: " + ("y >= -0.5; z >= -0.5" if args.case == "adjacent"
                                      else "-0.5 <= y <= 0.5"), flush=True)
            print(f"Vertices={record['vertex_count']}; facets={record['facet_count']}; "
                  f"active truncation facets={record['active_truncation_facets']}/4; "
                  f"volume={record['target_volume']:.12g}; diameter={record['target_diameter']:.12g}", flush=True)
            print(f"Target saved: {target_path}", flush=True)
        if args.generate_only:
            return 0
        try:
            import partitions_slsqp as engine
        except ModuleNotFoundError as exc:
            if exc.name == "partitions_slsqp":
                raise ParameterError("Keep partitions_slsqp.py in the same directory as this script.") from exc
            raise
        forwarded = ["--vertices", str(target_path), "--parts", "4",
                     "--output-dir", str(args.output_dir), *search_args]
        if args.quiet:
            forwarded.append("--quiet")
        return engine.main(forwarded)
    except (ValueError, OSError, QhullError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed checkpoints are retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
