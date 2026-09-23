#!/usr/bin/env python3
"""Coverage-preserving membership search for optimize_partition.py.

Keep this file beside optimize_partition.py. Python >= 3.10; dependencies are
numpy, scipy and shapely. No additional optimizer or commercial solver is used.

API:
    result = check_membership_changes(model, z, max_rounds=3)
    trial = try_membership_change(model, z, proposed_groups)

The saved reference boundary triangulation is retained. A proposed membership
pattern is admissible only if every reference triangle has an owner hull and
the common point remains in all hulls. All target-face/edge incidences and target
vertices remain fixed. This is a sufficient coverage condition, not a necessary
one. A failed certificate does NOT imply that no geometric cover is possible.

Each admissible pattern is a convex coordinate problem solved by the existing
optimizer. Candidate quality is measured AFTER coordinate reoptimization.
Neither current coordinates nor an unconverged solver objective are used to
rule out a candidate. Numerical lower bounds provide optional pruning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Sequence

# Avoid expensive oversubscription for this small dense problem when run as a
# script. Importing the module does not change the caller's thread settings.
if __name__ == "__main__":
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import optimize_partition as op


class MembershipError(op.InputError):
    """Membership pattern is unsupported by the retained coverage certificate."""


def _key(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(group)) for group in groups)


def _pairs(groups: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(sorted({tuple(sorted(p)) for group in groups
                              for p in itertools.combinations(group, 2)}), dtype=int)


def _memberships(groups: Sequence[Sequence[int]], n: int) -> list[set[int]]:
    owners = [set() for _ in range(n)]
    for k, group in enumerate(groups):
        for j in group:
            owners[j].add(k)
    return owners


def _copy_groups(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    return [sorted(int(j) for j in group) for group in groups]


def _groups_checked(model: op.Model, groups: Sequence[Sequence[int]]) -> list[list[int]]:
    n, k = len(model.reference), len(model.data.groups)
    if len(groups) != k:
        raise MembershipError(f"Expected {k} hulls.")
    clean = []
    for i, group in enumerate(groups):
        if (len(group) < 4 or any(not isinstance(j, (int, np.integer))
                                 or isinstance(j, (bool, np.bool_)) or not 0 <= j < n
                                 for j in group) or len(set(group)) != len(group)):
            raise MembershipError(f"Hull {i + 1} needs at least four distinct valid indices.")
        clean.append(sorted(int(j) for j in group))
    if set.union(*(set(g) for g in clean)) != set(range(n)):
        raise MembershipError("Every point must remain in at least one hull.")
    if any(model.common not in group for group in clean):
        raise MembershipError(f"Protect common point {model.common}: it must remain in every hull.")
    return clean


def model_with_memberships(model: op.Model,
                           groups: Sequence[Sequence[int]]) -> op.Model:
    """Replace co-hull constraints, retaining the original affine coordinate model.

    The original model must have been built/validated with op.build_model().
    Never change its reference geometry, target planes, or incidence lists here.
    Inputs are not modified. Hull numbering in Python lists is zero-based.
    """
    if not model.certificate or not model.certificate.get("valid_numerically"):
        raise MembershipError("A validated reference boundary mesh is required.")
    groups = _groups_checked(model, groups)
    owners = _memberships(groups, len(model.reference))
    triangles = []
    missing = []
    for index, triangle in enumerate(model.certificate["triangles"]):
        available = set.intersection(*(owners[j] for j in triangle["vertices"]))
        if not available:
            missing.append((index, triangle["vertices"]))
            continue
        previous = triangle["owner"] - 1
        owner = previous if previous in available else min(available)
        triangles.append({**triangle, "owner": owner + 1})
    if missing:
        raise MembershipError(
            f"Retained mesh has {len(missing)} ownerless triangle(s), e.g. {missing[:4]}. "
            "This move is uncertified by this mesh; it is not a proof of noncoverage.")
    pairs = _pairs(groups)
    a, b = pairs.T
    certificate = {**model.certificate, "triangles": triangles,
                   "common_point": model.common}
    data = op.Data(model.data.points.copy(), groups, model.data.stored)
    return replace(model, data=data, pairs=pairs, C=model.B[a] - model.B[b],
                   v=model.reference[a] - model.reference[b], certificate=certificate)


def membership_delta(before: Sequence[Sequence[int]],
                     after: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """Human-readable changes: hull numbers one-based, point indices zero-based."""
    return [{"hull": i + 1, "removed": sorted(set(a) - set(b)),
             "added": sorted(set(b) - set(a))}
            for i, (a, b) in enumerate(zip(before, after)) if set(a) != set(b)]


@dataclass
class Trial:
    status: str
    model: op.Model | None = None
    z: np.ndarray | None = None
    diameter: float | None = None
    report: dict[str, Any] | None = None


def try_membership_change(
    model: op.Model,
    z: np.ndarray,
    new_groups: Sequence[Sequence[int]],
    *,
    improvement_tol: float = 1e-8,
    maxiter: int = 1000,
    ftol: float = 1e-13,
    feasibility_tol: float = 1e-10,
    prune: bool = True,
    incumbent_diameter: float | None = None,
) -> Trial:
    """Certify, reoptimize, and test one proposed change without mutating inputs.

    Returns a feasible model/coordinate pair whenever reoptimization was done
    and its independent coverage check passed, even if it did not improve.
    Rejection statuses distinguish a certificate failure from an LP lower bound
    and from a numerical solver/coverage failure. Improvement means an absolute
    decrease greater than improvement_tol, not a rounded display change.

    'prune=False' disables both fixed-vertex and tangent-LP screening. It is useful
    for checking the numerical pruning independently. All reports and bounds are
    floating-point, not exact/interval-arithmetic certificates.
    """
    if (not np.isfinite(improvement_tol) or improvement_tol <= 0
            or maxiter < 1 or not np.isfinite(ftol) or ftol <= 0
            or not np.isfinite(feasibility_tol) or feasibility_tol <= 0):
        raise ValueError("Tolerances must be positive and finite; maxiter must be positive.")
    z = np.asarray(z, dtype=float).copy()
    if z.shape != (model.q,) or not np.all(np.isfinite(z)):
        raise ValueError(f"Expected {model.q} finite coordinates.")
    if model.point_violation(z) > feasibility_tol * model.scale:
        raise ValueError("Input coordinates violate the saved affine model/target.")
    baseline = (op.max_diameter(model.positions(z), model.data.groups)
                if incumbent_diameter is None else float(incumbent_diameter))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Incumbent diameter must be positive and finite.")
    report: dict[str, Any] = {"incumbent_diameter": baseline,
                              "improvement_tolerance": improvement_tol}
    try:
        candidate = model_with_memberships(model, new_groups)
    except MembershipError as exc:
        return Trial("certificate_rejected", report={**report, "reason": str(exc)})
    report["changes"] = membership_delta(model.data.groups, candidate.data.groups)
    ff = candidate.pairs[np.all(candidate.pairs < candidate.fixed, axis=1)]
    fixed_lower = (float(np.max(np.linalg.norm(candidate.reference[ff[:, 0]]
                                               - candidate.reference[ff[:, 1]], axis=1)))
                   if len(ff) else 0.0)
    # Guard for floating arithmetic, in addition to the user improvement threshold.
    fixed_lower = max(0.0, fixed_lower - 1e-12 * candidate.scale)
    report["fixed_vertex_lower_bound"] = fixed_lower
    cutoff = baseline - improvement_tol
    if prune and fixed_lower >= cutoff:
        return Trial("fixed_bound_pruned", report=report)
    if prune:
        bound = op.lower_bound(candidate, z, baseline)
        report["initial_lower_bound"] = bound
        if bound.get("success") and bound["diameter_lower_with_allowance"] >= cutoff:
            return Trial("tangent_bound_pruned", report=report)
    new_z, solver = op.solve_model(candidate, z, maxiter, ftol, feasibility_tol, 0)
    points = candidate.positions(new_z)
    diameter = op.max_diameter(points, candidate.data.groups)
    coverage = op.check_cover(candidate, points, max(1e-10, 5 * feasibility_tol))
    report.update(solver=solver, coverage=coverage, diameter=diameter,
                  improvement=baseline - diameter)
    if not coverage["valid_numerically"]:
        return Trial("coverage_check_failed", report=report)
    bound = op.lower_bound(candidate, new_z, diameter)
    report["final_lower_bound"] = bound
    if diameter < cutoff:
        status = "improved"
    elif bound.get("success") and bound["diameter_lower_with_allowance"] >= cutoff:
        status = "no_improvement_bound"
    else:
        status = "no_improvement_unresolved"
    return Trial(status, candidate, new_z, diameter, report)


@dataclass
class Proposal:
    groups: list[list[int]]
    move: dict[str, Any]


class Neighborhood:
    """Generate a finite, reproducible neighborhood on a fixed labeled mesh."""
    def __init__(self, model: op.Model, points: np.ndarray):
        self.model = model
        self.triangles = np.array([t["vertices"] for t in model.certificate["triangles"]], dtype=int)
        self.labels = np.array([t["owner"] - 1 for t in model.certificate["triangles"]], dtype=int)
        self.k = len(model.data.groups)
        used = set(int(j) for j in self.triangles.ravel())
        # Retain original memberships of points not used by the boundary mesh.
        self.anchors = [set(g) - used | {model.common} for g in model.data.groups]
        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for i, triangle in enumerate(self.triangles):
            for edge in itertools.combinations(triangle.tolist(), 2):
                edge_to_triangles.setdefault(tuple(sorted(edge)), []).append(i)
        self.adjacency = [set() for _ in self.triangles]
        for incidences in edge_to_triangles.values():
            for i, j in itertools.combinations(incidences, 2):
                self.adjacency[i].add(j)
                self.adjacency[j].add(i)
        self.points = points

    def groups_from_labels(self, labels: np.ndarray) -> list[list[int]]:
        groups = [set(s) for s in self.anchors]
        for triangle, owner in zip(self.triangles, labels):
            groups[int(owner)].update(int(j) for j in triangle)
        return [sorted(s) for s in groups]

    def patch_sets(self, max_size: int) -> Iterator[tuple[int, ...]]:
        """All same-owner, edge-connected patches of size at most max_size."""
        layer = {(i,) for i in range(len(self.triangles))}
        for size in range(1, max_size + 1):
            for patch in sorted(layer):
                yield patch
            if size == max_size:
                break
            next_layer = set()
            for patch in layer:
                neighbors = set.union(*(self.adjacency[i] for i in patch)) - set(patch)
                for j in neighbors:
                    if self.labels[j] == self.labels[patch[0]]:
                        next_layer.add(tuple(sorted((*patch, j))))
            layer = next_layer
            if not layer:
                break

    def proposals(self, *, max_patch_size: int, split_star_limit: int,
                  random_trials: int, random_moves: int,
                  rng: np.random.Generator) -> Iterator[Proposal]:
        groups = self.model.data.groups
        # Simultaneously remove every membership unnecessary for the chosen
        # triangle ownerships. This remains a valid cover even before optimization.
        yield Proposal(self.groups_from_labels(self.labels), {"type": "trim_redundant"})
        # All single deletions, including choices not made by canonical trimming.
        for source, group in enumerate(groups):
            for vertex in sorted(group):
                if vertex == self.model.common or len(group) <= 4:
                    continue
                changed = _copy_groups(groups)
                changed[source].remove(vertex)
                yield Proposal(changed, {"type": "delete", "point": vertex, "hull": source + 1})
        patches = set(self.patch_sets(max_patch_size))
        # Full source-owned vertex stars can be larger than max_patch_size.
        stars = []
        for vertex in sorted(set(int(j) for j in self.triangles.ravel())):
            for source in range(self.k):
                patch = tuple(int(i) for i in np.flatnonzero(
                    np.any(self.triangles == vertex, axis=1) & (self.labels == source)))
                if patch:
                    stars.append((vertex, source, patch))
                    patches.add(patch)
        ordered_patches = sorted(patches, key=lambda p: (len(p), p))
        for patch in ordered_patches:
            source = int(self.labels[patch[0]])
            for destination in range(self.k):
                if destination == source:
                    continue
                labels = self.labels.copy()
                labels[list(patch)] = destination
                yield Proposal(self.groups_from_labels(labels),
                               {"type": "patch", "triangles": list(patch),
                                "from_hull": source + 1, "to_hull": destination + 1})
        # Remove a vertex from a source hull, allowing its incident source-owned
        # triangles to go to DIFFERENT destination hulls. This is stronger than
        # transferring the whole star to one hull. No expensive candidate is
        # dropped using its current (movable) coordinates.
        for vertex, source, patch in stars:
            if not 2 <= len(patch) <= split_star_limit:
                continue
            destinations = [d for d in range(self.k) if d != source]
            for choices in itertools.product(destinations, repeat=len(patch)):
                if len(set(choices)) == 1:
                    continue  # Already generated as a whole-patch transfer.
                labels = self.labels.copy()
                labels[list(patch)] = choices
                yield Proposal(self.groups_from_labels(labels),
                               {"type": "split_star", "point": vertex,
                                "from_hull": source + 1, "triangles": list(patch),
                                "to_hulls": [int(d) + 1 for d in choices]})
        # Simultaneous mutations: intermediate changes need not be improving.
        # Only the final pattern is reoptimized and compared with the incumbent.
        for trial in range(random_trials):
            labels = self.labels.copy()
            changes = []
            count = int(rng.integers(2, random_moves + 1))
            for _ in range(count):
                patch = ordered_patches[int(rng.integers(len(ordered_patches)))]
                destination = int(rng.integers(self.k))
                labels[list(patch)] = destination
                changes.append({"triangles": list(patch), "to_hull": destination + 1})
            yield Proposal(self.groups_from_labels(labels),
                           {"type": "random_compound", "trial": trial, "changes": changes})


@dataclass
class SearchResult:
    model: op.Model
    z: np.ndarray
    report: dict[str, Any]

    @property
    def points(self) -> np.ndarray:
        return self.model.positions(self.z)

    @property
    def groups(self) -> list[list[int]]:
        return self.model.data.groups

    @property
    def diameter(self) -> float:
        return op.max_diameter(self.points, self.groups)

    @property
    def improved(self) -> bool:
        return self.report["membership_improvement"] > self.report["options"]["improvement_tol"]


def check_membership_changes(
    model: op.Model,
    z: np.ndarray,
    *,
    max_rounds: int = 3,
    max_patch_size: int = 3,
    split_star_limit: int = 4,
    random_trials: int = 0,
    random_moves: int = 3,
    seed: int = 0,
    max_candidates: int = 10000,
    time_limit: float | None = None,
    improvement_tol: float = 1e-8,
    maxiter: int = 1000,
    ftol: float = 1e-13,
    feasibility_tol: float = 1e-10,
    prune: bool = True,
    polish_start: bool = True,
    verbose: bool = True,
    checkpoint: Callable[[op.Model, np.ndarray, dict[str, Any]], None] | None = None,
) -> SearchResult:
    """Best-improvement local search, returning the best verified feasible cover.

    No memberships/coordinates are changed in the supplied model or z. The time
    limit is checked between candidates; it cannot interrupt an individual SLSQP
    or LP solve. max_candidates counts DISTINCT nonincumbent membership patterns
    (including screened ones) over the whole call, not just nonlinear solves.

    Status 'no_improvement_found' is local to this retained mesh and generated
    neighborhood. It is NOT a global lower bound over all membership patterns.
    'candidate_limit', 'time_limit', 'round_limit', or 'interrupted' explicitly
    mark a truncated search. Reports distinguish unresolved numerical cases.
    """
    if (max_rounds < 1 or max_patch_size < 1 or split_star_limit < 0
            or random_trials < 0 or random_moves < 2 or max_candidates < 1
            or maxiter < 1 or any(not np.isfinite(v) or v <= 0 for v in
                                 (improvement_tol, ftol, feasibility_tol))
            or (time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0))):
        raise ValueError("Invalid search limits or tolerances.")
    # Recheck all original memberships against the validated retained mesh.
    current = model_with_memberships(model, model.data.groups)
    current_z = np.asarray(z, dtype=float).copy()
    if current_z.shape != (model.q,) or not np.all(np.isfinite(current_z)):
        raise ValueError(f"Expected {model.q} finite coordinates.")
    if model.point_violation(current_z) > feasibility_tol * model.scale:
        raise ValueError("Starting coordinates are not feasible in the saved model.")
    initial_groups = _copy_groups(current.data.groups)
    initial_diameter = op.max_diameter(current.positions(current_z), initial_groups)
    start_time = time.monotonic()
    initial_solver = None
    if polish_start:
        current_z, initial_solver = op.solve_model(current, current_z, maxiter, ftol,
                                                   feasibility_tol, 0)
    initial_coverage = op.check_cover(current, current.positions(current_z),
                                      max(1e-10, 5 * feasibility_tol))
    if not initial_coverage["valid_numerically"]:
        raise MembershipError("Starting cover failed the independent face-union check.")
    baseline = op.max_diameter(current.positions(current_z), initial_groups)
    rng = np.random.default_rng(seed)
    total = Counter()
    records = []
    rounds = []
    accepted = []
    status = "round_limit"
    stop = False
    if verbose:
        print(f"Starting diameter after coordinate polish: {baseline:.15f}", flush=True)
    if checkpoint:
        checkpoint(current, current_z, {"phase": "baseline", "diameter": baseline})
    try:
        if initial_solver and initial_solver.get("status") == "interrupted":
            status, stop = "interrupted", True
        for round_no in range(1, max_rounds + 1):
            if stop:
                break
            points = current.positions(current_z)
            current_d = op.max_diameter(points, current.data.groups)
            current_pairs = set(map(tuple, current.pairs.tolist()))
            current_bound = op.lower_bound(current, current_z, current_d)
            # Superset pruning is justified only if the incumbent is already
            # optimal to the requested improvement resolution, numerically.
            dominance_ok = bool(prune and current_bound.get("success") and
                                current_bound["diameter_lower_with_allowance"]
                                >= current_d - improvement_tol)
            neighborhood = Neighborhood(current, points)
            best_model, best_z, best_d = current, current_z, current_d
            best_report = None
            counts = Counter()
            seen = {_key(current.data.groups)}
            completed = True
            for proposal in neighborhood.proposals(max_patch_size=max_patch_size,
                    split_star_limit=split_star_limit, random_trials=random_trials,
                    random_moves=random_moves, rng=rng):
                if time_limit is not None and time.monotonic() - start_time >= time_limit:
                    status, stop, completed = "time_limit", True, False
                    break
                key = _key(proposal.groups)
                if key in seen:
                    counts["duplicates_or_incumbent"] += 1
                    continue
                if total["candidates"] >= max_candidates:
                    status, stop, completed = "candidate_limit", True, False
                    break
                seen.add(key)
                counts["candidates"] += 1
                total["candidates"] += 1
                try:
                    candidate = model_with_memberships(current, proposal.groups)
                except MembershipError as exc:
                    trial = Trial("certificate_rejected", report={"reason": str(exc)})
                else:
                    if dominance_ok and current_pairs <= set(map(tuple, candidate.pairs.tolist())):
                        trial = Trial("constraint_superset_pruned", report={
                            "reason": "All incumbent co-hull pair constraints remain; extra ones cannot help.",
                            "incumbent_lower_bound": current_bound["diameter_lower_with_allowance"]})
                    else:
                        trial = try_membership_change(current, current_z, proposal.groups,
                            incumbent_diameter=best_d, improvement_tol=improvement_tol,
                            maxiter=maxiter, ftol=ftol, feasibility_tol=feasibility_tol,
                            prune=prune)
                counts[trial.status] += 1
                record = {"round": round_no, "number": int(total["candidates"]),
                          "status": trial.status, "move": proposal.move,
                          "changes": membership_delta(current.data.groups, proposal.groups),
                          **(trial.report or {})}
                records.append(record)
                if trial.status == "improved":
                    best_model, best_z, best_d = trial.model, trial.z, float(trial.diameter)
                    best_report = record
                    if verbose:
                        print(f"  candidate {total['candidates']}: {best_d:.15f} "
                              f"({proposal.move['type']})", flush=True)
                    if checkpoint:
                        checkpoint(best_model, best_z, record)
                if trial.report and trial.report.get("solver", {}).get("status") == "interrupted":
                    status, stop, completed = "interrupted", True, False
                    break
                if verbose and total["candidates"] % 500 == 0:
                    print(f"  checked {total['candidates']} patterns; best {best_d:.15f}", flush=True)
            for key_count, value in counts.items():
                if key_count != "candidates":
                    total[key_count] += value
            rounds.append({"round": round_no, "input_diameter": current_d,
                           "best_diameter": best_d, "completed": completed,
                           "counts": dict(counts)})
            if best_report is not None:
                current, current_z = best_model, best_z
                accepted.append(best_report)
            if verbose:
                print(f"Round {round_no}: {dict(counts)}; best {best_d:.15f}", flush=True)
            if stop:
                break
            if best_report is None:
                status = "no_improvement_found"
                break
    except KeyboardInterrupt:
        # Retain a new best already verified during a partially explored round.
        if "best_model" in locals():
            current, current_z = best_model, best_z
            if best_report is not None and best_report not in accepted:
                accepted.append(best_report)
        status = "interrupted"
    final_points = current.positions(current_z)
    final_d = op.max_diameter(final_points, current.data.groups)
    final_check = op.check_cover(current, final_points, max(1e-10, 5 * feasibility_tol))
    if not final_check["valid_numerically"]:
        raise MembershipError("Final independent coverage check failed.")
    report = {"format_version": 1, "status": status,
              "initial_diameter": initial_diameter,
              "baseline_diameter_after_polishing": baseline,
              "coordinate_only_improvement": initial_diameter - baseline,
              "optimized_max_diameter": final_d,
              "membership_improvement": baseline - final_d,
              "membership_changed": _key(initial_groups) != _key(current.data.groups),
              "changes": membership_delta(initial_groups, current.data.groups),
              "counts": dict(total), "rounds": rounds, "accepted_moves": accepted,
              "candidate_reports": records,
              "initial_solver": initial_solver, "initial_coverage_check": initial_coverage,
              "coverage_check": final_check,
              "optimized_hulls": op.diameters(final_points, current.data.groups),
              "optimality_bound": op.lower_bound(current, current_z, final_d),
              "elapsed_seconds": time.monotonic() - start_time,
              "options": dict(max_rounds=max_rounds, max_patch_size=max_patch_size,
                              split_star_limit=split_star_limit, random_trials=random_trials,
                              random_moves=random_moves, seed=seed, max_candidates=max_candidates,
                              time_limit=time_limit, improvement_tol=improvement_tol,
                              maxiter=maxiter, ftol=ftol, feasibility_tol=feasibility_tol,
                              prune=prune, polish_start=polish_start),
              "indexing": "Point and triangle indices zero-based; reported hull numbers one-based.",
              "scope": "Local search on a retained reference boundary mesh; target vertices, "
                       "common-point memberships, and original face/edge incidences are fixed. "
                       "A negative search result does not exclude other memberships or triangulations.",
              "arithmetic_note": "Coverage checks and lower-bound pruning use floating-point "
                                 "arithmetic, not interval/exact arithmetic. Current distances are "
                                 "never used to prune movable-point candidates.",
              "model": op.model_record(current)}
    return SearchResult(current, current_z, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, help="Original optimizer report, or a previous search report. Strongly recommended.")
    parser.add_argument("--fixed-count", type=int, default=23)
    parser.add_argument("--incidence-tol", type=float, default=1e-6)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--split-star-limit", type=int, default=4)
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--random-moves", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--time-limit", type=float, help="Seconds; checked between candidates, not a hard per-solver timeout.")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--feasibility-tol", type=float, default=1e-10)
    parser.add_argument("--improvement-tol", type=float, default=1e-8)
    parser.add_argument("--no-prune", action="store_true", help="Disable all lower-bound and dominance screening.")
    parser.add_argument("--no-polish", action="store_true", help="Skip initial coordinate-only solve; useful on already optimized input.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name(args.input.stem + "_memberships.txt")
    report_path = args.report or output.with_suffix(".json")
    used_paths = [args.input.resolve(), output.resolve(), report_path.resolve()]
    if len(set(used_paths)) != 3:
        parser.error("Input, output and report must be distinct. Input is never overwritten.")
    if args.model and args.model.resolve() in (output.resolve(), report_path.resolve()):
        parser.error("Do not overwrite the input model report.")
    try:
        data = op.read_data(args.input)
        saved = json.loads(args.model.read_text(encoding="utf-8"))["model"] if args.model else None
        model = op.build_model(data, args.fixed_count, args.incidence_tol, saved)
        z = model.coordinates(data.points)
        def checkpoint(m: op.Model, zz: np.ndarray, info: dict[str, Any]) -> None:
            points = m.positions(zz)
            diameter = op.max_diameter(points, m.data.groups)
            op.write_data(output, points, m.data.groups, diameter)
            op.atomic_text(report_path, json.dumps({"format_version": 1,
                "status": "checkpoint_search_in_progress", "optimized_max_diameter": diameter,
                "last_verified_candidate": info, "model": op.model_record(m)},
                indent=2, allow_nan=False) + "\n")
        result = check_membership_changes(model, z, max_rounds=args.rounds,
            max_patch_size=args.patch_size, split_star_limit=args.split_star_limit,
            random_trials=args.random_trials, random_moves=args.random_moves, seed=args.seed,
            max_candidates=args.max_candidates, time_limit=args.time_limit,
            improvement_tol=args.improvement_tol, maxiter=args.maxiter, ftol=args.ftol,
            feasibility_tol=args.feasibility_tol, prune=not args.no_prune,
            polish_start=not args.no_polish, verbose=not args.quiet, checkpoint=checkpoint)
        op.write_data(output, result.points, result.groups, result.diameter)
        op.atomic_text(report_path, json.dumps(result.report, indent=2, allow_nan=False) + "\n")
        print(f"Status: {result.report['status']}")
        print(f"Diameter: {result.report['baseline_diameter_after_polishing']:.15f} -> {result.diameter:.15f}")
        print(f"Memberships changed: {result.report['membership_changed']}")
        print(f"Counts: {result.report['counts']}")
        print(f"Saved: {output}\nReport: {report_path}")
        return 0
    except (op.InputError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
