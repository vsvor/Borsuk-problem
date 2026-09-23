# Reproduction and file formats

Run the commands in the top-level README from the repository root. Search
programs reside together in `scripts/`, preserving their sibling imports.
Repository tools locate input fixtures relative to their own paths, not to a
machine-specific `/mnt/data` directory. Old absolute paths inside historical
JSON metadata describe the original run and are not dependencies.

## Five-line format

1. Number of cells.
2. Stored numerical maximum diameter.
3. Total number of coordinate triples.
4. A JSON list of cell membership lists (zero-based point indices).
5. A JSON list of coordinate triples.

The first `fixed_count` points are target vertices. The count is specific to the
file: 23 for the three-truncation and balanced-A records, 26 for balanced B,
29/26/26 for five-cut/four-adjacent/four-opposite respectively. Point ordering
need not agree across different records or symbolic reconstructions.

## Checkpoint families are distinct

* `six_planes/model.json`: eight plane parameters and four scores; load with
  `six_plane_search.py`, not `optimize_partition.py --model`.
* `six_planes_relaxed/model.json` and `best_cover/model.json`: local optimizer
  incidence reports. Pass the matching report on restarts.
* `best_cover/global_engine.json`: native global-search checkpoint for a
  separately exported solution at the same objective level. Do not mix its
  coordinates with the local checkpoint simply because the objectives agree.
* `balanced/*/continuation.json`: balancing-program checkpoints containing their
  original source records. Use `balance_truncations.py` or its `delta_balance.py`
  alias. The same-directory raw text file is called `partition.txt`; pass the
  JSON checkpoint explicitly rather than expecting automatic same-stem lookup.
* `balanced/seeds/*/best.json`: native starting records usable by the balance
  program. Both have matching `best.txt` files.

Changing target parameters requires continuation or rebuilding the model.
Inferring incidences anew at an optimized point can accidentally freeze an
additional face constraint. Keep the original reference model.

## Search nondeterminism

A random seed is recorded, but concurrent completion order, floating-point
libraries and early-pruning incumbent updates can affect the sequence of
accepted results. A repeated finite multistart search need not reproduce the
same final coordinates or history bit-for-bit. The saved data plus deterministic
verification are the reproducibility target. Use one worker for easier search
comparisons; no exact global-optimum guarantee is attached to a search count.

## Test commands

```bash
python3 tools/run_tests.py
python3 tools/run_tests.py --pattern test_six_plane_symbolic.py
python3 tools/check_results.py --checksums
python3 tools/verify_results.py --full --overlaps
```

Tests use temporary directories. Search scripts retain best/previous-best
checkpoints rather than full iteration logs. Test logs in `reports/repository`
are validation records, not archived search histories.

For the supplementary exact three-point calculation:

```bash
python3 supplementary/truncation_comparison/triangle_certificate.py
```

This uses SymPy. It writes its JSON beside the script, so a rerun may update that
checked-in output; retain or compare the original checksum as appropriate.
For the comparison, use a scratch copy to avoid overwriting archived reports:

```bash
mkdir -p runs
cp -r supplementary/truncation_comparison runs/comparison
python3 runs/comparison/compare_records.py --directory runs/comparison
```

## Tested environment

The actual assembly/test environment is recorded in
`reports/repository/environment.json`; `requirements-tested.txt` records its
scientific-package versions. `requirements.txt` intentionally permits a wider
range. Supporting Python 3.10+ syntax is not a statement that every package/
Python combination in that range was tested here.
