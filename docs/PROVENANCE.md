# Provenance and assembly changes

The source is the collection of code, records and verification packages supplied
in this research conversation. `reports/import_provenance.json` records each
copied path, source artifact/ZIP extraction path and input SHA-256.

Primary coordinate files and model JSON files are preserved byte-for-byte.
Historical reports are under `reports/historical/`; large per-candidate search
logs and duplicated ZIP archives are not included. Original inputs known to
have diameter inconsistencies are retained under `data/original/` only, clearly
separated from results. In particular, the `0.965967...` line in the original
`r_dod_4_0966.txt` is not its actual maximum co-hull distance; do not cite it as a
verified result.

New during this assembly:

* `scripts/six_plane_symbolic.py`, its two score-grid constructions, exact field
  calculations, rational squared-diameter expressions and certificates.
* `scripts/delta_balance.py`, a thin alias for the original balance program.
* Portable batch-verification, result-audit and test-launching tools.
* Fresh exact padded and raw checks of all five primary final records.
* Repository-relative adaptations of the inherited test fixture locations.
* The result index, documentation, dependency records, integrity manifest and
  repository-level tests.

The old source programs were otherwise preserved. The original Adam program
is user-supplied historical material; it is not represented as newly authored
here. The five-cut triangle certificate belongs to the earlier comparison and
must not be confused with the newly written six-plane symbolic verifier.

No missing large search histories have been reconstructed or invented. The
fresh test report lists what was actually rerun. A research result called
“best” in this repository means best retained in the stated project/family, not
a survey claim about all publications or a proof of a global optimum.

Bibliographic metadata for the related article was checked against the arXiv
abstract record. No copy of the article or third-party library source is bundled.
