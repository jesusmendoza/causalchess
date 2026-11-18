Full-vocabulary next-move transfer
Data, embeddings, fitted coefficients and checkpoint binaries are excluded.
Use the root README for reproduction and local checkpoint preparation.
All 27 completed per-run JSON measurements and summary statistics are included.
verify_runs.py audits the original data, checkpoint and result hashes; it requires the data and selected weights, which are excluded.
The pure-float64 reference measurement JSON is included as a numerical diagnostic, not as an extra experiment replicate.
The supplied run_checks.json records the audit performed with the original local artifacts.
No sealed validation was used.

The reviewer archive keeps the 27 original measurements in reported_results/ under the run directory. To recalculate the published tables from these measurements, run python3 paper/revision_tog_2026/r2_nextmove_full_20261116/reported_results/summarize.py. A new training run writes to a separate set of output paths. The five protocol tests create their own temporary inputs and can be run before preparing data or model weights.
