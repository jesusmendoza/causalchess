# Evaluation code: current scope and limits

This catalogue describes the evaluations reported in the ToG revision. The repository contains source code, not the paper's datasets or trained model checkpoints. The checkpoint paths shown by scripts are expected local outputs; they are not files included in the repository.

Command-line options differ by script. Some evaluation programs accept `--ckpt`; other table or figure builders use fixed paths or different option names. Read the individual script's `--help` output and defaults before running it. There is no claim that every file in `evaluation/` accepts a common `--ckpt` option or that one runner regenerates every paper result.

## Data and target definitions

### Previous-move pretraining

The pretraining label is the **previous UCI move string**, not a previous-position tensor. The model predicts one of 1,928 observed move classes from the current board. The Lichess training pipeline filters June–August 2023 standard rated games at 2400 Elo for each player, begins recording examples at `--min-ply 6`, and stops after ply 200. See the repository `README.md` for the command.

The `PMP-no-mem` label means the training data was deduplicated by the exact `(position, previous-move)` pair. Distinct move labels from the same board remain distinct. The result tests sensitivity to exact pair repetition; it does not mean the model has no memorization or establish generalization to unseen games.

### Diagnostic labels

- **Game phase** is assigned from material for diagnostics; it is not the pretraining target.
- **Castled** is a board-configuration proxy: the king is on c1/g1 or c8/g8 and that side has no remaining castling rights. It does not mean that the input records castling rights or that the probe verifies a castling move in the game history.
- Other board-observable labels, such as piece count and material, are interpreted separately from game-state readouts such as turn and in-check.

## Model baselines and availability

- **PMP-mem**: trained on the corpus with repeated pairs; paper validation top-1 is 40.4%.
- **PMP-no-mem**: trained after exact pair deduplication; paper validation top-1 is 35.0% on 22.56 million distinct pairs.
- **Matched next-move CNN**: trained with the same six-block CNN trunk to predict the following move. Its vocabulary has 1,932 UCI classes and its validation top-1 is 36.2%. This is a trained baseline, distinct from the frozen-embedding linear transfer evaluation below.
- **Raw bitboards** and **untrained CNN**: controls used in the corresponding probes.
- **Autoencoder (AE)** and **SimCLR-flip**: reconstruction and contrastive baselines used in the paper.

Model checkpoints and datasets are excluded. Reproduction of an evaluation that loads a trained checkpoint requires obtaining the data and training the checkpoint locally. A filename in a script or example is not evidence that the corresponding binary is present.

## Probes and transfer

| Script | What it evaluates | Interpretation limit |
|---|---|---|
| `test_probes.py` | Nine single-split linear concepts for Table III and Figure 1, using final checkpoints and a shared seed-42 sample; optional `--json-output` preserves unrounded scores. | Uses float64 probe design matrices, LogisticRegression with max_iter=20000, and Ridge with SVD. This single split does not measure pretraining variance. |
| `test_probes_raw_baseline.py` | Raw bitboards, frozen PMP and matched next-move embeddings, and untrained controls on the concept probes; reports mean and standard deviation over probe seeds. | Position-level probe splits are not held out by game. Castled is the proxy defined above. |
| `test_probes_strategic.py` | Linear readouts for board/game-state labels and strategic-feature diagnostics, including material-derived phase. | Distinguish directly observable properties from readouts; scores do not establish human-like strategic reasoning. |
| `test_probes_mlp_vs_linear.py` | MLP compared with Ridge on the same embeddings and positions; probe seed fixed to 42. | This is a single seeded comparison, not a multi-seed estimate. |
| `test_nextmove.py` | Frozen-embedding linear transfer to next-move prediction using the top 500 move classes. | The vocabulary is selected before the row/position split. The splits are not by game, so the result does not establish transfer to unseen games. |
| `test_finetune.py` | End-to-end next-move fine-tuning under the paper's small-data protocol. | The train/validation split is row/position-level, not grouped by game. The top-500 task does not establish unseen-game generalization. |
| `test_elo_predictor.py` | Ridge prediction of a player's Elo from the mean embedding of 40 positions, with player-level split seeds. | This is a skill readout from player aggregates, not player-style identification or an engine-strength result. |
| `build_strategic_table.py` | Rebuilds Table V Ridge scores from local production checkpoints. | Requires the checkpoint files and corpus; it does not download them. |

The manuscript reports for Elo in Table VIII: PMP-mem `R² = 0.517 ± 0.009`, `MAE = 193 ± 2` Elo; AE `R² = 0.510 ± 0.003`, `MAE = 195 ± 2` Elo. These are estimates across the reported player-split seeds, not claims about chess-engine strength.

## Player-identification evaluation

- `test_style.py` and `test_style_endgame.py` are position-split diagnostics. The original split had 676 of 677 validation positions from games also represented in training; those scores do not support a style-generalization claim.
- `test_style_game_split.py` uses five fixed 80/20 group splits by game and balanced accuracy. On 3,384 balanced endgame positions from 240 games, the reported means are 28.74 ± 3.24% for PMP-mem and 28.91 ± 2.24% for an untrained CNN, with 25% chance level. The revision withdraws the earlier player-style interpretation. These splits reuse one finite game set and one trained checkpoint, so their spread is descriptive rather than independent replication.
- The style-analysis filter uses `MIN_PLY = 50`; it is separate from the pretraining corpus's `--min-ply 6`.

## Trajectory figures and diagnostics

`trajectory_games.py` is the shared source for the complete Kasparov–Topalov 1999 PGN used by the revised trajectory analyses. It validates 87 plies and 88 positions, including the initial board, final SAN `44.Qa7`, result `1-0`, and the final FEN. An illegal move or mismatch raises an error.

The distance panels in Figures 3 and 4 report consecutive distances after L2-normalizing each position embedding. These unit-sphere chord distances lie in `[0, 2]`; they control for differences in embedding scale, but they do not align model coordinates. Figure 3's PCA panels are fitted independently for PMP-mem and PMP-no-mem, so their axes are not directly comparable.

The trajectory metrics also retain distances in each model's raw embedding units. Raw values in the trajectory results are scale-dependent and must not be compared directly across training objectives. The four-model script fixes seed 42, runs inference on CPU, and records the PGN and checkpoint hashes plus Python, NumPy, and PyTorch versions in its JSON output. The complete Kasparov–Topalov trajectory is a single-game case study; it does not estimate global trajectory smoothness or performance across a population of games. The other named-game paths are additional illustrative examples.

## Running an analysis

Start with the data and training commands in the repository `README.md`. Then inspect the selected script's usage and supply the required local corpus, PGN, and checkpoint paths. Figure and table builders may use fixed filenames; check their arguments before running them. Re-running stochastic training or evaluations can produce different values from the paper, especially where the manuscript identifies a single checkpoint or a single-run analysis.

### Regenerate trajectory results and figures

After training the four local checkpoints named in the builders:

```bash
python3 evaluation/trajectory_games.py
python3 evaluation/make_trajectories_four_models.py
python3 evaluation/make_trajectories_comparison_figure.py
python3 evaluation/test_landscape.py --ckpt models/prev_move_cnn_v2.pt
```

The four-model builder writes the unrounded trajectory JSON and native-scale diagnostics under `paper/revision_tog_2026/`; it also creates Figure 3. The other two builders create Figures 4 and 2, respectively. The single-model `test_trajectories.py` writes a separate diagnostic figure and does not overwrite Figure 3.

### Regenerate the linear-probe heatmap

Run `test_probes.py` once for each checkpoint, retaining its JSON and log in the corresponding directory:

| Directory under `evaluation/results_revision_20260930/` | Local checkpoint |
|---|---|
| `pmp_mem` | `models/prev_move_cnn_v2.pt` |
| `pmp_nomem` | `models/prev_move_cnn_v2_dedup.pt` |
| `ae` | `models/ae_cnn.pt` |
| `simclr` | `models/simclr_cnn_final.pt` |

Example for PMP-mem:

```bash
mkdir -p evaluation/results_revision_20260930/pmp_mem
python3 evaluation/test_probes.py --ckpt models/prev_move_cnn_v2.pt \
  --json-output evaluation/results_revision_20260930/pmp_mem/probes.json \
  > evaluation/results_revision_20260930/pmp_mem/probes.log 2>&1
```

Repeat with the other three directory/checkpoint pairs, then run `python3 evaluation/make_probe_lift_heatmap.py`. The builder prefers unrounded JSON metrics. These result directories are generated locally and excluded from the repository.
