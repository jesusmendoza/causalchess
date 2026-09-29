# CausalChess — Previous-Move Pretraining (PMP)

Self-supervised representation learning for chess. We learn a 256-d position
embedding **without engine evaluations or best-move labels** by training a CNN
to solve a novel pretext task: **predict the previous move** that produced a
given position (a 1,928-class retrodiction task).

The resulting embedding linearly decodes strategic concepts (castling, turn,
material), supports interpretable latent arithmetic, and predicts player Elo
— all from frozen features. A game-held-out player-identification probe did
not establish a reliable style signal.

> This repository contains the code to reproduce the companion paper, currently
> under review. It is a code-only release; the manuscript is distributed through
> the publisher.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset streaming also requires `curl` and `pzstd` on `PATH`.

A `stockfish` binary on `PATH` is required only for engine evaluations and
tournaments (not for training or the core embedding analyses).

---

## Models

| Model | File | Val acc | Notes |
|-------|------|---------|-------|
| CausalChess-mem | `models/prev_move_cnn_v2.pt` | 40.4% | 25.3M pairs (with duplicates) |
| CausalChess-no-mem | `models/prev_move_cnn_v2_dedup.pt` | 35.0% | 22.56M distinct (position, previous-move) pairs |
| Autoencoder (baseline) | `models/ae_cnn.pt` | — | reconstruction |
| SimCLR-flip (baseline) | `models/simclr_cnn_final.pt` | — | contrastive (color-flip) |

Trained on Lichess standard rated games from June–August 2023 (both players ≥ 2400 Elo).
Out-of-distribution validation is sealed on December 2024 games.

---

## Reproduce

### 1. Build the dataset

```bash
# Stream Lichess (June-August 2023), filter both players >= 2400 Elo, extract (fen, prev_move) pairs
python3 download_lichess.py --months 2023-06 2023-07 2023-08 --min-elo 2400 \
    --output lichess_data/prev_move_2400.tsv

# Pack into the binary format used by the trainer
python3 prepare_prev_move_bin.py        # -> data/
python3 prepare_dedup_dataset.py        # -> deduplicated variant
```

### 2. Train

```bash
python3 prev_move_train.py --arch cnn   # CausalChess-mem / no-mem
python3 autoencoder_train.py            # AE baseline
python3 contrastive_train.py            # SimCLR-flip baseline
```

### 3. Bayes ceiling

```bash
python3 estimate_bayes_ceiling.py
python3 fit_kappa_empirical_bayes.py
```

### 4. Evaluation suite

The full paper evaluation lives in `evaluation/` (see `evaluation/TESTS.md`
for the catalogue, runtimes, and `--ckpt` flags).

```bash
bash evaluation/run_all.sh
python3 evaluation/build_comparison_table.py   # regenerates the results comparison table
python3 evaluation/build_strategic_table.py   # Table V
python3 evaluation/test_probes_mlp_vs_linear.py  # seeded MLP/Ridge probe
python3 evaluation/test_style_game_split.py   # game-held-out player test
```

---

## Repository layout

```
.                      training / data / Bayes-ceiling scripts (root)
prev_move_models.py    network architectures (MLP, CNN-ResNet, LSTM, Transformer, dual-head)
prev_move_train.py     PMP training entry point
evaluation/            evaluation suite (probes, game-held-out player test, arithmetic, Elo, trajectories)
```

Datasets, model checkpoints and logs are not version-controlled (see
`.gitignore`); regenerate them with the steps above.

---

<!-- Author information withheld for double-anonymous review. -->
