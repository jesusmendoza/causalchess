# CausalChess: Previous-Move Pretraining

CausalChess learns a 256-dimensional chess-position embedding without engine evaluations or expert best-move labels. The pretraining task is a **1,928-way UCI move classification problem**: given a board position, predict the previous move that led to it.

The paper reports 40.4% top-1 validation accuracy on 25.3 million training pairs. After removing exact duplicate `(position, previous-move)` pairs, the dataset has 22.56 million distinct pairs and the model reaches 35.0%. This deduplication tests sensitivity to repeated pairs; it does not establish the absence of all memorization or generalization to new games.

The paper separates properties directly observable from the board, such as material and a castled-king-square signature, from game-state readouts such as turn and in-check. It also reports qualitative latent-arithmetic examples and Elo estimation from frozen embeddings. A game-held-out player-identification evaluation did not establish a reliable style signal. The paper does not claim an engine-strength gain or a model of human cognition.

This repository contains source code. **Datasets and trained checkpoint binaries are not included.** Evaluation commands that mention paths such as `models/prev_move_cnn_v2.pt` expect locally generated checkpoints, not files downloadable from this repository.

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Streaming the Lichess archives also requires `curl` and `pzstd` on `PATH`. A Stockfish binary is needed only for evaluations that use Stockfish.

## Prepare the previous-move dataset

The production corpus uses standard rated Lichess games from June–August 2023, requires both players to have at least 2400 Elo, and sets `--min-ply 6`. The first example from a game is recorded after its sixth half-move; later examples are retained through ply 200.

```bash
python3 download_lichess.py \
  --months 2023-06 2023-07 2023-08 \
  --min-elo 2400 \
  --min-ply 6 \
  --output lichess_data/prev_move_2400.tsv

python3 prepare_prev_move_bin.py \
  --tsv lichess_data/prev_move_2400.tsv \
  --move-col prev_move \
  --outdir data
```

The packer writes `chess_x.bin`, `chess_y.bin`, `chess_phase.bin`, `chess_vocab.txt`, and `chess_meta.txt`. The previous-move UCI vocabulary is shared by the deduplicated variant. `chess_phase.bin` uses material-based opening/middlegame/endgame labels for diagnostics; phase is not the pretraining target.

To create the deduplicated files:

```bash
python3 prepare_dedup_dataset.py
```

It writes `chess_x_dedup.bin`, `chess_y_dedup.bin`, `chess_phase_dedup.bin`, and `chess_meta_dedup.txt` under `data/`, retaining the shared `chess_vocab.txt`.

## Train the two PMP checkpoints

These commands create local model files. They do not download checkpoints.

```bash
python3 prev_move_train.py \
  --task prev_move --arch cnn --bin-dir data \
  --output models/prev_move_cnn_v2.pt

python3 prev_move_train.py \
  --task prev_move --arch cnn --bin-dir data --dedup \
  --output models/prev_move_cnn_v2_dedup.pt
```

## Evaluation code

The `evaluation/` directory contains probe, transfer, player-identification, arithmetic, Elo, trajectory, and figure scripts. See [`evaluation/TESTS.md`](evaluation/TESTS.md) for the current catalogue and its limitations. **Command-line options vary by script**: some evaluation scripts accept `--ckpt`, while figure builders and table builders may use fixed checkpoint paths. Check the individual script's usage before running it.

The player-style re-evaluation uses five group splits by game; its `MIN_PLY = 50` filter is separate from the training corpus's `--min-ply 6`. It reports balanced accuracy because held-out game splits do not preserve equal class sizes.

## Data and model availability

Game archives, derived datasets, logs, and model checkpoints are excluded from version control. Recreate data with the preparation commands above and train checkpoints locally before running analyses that require them. The repository README describes expected outputs; it does not imply the binaries are present.

<!-- Author information withheld for double-anonymous review. -->
