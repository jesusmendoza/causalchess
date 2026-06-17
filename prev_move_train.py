#!/usr/bin/env python3
"""
prev_move_train.py — Train previous-move prediction with selectable architecture.

Supports: mlp, attention, lstm.

Usage:
    python3 prev_move_train.py --arch mlp --data lichess_data/prev_move_2400.tsv --epochs 20
    python3 prev_move_train.py --arch attention --data lichess_data/prev_move_2400.tsv --epochs 20
    python3 prev_move_train.py --arch lstm --data lichess_data/prev_move_2400.tsv --epochs 20
"""
import argparse
import csv
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prev_move_models import create_model, MODELS


def uci_to_squares(uci: str):
    """Convert UCI move (e.g. 'e2e4') to (from_sq_idx, to_sq_idx) 0-63."""
    files = 'abcdefgh'
    from_sq = files.index(uci[0]) + (int(uci[1]) - 1) * 8
    to_sq = files.index(uci[2]) + (int(uci[3]) - 1) * 8
    return from_sq, to_sq


def build_move_vocab(tsv_path, max_rows=0):
    moves = set()
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            moves.add(row['prev_move'])
            if max_rows > 0 and i >= max_rows:
                break
    return {m: i for i, m in enumerate(sorted(moves))}


def fen_to_bitboards(fen: str) -> np.ndarray:
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    bits = np.zeros(768, dtype=np.float32)
    sq = 56
    for ch in fen.split()[0]:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            idx = piece_map.get(ch)
            if idx is not None:
                bits[idx * 64 + sq] = 1.0
            sq += 1
    return bits


class PrevMoveDataset(Dataset):
    def __init__(self, tsv_path, move_vocab, max_rows=0):
        self.data = []  # (fen, move_idx, from_sq, to_sq)
        n_unknown = 0
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                move = row['prev_move']
                if move not in move_vocab:
                    n_unknown += 1
                    continue
                try:
                    from_sq, to_sq = uci_to_squares(move)
                except (IndexError, ValueError):
                    n_unknown += 1
                    continue
                self.data.append((row['fen'], move_vocab[move], from_sq, to_sq))
                if max_rows > 0 and len(self.data) >= max_rows:
                    break
        if n_unknown > 0:
            print(f"  Warning: {n_unknown} unknown moves skipped")
        print(f"  Dataset: {len(self.data)} samples, {len(move_vocab)} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, label, from_sq, to_sq = self.data[idx]
        return torch.from_numpy(fen_to_bitboards(fen)), label, from_sq, to_sq


class BinaryPrevMoveDataset(Dataset):
    """Loads pre-computed binary data. Returns PACKED int64 bitboards; unpacking
    happens on GPU in the training loop (batched, much faster than per-item).

    Files (with optional suffix for dedup variant):
      data/chess_x{suffix}.bin (N*12 uint64), data/chess_y{suffix}.bin (N uint16),
      data/chess_meta{suffix}.txt, data/chess_vocab.txt (shared vocab).
    """
    def __init__(self, data_dir="data", suffix=""):
        meta_path = os.path.join(data_dir, f"chess_meta{suffix}.txt")
        with open(meta_path) as f:
            meta = dict(line.strip().split('=') for line in f if '=' in line)
        self.n = int(meta['n_samples'])
        self.n_moves = int(meta['n_moves'])

        # Load packed bitboards as int64 torch tensor (zero-copy for __getitem__)
        x_np = np.fromfile(os.path.join(data_dir, f"chess_x{suffix}.bin"),
                           dtype=np.int64).reshape(self.n, 12)
        self.x = torch.from_numpy(x_np)  # (N, 12) int64
        y_np = np.fromfile(os.path.join(data_dir, f"chess_y{suffix}.bin"),
                           dtype=np.uint16).astype(np.int64)
        self.y = torch.from_numpy(y_np)  # (N,) int64

        # Load vocab
        vocab_path = os.path.join(data_dir, "chess_vocab.txt")
        self.idx_to_move = {}
        for line in open(vocab_path):
            m, i = line.strip().split('\t')
            self.idx_to_move[int(i)] = m

        # Vectorized from_sq/to_sq precompute from vocab + y
        fs_arr = np.zeros(self.n_moves, dtype=np.int64)
        ts_arr = np.zeros(self.n_moves, dtype=np.int64)
        for i, m in self.idx_to_move.items():
            try:
                fs, ts = uci_to_squares(m)
                fs_arr[i] = fs
                ts_arr[i] = ts
            except (IndexError, ValueError):
                pass
        # Now index y into these lookup tables (super fast, vectorized)
        self.from_sq = torch.from_numpy(fs_arr[y_np])
        self.to_sq = torch.from_numpy(ts_arr[y_np])

        print(f"  Binary dataset: {self.n} samples, {self.n_moves} classes "
              f"({self.x.element_size() * self.x.nelement() / 1e6:.0f} MB packed)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return packed int64 bitboards — unpack happens on GPU in train loop
        return self.x[idx], self.y[idx], self.from_sq[idx], self.to_sq[idx]


def unpack_bitboards_gpu(x_packed):
    """Unpack (batch, 12) int64 bitboards → (batch, 768) float32 on GPU.
    Each int64 has 64 bits where bit k = 1 if piece is at square k.
    Output layout: [piece0_sq0..63, piece1_sq0..63, ..., piece11_sq0..63]
    """
    # Shifts: (64,)
    shifts = torch.arange(64, dtype=x_packed.dtype, device=x_packed.device)
    # Broadcast: (batch, 12, 64) = ((batch, 12, 1) >> (64,)) & 1
    bits = ((x_packed.unsqueeze(-1) >> shifts) & 1).float()
    # Reshape to (batch, 768)
    return bits.reshape(x_packed.size(0), -1)


def evaluate_embeddings(model, device):
    """Quick sanity check: distance between known positions."""
    positions = {
        "sicilian":   "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "french":     "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "queens_gam": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "endgame_KP": "8/5k2/8/8/8/8/4PK2/8 w - - 0 1",
        "endgame_KR": "8/5k2/8/8/8/8/4RK2/8 w - - 0 1",
        "mid_open":   "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    }
    model.eval()
    embs = {}
    with torch.no_grad():
        for name, fen in positions.items():
            x = torch.from_numpy(fen_to_bitboards(fen)).unsqueeze(0).to(device)
            embs[name] = model.get_embedding(x).cpu().numpy()[0]

    pairs = [
        ("sicilian", "french", "same e4 family"),
        ("sicilian", "queens_gam", "e4 vs d4"),
        ("endgame_KP", "endgame_KR", "similar endgames"),
        ("mid_open", "endgame_KP", "midgame vs endgame"),
    ]
    print("    Embedding distances:")
    for a, b, desc in pairs:
        d = np.linalg.norm(embs[a] - embs[b])
        print(f"      {d:>6.1f}  {desc}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {args.arch}")

    if args.bin_dir:
        suffix = "_dedup" if args.dedup else ""
        print(f"Loading binary dataset from {args.bin_dir} "
              f"(suffix={suffix!r})...")
        ds = BinaryPrevMoveDataset(data_dir=args.bin_dir, suffix=suffix)
        n_moves = ds.n_moves
        # Save vocab info from binary for compatibility
        vocab = {ds.idx_to_move[i]: i for i in range(n_moves)}
    else:
        print("Building vocab...")
        vocab = build_move_vocab(args.data)
        n_moves = len(vocab)
        print(f"  {n_moves} moves")

        vocab_path = os.path.join(os.path.dirname(args.output), f"vocab_{args.arch}.txt")
        with open(vocab_path, 'w') as f:
            for m, i in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{m}\t{i}\n")

        print("Loading data...")
        ds = PrevMoveDataset(args.data, vocab, max_rows=args.max_samples)
    n = len(ds)
    n_val = max(1, n // 10)
    # Sequential split: first 90% train, last 10% val.
    # TSV was extracted in game order, so this keeps ~whole games on each side.
    # Avoids the position-adjacency leakage of random_split.
    train_ds = torch.utils.data.Subset(ds, range(0, n - n_val))
    val_ds = torch.utils.data.Subset(ds, range(n - n_val, n))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"  Train: {n - n_val}, Val: {n_val}")

    model = create_model(args.arch, n_moves, embed_dim=args.embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    is_dual = hasattr(model, 'is_dual') and model.is_dual

    # Resume from checkpoint if exists
    checkpoint_path = args.output.replace('.pt', '_checkpoint.pt')
    start_epoch = 1
    best_val_acc = 0.0
    if os.path.exists(checkpoint_path):
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.1f}%")

    n_batches_total = len(train_dl)
    log_every = max(1, n_batches_total // 20)  # log 20 times per epoch
    use_binary = args.bin_dir != ""
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        batch_idx = 0

        for x, y, from_sq, to_sq in train_dl:
            batch_idx += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            from_sq = from_sq.to(device, non_blocking=True)
            to_sq = to_sq.to(device, non_blocking=True)
            if use_binary:
                x = unpack_bitboards_gpu(x)  # (batch, 12) int64 → (batch, 768) float32

            if is_dual:
                logits_f, logits_t = model.forward_dual(x)
                loss = F.cross_entropy(logits_f, from_sq) + F.cross_entropy(logits_t, to_sq)
                # Accuracy: both from AND to correct
                pred_f = logits_f.argmax(1)
                pred_t = logits_t.argmax(1)
                train_correct += ((pred_f == from_sq) & (pred_t == to_sq)).sum().item()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                train_correct += (logits.argmax(1) == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_total += x.size(0)

            if batch_idx % log_every == 0:
                elapsed = time.time() - t0
                pct = batch_idx / n_batches_total * 100
                eta = elapsed / batch_idx * (n_batches_total - batch_idx)
                print(f"    batch {batch_idx}/{n_batches_total} ({pct:.0f}%) "
                      f"loss={train_loss/train_total:.4f} "
                      f"acc={train_correct/train_total*100:.1f}% "
                      f"elap={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y, from_sq, to_sq in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                from_sq = from_sq.to(device, non_blocking=True)
                to_sq = to_sq.to(device, non_blocking=True)
                if use_binary:
                    x = unpack_bitboards_gpu(x)
                if is_dual:
                    lf, lt = model.forward_dual(x)
                    val_correct += ((lf.argmax(1) == from_sq) & (lt.argmax(1) == to_sq)).sum().item()
                else:
                    val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>2}/{args.epochs}: "
              f"loss={train_loss/train_total:.4f} "
              f"train={train_acc:.1f}% val={val_acc:.1f}% "
              f"({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'arch': args.arch,
                'vocab': vocab,
                'embed_dim': args.embed_dim,
                'n_moves': n_moves,
                'val_acc': val_acc,
            }, args.output)
            print(f"    ★ Best ({val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            print(f"    No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping: no improvement in {args.patience} epochs")
                break

        # Save checkpoint every epoch (for resume on crash)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'arch': args.arch,
            'vocab': vocab,
            'embed_dim': args.embed_dim,
            'n_moves': n_moves,
        }, checkpoint_path)

        # Embedding sanity check every 5 epochs
        if epoch % 5 == 0:
            evaluate_embeddings(model, device)

    print(f"\nDone. Best val: {best_val_acc:.1f}%. Saved to {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(MODELS.keys()), default="mlp")
    ap.add_argument("--data", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--output", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers (0=serial main thread, 4+=parallel)")
    ap.add_argument("--bin-dir", default="",
                    help="If set, load pre-computed binary data from this dir "
                         "(chess_x.bin + chess_y.bin + chess_meta.txt). "
                         "Skips TSV parsing for much faster training.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stopping: break if val_acc doesn't improve for N epochs")
    ap.add_argument("--dedup", action="store_true",
                    help="Use deduplicated dataset (chess_x_dedup.bin / chess_y_dedup.bin / "
                         "chess_meta_dedup.txt). Trains CausalChess-no-mem. "
                         "Requires prepare_dedup_dataset.py to have been run first.")
    args = ap.parse_args()
    if not args.output:
        tag = "_dedup" if args.dedup else ""
        args.output = f"models/prev_move_{args.arch}{tag}.pt"
    os.makedirs("models", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
