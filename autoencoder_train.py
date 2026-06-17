#!/usr/bin/env python3
"""
autoencoder_train.py — Baseline: reconstruct-the-position autoencoder.

Same encoder as CausalChess CNN (identical backbone). Decoder is a single
linear layer that reconstructs the 12×8×8 = 768-bit bitboard. Loss is BCE
per bit. Purpose: a baseline that learns representations preserving
INFORMATION but without any task signal — to test whether task-driven
pretraining (previous-move prediction) adds value beyond reconstruction.

Hyperparameters mirror the CausalChess-mem run for apples-to-apples:
  batch 4096, lr 1e-3, cosine annealing, patience 5, same dataset.

Checkpoint format matches CausalChess so the downstream test suite (with
--ckpt flag) works out of the box.

Usage:
    python3 autoencoder_train.py --bin-dir data \
        --output models/ae_cnn.pt --epochs 100
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prev_move_models import create_model
from prev_move_train import BinaryPrevMoveDataset, unpack_bitboards_gpu


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    suffix = "_dedup" if args.dedup else ""
    print(f"Loading binary dataset (suffix={suffix!r})...")
    ds = BinaryPrevMoveDataset(data_dir=args.bin_dir, suffix=suffix)
    n = len(ds)
    n_val = max(1, n // 10)
    # Sequential split (same as CausalChess-mem for comparability)
    train_ds = torch.utils.data.Subset(ds, range(0, n - n_val))
    val_ds = torch.utils.data.Subset(ds, range(n - n_val, n))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"  Train: {n - n_val}, Val: {n_val}")

    model = create_model('ae', ds.n_moves, embed_dim=args.embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} params (AE: encoder + Linear decoder)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if exists
    checkpoint_path = args.output.replace('.pt', '_checkpoint.pt')
    start_epoch = 1
    best_val_bce = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"  Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_bce = ckpt.get('best_val_bce', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_val_bce={best_val_bce:.4f}")

    n_batches_total = len(train_dl)
    log_every = max(1, n_batches_total // 20)
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bits_correct = 0
        train_bits_total = 0
        t0 = time.time()
        batch_idx = 0

        for x_packed, _y, _f, _t in train_dl:
            batch_idx += 1
            x_packed = x_packed.to(device, non_blocking=True)
            x = unpack_bitboards_gpu(x_packed)  # (batch, 768) float32 binary

            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            # bit accuracy (sigmoid > 0.5 matches target)
            with torch.no_grad():
                pred = (logits > 0).float()
                train_bits_correct += (pred == x).sum().item()
                train_bits_total += x.numel()

            if batch_idx % log_every == 0:
                elapsed = time.time() - t0
                pct = batch_idx / n_batches_total * 100
                eta = elapsed / batch_idx * (n_batches_total - batch_idx)
                print(f"    batch {batch_idx}/{n_batches_total} ({pct:.0f}%) "
                      f"bce={train_loss/((batch_idx)*args.batch_size):.4f} "
                      f"bit_acc={100*train_bits_correct/train_bits_total:.2f}% "
                      f"elap={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_bits_correct = 0
        val_bits_total = 0
        val_n = 0
        with torch.no_grad():
            for x_packed, _y, _f, _t in val_dl:
                x_packed = x_packed.to(device, non_blocking=True)
                x = unpack_bitboards_gpu(x_packed)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, x)
                val_loss += loss.item() * x.size(0)
                pred = (logits > 0).float()
                val_bits_correct += (pred == x).sum().item()
                val_bits_total += x.numel()
                val_n += x.size(0)

        val_bce = val_loss / max(val_n, 1)
        val_bit_acc = 100 * val_bits_correct / max(val_bits_total, 1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>2}/{args.epochs}: "
              f"train_bce={train_loss/(n-n_val):.4f} "
              f"val_bce={val_bce:.4f} "
              f"val_bit_acc={val_bit_acc:.2f}% "
              f"({elapsed:.0f}s)")

        if val_bce < best_val_bce:
            best_val_bce = val_bce
            epochs_no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'arch': 'ae',
                'vocab': {},  # not used for AE
                'embed_dim': args.embed_dim,
                'n_moves': ds.n_moves,  # for checkpoint interface compat
                'val_acc': val_bit_acc,  # repurpose field for bit accuracy
                'val_bce': val_bce,
            }, args.output)
            print(f"    ★ Best (bce={val_bce:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping: no improvement in {args.patience} epochs")
                break

        # Resume checkpoint every epoch
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_bce': best_val_bce,
            'arch': 'ae',
            'vocab': {},
            'embed_dim': args.embed_dim,
            'n_moves': ds.n_moves,
        }, checkpoint_path)

    print(f"\nDone. Best val_bce: {best_val_bce:.4f}. Saved to {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="data")
    ap.add_argument("--output", default="models/ae_cnn.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dedup", action="store_true",
                    help="Use the dedup dataset (same as no-mem).")
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
