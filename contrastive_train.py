#!/usr/bin/env python3
"""
contrastive_train.py — SimCLR-style baseline. Contrastive pretraining with
color-symmetry + vertical-flip as the augmentation.

For each position we create two views:
  v1 = original 12-bitboard position
  v2 = color-swapped + vertically-flipped position (same strategic structure,
       board seen from the other side)

Both views pass through the SAME encoder. A small projection head maps the
256-dim embedding to a 128-d contrastive space. NT-Xent (normalized
temperature cross-entropy) pulls matching pairs together and pushes
non-matching apart.

At inference time we discard the projection head and use the encoder output
as the representation (standard SimCLR practice).

Same encoder architecture as CausalChess (CNNNet). Same hyperparameters as
mem / AE (batch 4096, lr 1e-3, cosine annealing, patience 5).

Checkpoint saved in the format consumed by the downstream test suite.

Usage:
    python3 contrastive_train.py --bin-dir data --output models/simclr_cnn.pt
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


def _byteswap_u64(x):
    """Reverse byte order of each int64 element. Bytes 0..7 → 7..0.
    For chess bitboards this is a vertical flip (rank 0 ↔ rank 7 etc.).

    Works on signed int64 tensors (bit 63 may be set). Extracts each byte
    via (x >> (8*i)) & 0xFF — the arithmetic right shift sign-extends, but
    the & 0xFF mask keeps only the low byte, which is the byte we want.
    """
    result = torch.zeros_like(x)
    for i in range(8):
        byte = (x >> (8 * i)) & 0xFF         # extract byte i (always 0..255)
        result = result | (byte << (8 * (7 - i)))
    return result


def augment_color_flip(x_packed):
    """Color-swap + vertical-flip augmentation on packed bitboards.

    Input:  (batch, 12) int64 — original bitboards (pieces × 64 squares).
    Output: (batch, 12) int64 — same strategic board seen from the other
            side: board flipped vertically AND piece colors swapped.

    This is a chess-meaningful symmetry: the resulting position has the
    same strategic structure (threats, development, pawn chains) as the
    original, just with colors reversed. A well-trained embedding should
    recognize both as semantically similar.
    """
    flipped = _byteswap_u64(x_packed)
    white, black = flipped[:, 0:6], flipped[:, 6:12]
    return torch.cat([black, white], dim=1)


class SimCLRModel(nn.Module):
    """CausalChess encoder + 2-layer projection head for contrastive training."""
    def __init__(self, embed_dim=256, proj_dim=128):
        super().__init__()
        # Reuse the CNN encoder; n_moves is unused but required by create_model
        self.encoder = create_model('cnn', n_moves=1, embed_dim=embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        z = self.encoder.get_embedding(x)
        return self.proj(z)

    def get_embedding(self, x):
        return self.encoder.get_embedding(x)


def nt_xent_loss(z, temperature=0.1):
    """NT-Xent loss (normalized temperature cross-entropy).

    Input z: (2N, d) — first N are view-1 embeddings, next N are view-2.
    Positive pair: row i matches row i+N.
    """
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature            # (2N, 2N)
    # Mask self-similarity (diagonal → -inf)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    # Targets: row i → i+N (mod 2N)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    suffix = "_dedup" if args.dedup else ""
    print(f"Loading binary dataset (suffix={suffix!r})...")
    ds = BinaryPrevMoveDataset(data_dir=args.bin_dir, suffix=suffix)
    n = len(ds)
    n_val = max(1, n // 10)
    train_ds = torch.utils.data.Subset(ds, range(0, n - n_val))
    val_ds = torch.utils.data.Subset(ds, range(n - n_val, n))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"  Train: {n - n_val}, Val: {n_val}")

    model = SimCLRModel(embed_dim=args.embed_dim, proj_dim=128).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Model: {total_params:,} params (encoder: {enc_params:,}, projection: {total_params-enc_params:,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_path = args.output.replace('.pt', '_checkpoint.pt')
    start_epoch = 1
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"  Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    n_batches_total = len(train_dl)
    log_every = max(1, n_batches_total // 20)
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        t0 = time.time()
        batch_idx = 0

        for x_packed, _y, _f, _t in train_dl:
            batch_idx += 1
            x_packed = x_packed.to(device, non_blocking=True)

            # Build two views
            x_packed_aug = augment_color_flip(x_packed)

            x1 = unpack_bitboards_gpu(x_packed)      # (B, 768) float
            x2 = unpack_bitboards_gpu(x_packed_aug)

            z1 = model(x1)   # (B, 128)
            z2 = model(x2)
            z = torch.cat([z1, z2], dim=0)  # (2B, 128)

            loss = nt_xent_loss(z, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x1.size(0)
            train_n += x1.size(0)

            if batch_idx % log_every == 0:
                elapsed = time.time() - t0
                pct = batch_idx / n_batches_total * 100
                eta = elapsed / batch_idx * (n_batches_total - batch_idx)
                print(f"    batch {batch_idx}/{n_batches_total} ({pct:.0f}%) "
                      f"loss={train_loss/train_n:.4f} "
                      f"elap={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x_packed, _y, _f, _t in val_dl:
                x_packed = x_packed.to(device, non_blocking=True)
                x_packed_aug = augment_color_flip(x_packed)
                x1 = unpack_bitboards_gpu(x_packed)
                x2 = unpack_bitboards_gpu(x_packed_aug)
                z1 = model(x1)
                z2 = model(x2)
                z = torch.cat([z1, z2], dim=0)
                loss = nt_xent_loss(z, temperature=args.temperature)
                val_loss += loss.item() * x1.size(0)
                val_n += x1.size(0)

        train_loss /= max(1, train_n)
        val_loss /= max(1, val_n)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>2}/{args.epochs}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save encoder-compatible checkpoint. The downstream test suite
            # expects arch='cnn' and loads via create_model. We save the
            # encoder's state_dict under the canonical key so load_state_dict
            # works on a freshly-constructed CNNNet.
            torch.save({
                'model': model.encoder.state_dict(),
                'arch': 'cnn',   # so downstream tests call create_model('cnn')
                'vocab': {},
                'embed_dim': args.embed_dim,
                'n_moves': 1,    # head ignored for probes/style/etc.
                'val_acc': -val_loss,  # more-negative loss = better; for display
                'val_loss_ntxent': val_loss,
                'trained_as': 'simclr',
            }, args.output)
            print(f"    ★ Best (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping")
                break

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'embed_dim': args.embed_dim,
        }, checkpoint_path)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}. Saved to {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="data")
    ap.add_argument("--output", default="models/simclr_cnn.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="NT-Xent temperature. 0.1-0.5 typical; smaller = sharper.")
    ap.add_argument("--dedup", action="store_true",
                    help="Use the dedup dataset.")
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
