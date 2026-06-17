#!/usr/bin/env python3
"""
prev_move_models.py — Three architectures for previous-move prediction.

1. MLP:       768 → 512 → 512+res → 256 (embedding) → n_moves
2. Attention:  768 → 64 tokens of 12-d → self-attention → 256 (embedding) → n_moves
3. LSTM:       768 → 12 sequential piece-planes → LSTM → 256 (embedding) → n_moves

All share the same interface: forward(x) → logits, get_embedding(x) → 256-d vector.

Usage:
    python3 prev_move_train.py --arch mlp|attention|lstm --data lichess_data/prev_move_2400.tsv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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



# ── 1. MLP (baseline, already proven: 35.5%) ────────────────────

class MLPNet(nn.Module):
    def __init__(self, n_moves, embed_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, embed_dim)
        self.head = nn.Linear(embed_dim, n_moves)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + h1)
        emb = F.relu(self.fc3(h2))
        return self.head(emb)

    def get_embedding(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + h1)
        return F.relu(self.fc3(h2))


# ── 2. Attention (spatial: 64 squares as tokens) ────────────────
#
# Idea: treat the board as 64 tokens (one per square).
# Each token has 12 features (one per piece type: is there a WP, WN, ... BK?).
# Self-attention learns relationships between squares (e.g., rook attacks king).
# Pool → embedding → predict previous move.

class AttentionNet(nn.Module):
    def __init__(self, n_moves, embed_dim=256, n_heads=4, n_layers=3, d_model=64):
        super().__init__()
        self.d_model = d_model

        # Project 12 piece features per square → d_model
        self.input_proj = nn.Linear(12, d_model)

        # Learnable positional encoding (64 squares)
        self.pos_enc = nn.Parameter(torch.randn(64, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool (mean) → embedding
        self.fc_embed = nn.Linear(d_model, embed_dim)
        self.head = nn.Linear(embed_dim, n_moves)

    def _encode(self, x):
        # x: (batch, 768) → reshape to (batch, 64, 12)
        # Input is 12 bitboards × 64 squares, flattened as piece-major.
        # Reshape: [P0..P63, N0..N63, ..., K0..K63] → [sq0: P,N,B,R,Q,K,p,n,b,r,q,k; sq1: ...]
        b = x.size(0)
        pieces = x.view(b, 12, 64)       # (batch, 12_pieces, 64_squares)
        tokens = pieces.permute(0, 2, 1)  # (batch, 64_squares, 12_features)

        h = self.input_proj(tokens) + self.pos_enc.unsqueeze(0)
        h = self.transformer(h)
        pooled = h.mean(dim=1)            # (batch, d_model)
        emb = F.relu(self.fc_embed(pooled))
        return emb

    def forward(self, x):
        emb = self._encode(x)
        return self.head(emb)

    def get_embedding(self, x):
        return self._encode(x)


# ── 3. LSTM (sequential: 12 piece planes as sequence) ───────────
#
# Idea: treat the 12 bitboards as a SEQUENCE of 12 "observations".
# Each observation = 64 bits (one piece type's placement).
# LSTM processes them in order: WP → WN → WB → WR → WQ → WK → BP → ... → BK.
# The hidden state after all 12 = summary of the position.
# This captures inter-piece dependencies (e.g., pawns define structure,
# then knights/bishops make sense in that context).

class LSTMNet(nn.Module):
    def __init__(self, n_moves, embed_dim=256, hidden_dim=256, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project 64-bit plane → hidden_dim
        self.input_proj = nn.Linear(64, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=0.1
        )

        self.fc_embed = nn.Linear(hidden_dim, embed_dim)
        self.head = nn.Linear(embed_dim, n_moves)

    def _encode(self, x):
        # x: (batch, 768) → (batch, 12, 64)
        b = x.size(0)
        planes = x.view(b, 12, 64)  # 12 piece planes, each 64 squares

        h = F.relu(self.input_proj(planes))  # (batch, 12, hidden_dim)
        _, (h_n, _) = self.lstm(h)           # h_n: (n_layers, batch, hidden)
        last_hidden = h_n[-1]                 # (batch, hidden_dim)
        emb = F.relu(self.fc_embed(last_hidden))
        return emb

    def forward(self, x):
        emb = self._encode(x)
        return self.head(emb)

    def get_embedding(self, x):
        return self._encode(x)


# ── 4. CNN 8×8 (AlphaZero-style spatial) ────────────────────────
#
# Treat the board as 8×8 image with 12 channels (one per piece type).
# Small ResNet captures local spatial patterns (pawn chains, piece clusters).
# Proven architecture for chess (AlphaZero, Lc0).

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class CNNNet(nn.Module):
    def __init__(self, n_moves, embed_dim=256, n_filters=128, n_blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(12, n_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_filters)
        self.blocks = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_blocks)])
        # Global average pool → embed
        self.fc_embed = nn.Linear(n_filters, embed_dim)
        self.head = nn.Linear(embed_dim, n_moves)

    def _encode(self, x):
        b = x.size(0)
        # x: (batch, 768) → (batch, 12, 8, 8)
        board = x.view(b, 12, 8, 8)
        h = F.relu(self.input_bn(self.input_conv(board)))
        h = self.blocks(h)
        h = h.mean(dim=[2, 3])  # global avg pool → (batch, n_filters)
        return F.relu(self.fc_embed(h))

    def forward(self, x):
        return self.head(self._encode(x))

    def get_embedding(self, x):
        return self._encode(x)


# ── 5. Dual-head (from_square + to_square) ──────────────────────
#
# Instead of predicting 1 of ~1926 moves, predict:
#   from_square: 64 classes
#   to_square:   64 classes
# Reduces output space from 1926 → 128. Much easier to learn.
# For training, we decompose the UCI move into (from_sq, to_sq).
# This model needs special loss handling (see train script).

class DualHeadMLP(nn.Module):
    def __init__(self, n_moves, embed_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, embed_dim)
        self.head_from = nn.Linear(embed_dim, 64)
        self.head_to = nn.Linear(embed_dim, 64)
        # Also keep a combined head for compatibility with evaluation
        self.head = nn.Linear(embed_dim, n_moves)
        self.is_dual = True

    def forward(self, x):
        emb = self._encode(x)
        return self.head(emb)

    def forward_dual(self, x):
        emb = self._encode(x)
        return self.head_from(emb), self.head_to(emb)

    def _encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + h1)
        return F.relu(self.fc3(h2))

    def get_embedding(self, x):
        return self._encode(x)


# ── 6. CNN + Dual-head (best of both) ───────────────────────────

class CNNDualHead(nn.Module):
    def __init__(self, n_moves, embed_dim=256, n_filters=128, n_blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(12, n_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_filters)
        self.blocks = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_blocks)])
        self.fc_embed = nn.Linear(n_filters, embed_dim)
        self.head_from = nn.Linear(embed_dim, 64)
        self.head_to = nn.Linear(embed_dim, 64)
        self.head = nn.Linear(embed_dim, n_moves)
        self.is_dual = True

    def _encode(self, x):
        b = x.size(0)
        board = x.view(b, 12, 8, 8)
        h = F.relu(self.input_bn(self.input_conv(board)))
        h = self.blocks(h)
        h = h.mean(dim=[2, 3])
        return F.relu(self.fc_embed(h))

    def forward(self, x):
        return self.head(self._encode(x))

    def forward_dual(self, x):
        emb = self._encode(x)
        return self.head_from(emb), self.head_to(emb)

    def get_embedding(self, x):
        return self._encode(x)


# ── 7. CNN Autoencoder (baseline: reconstruction-only) ──────────
#
# Same encoder as CNNNet (identical architecture, for apples-to-apples
# comparison with CausalChess). Decoder is a single Linear(embed_dim, 768)
# that reconstructs the 12×8×8 = 768-bit bitboard representation.
#
# Loss: binary cross-entropy over 768 bits.
#
# The n_moves argument is kept for interface compatibility (ignored).

class CNNAutoencoder(nn.Module):
    def __init__(self, n_moves, embed_dim=256, n_filters=128, n_blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(12, n_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_filters)
        self.blocks = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_blocks)])
        self.fc_embed = nn.Linear(n_filters, embed_dim)
        self.decoder = nn.Linear(embed_dim, 768)  # reconstruct raw bitboard bits

    def _encode(self, x):
        b = x.size(0)
        board = x.view(b, 12, 8, 8)
        h = F.relu(self.input_bn(self.input_conv(board)))
        h = self.blocks(h)
        h = h.mean(dim=[2, 3])
        return F.relu(self.fc_embed(h))

    def forward(self, x):
        # Returns reconstruction logits (sigmoid applied in the loss).
        return self.decoder(self._encode(x))

    def get_embedding(self, x):
        return self._encode(x)


# ── Factory ──────────────────────────────────────────────────────

MODELS = {
    'mlp': MLPNet,
    'attention': AttentionNet,
    'lstm': LSTMNet,
    'cnn': CNNNet,
    'dual_mlp': DualHeadMLP,
    'cnn_dual': CNNDualHead,
    'ae': CNNAutoencoder,
    'simclr': CNNNet,  # same encoder; projection head is external (training only)
}

def create_model(arch, n_moves, embed_dim=256):
    cls = MODELS.get(arch)
    if cls is None:
        raise ValueError(f"Unknown arch {arch!r}. Available: {list(MODELS.keys())}")
    return cls(n_moves, embed_dim=embed_dim)
