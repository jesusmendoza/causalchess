import json
import torch
import chess
import sys
import os

sys.path.append('..')
sys.path.append('nnue-pytorch')
from prev_move_models import CNNNet
from model.modules.features.halfka_v2_hm import _halfka_idx

# Config
INPUT_JSONL = "../test_tight_15cp.jsonl"
OUTPUT_PT = "nnuecc_dataset.pt"
MODEL_PATH = "../models/prev_move_cnn_v2.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fen_to_planes(fen):
    board = chess.Board(fen)
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            r, c = divmod(sq, 8)
            p_str = piece.symbol()
            planes[piece_idx[p_str], r, c] = 1.0
    return planes

def process_fen_halfkpa(fen):
    board = chess.Board(fen)
    white_indices = []
    black_indices = []
    
    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)
    
    # HalfKAv2 hm rules: 
    # white pov uses True, black pov uses False.
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            # White perspective
            w_idx = _halfka_idx(True, wk_sq, sq, p)
            white_indices.append(w_idx)
            # Black perspective
            b_idx = _halfka_idx(False, bk_sq, sq, p)
            black_indices.append(b_idx)
            
    # Pad to 32
    while len(white_indices) < 32:
        white_indices.append(0)
    while len(black_indices) < 32:
        black_indices.append(0)
        
    return white_indices[:32], black_indices[:32]

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../test_tight_15cp.jsonl")
    parser.add_argument("--output", default="nnuecc_dataset.pt")
    args = parser.parse_args()

    print("Loading PMP model...")
    model = CNNNet(n_moves=1928).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model_state = state['model'] if 'model' in state else state
    model.load_state_dict(model_state)
    model.eval()

    us_list, them_list = [], []
    wi_list, bi_list = [], []
    outcome_list, score_list = [], []
    embed_list = []

    print(f"Parsing {args.input}...")
    with open(args.input, 'r') as f:
        lines = f.readlines()
        
    batch_size = 512
    planes_batch = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        if args.input.endswith('.tsv'):
            parts = line.split('\t')
            if parts[0] == 'fen' or len(parts) < 2:
                continue
            fen = parts[0]
            score_cp = float(parts[1])
            outcome = 0.5 # Unknown outcome for pure eval
        else:
            data = json.loads(line)
            fen = data['fen']
            score_cp = data['value'] * 300.0 # Approximate CP
            outcome = 0.5
        
        board = chess.Board(fen)
        us = 1.0 if board.turn == chess.WHITE else 0.0
        them = 1.0 - us
        
        us_list.append([us])
        them_list.append([them])
        score_list.append([score_cp])
        outcome_list.append([outcome])
        
        wi, bi = process_fen_halfkpa(fen)
        wi_list.append(wi)
        bi_list.append(bi)
        
        planes_batch.append(fen_to_planes(fen))
        
        if len(planes_batch) == batch_size or i == len(lines) - 1:
            batch_t = torch.stack(planes_batch).to(device)
            with torch.no_grad():
                # Get embeddings from the model's bottleneck directly
                x = batch_t.view(batch_t.size(0), -1) # CNNNet's get_embedding expects flat 768 vector
                emb = model.get_embedding(x) # 256-d embedding
                
            embed_list.append(emb.cpu())
            planes_batch = []
            print(f"Processed {i+1}/{len(lines)}", end='\r')

    print("\nSaving dataset to tensor format...")
    # Stack everything
    dataset = {
        'us': torch.tensor(us_list, dtype=torch.float32),
        'them': torch.tensor(them_list, dtype=torch.float32),
        'white_indices': torch.tensor(wi_list, dtype=torch.int64),
        'black_indices': torch.tensor(bi_list, dtype=torch.int64),
        'score': torch.tensor(score_list, dtype=torch.float32),
        'outcome': torch.tensor(outcome_list, dtype=torch.float32),
        'causal_embeddings': torch.cat(embed_list, dim=0)
    }
    
    # White/Black values are 1.0 for valid pieces, 0.0 for padding.
    # In our simplistic pad (using 0), we can just set values to 1.0 for now, 
    # or actually compute which ones are valid.
    dataset['white_values'] = (dataset['white_indices'] > 0).float()
    dataset['black_values'] = (dataset['black_indices'] > 0).float()
    
    # PSQT and LayerStack indices (Stockfish usually expects them)
    # We can fake them with zeros if not strictly doing bucketed networks, or compute them.
    # For now, zero tensors of appropriate shape:
    dataset['psqt_indices'] = torch.zeros(len(lines), dtype=torch.int64)
    dataset['layer_stack_indices'] = torch.zeros(len(lines), dtype=torch.int64)
    
    torch.save(dataset, args.output)
    print(f"Done! Saved {len(us_list)} positions to {args.output}")

if __name__ == "__main__":
    main()
