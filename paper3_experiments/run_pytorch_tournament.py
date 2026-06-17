import torch
import chess
import sys
import os
from collections import defaultdict
import random

sys.path.append('nnue-pytorch')
sys.path.append('..')
from model.config import NNUELightningConfig
from model.lightning_module import NNUE
from prev_move_models import CNNNet
from model.modules.features.halfka_v2_hm import _halfka_idx

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
    
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            w_idx = _halfka_idx(True, wk_sq, sq, p)
            white_indices.append(w_idx)
            b_idx = _halfka_idx(False, bk_sq, sq, p)
            black_indices.append(b_idx)
            
    while len(white_indices) < 32: white_indices.append(0)
    while len(black_indices) < 32: black_indices.append(0)
        
    return white_indices[:32], black_indices[:32]

class PyTorchEngine:
    def __init__(self, mode="baseline"):
        self.mode = mode
        print(f"Loading {mode} engine...")
        
        config = NNUELightningConfig()
        self.model = NNUE(
            config=config, max_epoch=1, num_batches_per_epoch=1
        ).to(device)
        self.model.load_state_dict(torch.load(f"best_model_{mode}.pt", map_location=device))
        self.model.eval()
        
        if mode == "nnuecc":
            self.cnn = CNNNet(n_moves=1928).to(device)
            state = torch.load("../models/prev_move_cnn_v2.pt", map_location=device)
            self.cnn.load_state_dict(state['model'] if 'model' in state else state)
            self.cnn.eval()

    def evaluate_batch(self, boards):
        # boards: list of chess.Board
        batch_size = len(boards)
        us_list, them_list = [], []
        wi_list, bi_list = [], []
        planes_batch = []
        
        for board in boards:
            fen = board.fen()
            us = 1.0 if board.turn == chess.WHITE else 0.0
            us_list.append([us])
            them_list.append([1.0 - us])
            
            wi, bi = process_fen_halfkpa(fen)
            wi_list.append(wi)
            bi_list.append(bi)
            
            if self.mode == "nnuecc":
                planes_batch.append(fen_to_planes(fen))

        us_t = torch.tensor(us_list, dtype=torch.float32).to(device)
        them_t = torch.tensor(them_list, dtype=torch.float32).to(device)
        wi_t = torch.tensor(wi_list, dtype=torch.int64).to(device)
        bi_t = torch.tensor(bi_list, dtype=torch.int64).to(device)
        
        # dummy values
        wv_t = torch.ones_like(wi_t, dtype=torch.float32).to(device)
        bv_t = torch.ones_like(bi_t, dtype=torch.float32).to(device)
        psqt_t = torch.zeros((batch_size,), dtype=torch.int64).to(device)
        ls_t = torch.zeros((batch_size,), dtype=torch.int64).to(device)
        
        causal_embeddings = None
        if self.mode == "nnuecc":
            batch_planes = torch.stack(planes_batch).to(device)
            with torch.no_grad():
                x = batch_planes.view(batch_size, -1)
                causal_embeddings = self.cnn.get_embedding(x)
        else:
            causal_embeddings = torch.zeros((batch_size, 256), dtype=torch.float32).to(device)

        with torch.no_grad():
            scores = self.model.model(
                us_t, them_t, wi_t, wv_t, bi_t, bv_t, psqt_t, ls_t, causal_embeddings
            )
            
        return scores.squeeze(-1).cpu().numpy()

    def get_best_move(self, board, depth=2):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        best_score = float('-inf')
        best_move = None
        
        # 2-ply search (Minimax)
        for move in legal_moves:
            board.push(move)
            
            # Enemy's turn
            enemy_moves = list(board.legal_moves)
            if not enemy_moves:
                if board.is_checkmate():
                    score = 10000.0 # We checkmated them!
                else:
                    score = 0.0 # Stalemate
            else:
                next_boards = []
                for e_move in enemy_moves:
                    board.push(e_move)
                    next_boards.append(board.copy())
                    board.pop()
                    
                # Evaluate enemy's responses
                enemy_scores = self.evaluate_batch(next_boards)
                # The enemy will pick the move that minimizes our score (minimizes from their perspective? No, NNUE outputs from side-to-move perspective)
                # Actually, enemy_scores is from our perspective (after enemy move, it's our turn again!)
                # Wait: NNUE always evaluates from the side to move. 
                # After enemy_move, it's OUR turn. So NNUE outputs score for US.
                # The enemy wants to MINIMIZE the score for us.
                score = enemy_scores.min()
                
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move


def run_match(engine_w, engine_b, opening_moves):
    board = chess.Board()
    for move_str in opening_moves:
        board.push_uci(move_str)
        
    moves_played = 0
    while not board.is_game_over() and moves_played < 200:
        if board.turn == chess.WHITE:
            move = engine_w.get_best_move(board)
        else:
            move = engine_b.get_best_move(board)
            
        if move is None:
            break
        board.push(move)
        moves_played += 1
        
    res = board.result()
    if res == "1/2-1/2" and moves_played >= 200:
        res = "1/2-1/2" # Draw by move limit
    
    return res

def main():
    print("Inicializando Motores de Python-PyTorch...")
    baseline = PyTorchEngine("baseline")
    nnuecc = PyTorchEngine("nnuecc")
    
    # 10 simple standard openings to force variety
    openings = [
        ["e2e4", "e7e5"],
        ["d2d4", "d7d5"],
    ]
    
    stats = {"Baseline": 0.0, "NNUECC": 0.0}
    
    print("\n--- INICIANDO TORNEO A 2-PLY (4 Partidas) ---")
    for i, op in enumerate(openings):
        print(f"\nRonda {i*2 + 1} (White: Baseline vs Black: NNUECC) Opening: {op}")
        res1 = run_match(baseline, nnuecc, op)
        print(f"Resultado: {res1}")
        if res1 == "1-0": stats["Baseline"] += 1.0
        elif res1 == "0-1": stats["NNUECC"] += 1.0
        else: stats["Baseline"] += 0.5; stats["NNUECC"] += 0.5
        
        print(f"Ronda {i*2 + 2} (White: NNUECC vs Black: Baseline) Opening: {op}")
        res2 = run_match(nnuecc, baseline, op)
        print(f"Resultado: {res2}")
        if res2 == "1-0": stats["NNUECC"] += 1.0
        elif res2 == "0-1": stats["Baseline"] += 1.0
        else: stats["Baseline"] += 0.5; stats["NNUECC"] += 0.5
        
        print(f"Score actual -> Baseline: {stats['Baseline']} - NNUECC: {stats['NNUECC']}")

    print("\n=== RESULTADO FINAL ===")
    print(f"Baseline: {stats['Baseline']}")
    print(f"NNUECC:   {stats['NNUECC']}")

if __name__ == "__main__":
    main()
