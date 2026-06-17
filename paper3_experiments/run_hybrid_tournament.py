import torch
import chess
import chess.engine
import sys
import time

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
            white_indices.append(_halfka_idx(True, wk_sq, sq, p))
            black_indices.append(_halfka_idx(False, bk_sq, sq, p))
            
    while len(white_indices) < 32: white_indices.append(0)
    while len(black_indices) < 32: black_indices.append(0)
        
    return white_indices[:32], black_indices[:32]

class HybridEngine:
    def __init__(self, mode="baseline"):
        self.mode = mode
        self.sf = chess.engine.SimpleEngine.popen_uci("stockfish")
        
        if mode == "stockfish":
            return
            
        config = NNUELightningConfig()
        self.model = NNUE(config=config, max_epoch=1, num_batches_per_epoch=1).to(device)
        self.model.load_state_dict(torch.load(f"best_model_{mode}.pt", map_location=device))
        self.model.eval()
        
        if mode == "nnuecc":
            self.cnn = CNNNet(n_moves=1928).to(device)
            state = torch.load("../models/prev_move_cnn_v2.pt", map_location=device)
            self.cnn.load_state_dict(state['model'] if 'model' in state else state)
            self.cnn.eval()

    def evaluate_batch(self, boards):
        batch_size = len(boards)
        us_list, wi_list, bi_list, planes_batch = [], [], [], []
        
        for board in boards:
            fen = board.fen()
            us = 1.0 if board.turn == chess.WHITE else 0.0
            us_list.append([us])
            wi, bi = process_fen_halfkpa(fen)
            wi_list.append(wi)
            bi_list.append(bi)
            if self.mode == "nnuecc":
                planes_batch.append(fen_to_planes(fen))

        us_t = torch.tensor(us_list, dtype=torch.float32).to(device)
        them_t = 1.0 - us_t
        wi_t = torch.tensor(wi_list, dtype=torch.int64).to(device)
        bi_t = torch.tensor(bi_list, dtype=torch.int64).to(device)
        
        wv_t = torch.ones_like(wi_t, dtype=torch.float32).to(device)
        bv_t = torch.ones_like(bi_t, dtype=torch.float32).to(device)
        psqt_t = torch.zeros((batch_size,), dtype=torch.int64).to(device)
        ls_t = torch.zeros((batch_size,), dtype=torch.int64).to(device)
        
        if self.mode == "nnuecc":
            with torch.no_grad():
                x = torch.stack(planes_batch).to(device).view(batch_size, -1)
                causal_embeddings = self.cnn.get_embedding(x)
        else:
            causal_embeddings = torch.zeros((batch_size, 256), dtype=torch.float32).to(device)

        with torch.no_grad():
            scores = self.model.model(us_t, them_t, wi_t, wv_t, bi_t, bv_t, psqt_t, ls_t, causal_embeddings)
        return scores.squeeze(-1).cpu().numpy()

    def get_best_move(self, board, depth=10):
        # 1. Use Stockfish to find the top 5 candidate moves at Depth 10
        info = self.sf.analyse(board, chess.engine.Limit(depth=depth), multipv=5)
        candidates = [res["pv"][0] for res in info if "pv" in res]
        
        if not candidates:
            legal = list(board.legal_moves)
            return legal[0] if legal else None
            
        if self.mode == "stockfish":
            return candidates[0]
            
        # 2. Re-rank the top 5 moves using our PyTorch PMP / Baseline model
        # We evaluate the LEAF node of the Stockfish PV, so PyTorch benefits from the 10-ply search tree
        leaf_boards = []
        valid_candidates = []
        for res in info:
            if "pv" in res:
                temp_board = board.copy()
                for m in res["pv"]:
                    temp_board.push(m)
                leaf_boards.append(temp_board)
                valid_candidates.append(res["pv"][0])
                
        if not valid_candidates:
            return candidates[0]
            
        # 3. Evaluate the leaf boards
        scores = self.evaluate_batch(leaf_boards)
        
        # We need to determine if we want to maximize or minimize the score.
        # NNUE outputs score from the perspective of the side-to-move of the leaf board.
        # If the PV length is EVEN, the side-to-move at the leaf is the SAME as the root. We want to MAXIMIZE.
        # If the PV length is ODD, the side-to-move at the leaf is the OPPONENT. We want to MINIMIZE.
        # Let's adjust all scores to be from the ROOT's perspective.
        root_scores = []
        for i, res in enumerate(info):
            if "pv" in res:
                pv_len = len(res["pv"])
                score = scores[len(root_scores)]
                if pv_len % 2 != 0:
                    score = -score # Flip to our perspective
                root_scores.append(score)
                
        best_idx = torch.tensor(root_scores).argmax().item()
        return valid_candidates[best_idx]

    def close(self):
        self.sf.quit()

def main():
    print("=== Preparando Torneo Híbrido (Triangular: Stockfish vs Baseline vs NNUECC) ===")
    print("Cargando motores en GPU...")
    engine_base = HybridEngine("baseline")
    engine_nnuecc = HybridEngine("nnuecc")
    engine_sf = HybridEngine("stockfish")
    print("Motores listos.\n")
    
    matchups = [
        ("Stockfish", engine_sf, "Baseline", engine_base),
        ("Stockfish", engine_sf, "NNUECC", engine_nnuecc),
        ("Baseline", engine_base, "NNUECC", engine_nnuecc)
    ]
    
    rounds_per_matchup = 100
    final_results = []
    
    with open("tournament_triangular_results.txt", "w") as f:
        f.write("=== RESULTADOS DEL TORNEO TRIANGULAR ===\n\n")
        
    for m_name_a, m_eng_a, m_name_b, m_eng_b in matchups:
        print(f"\n\n--- INICIANDO MATCH: {m_name_a} vs {m_name_b} ---")
        stats = {m_name_a: 0.0, m_name_b: 0.0}
        
        for i in range(1, rounds_per_matchup + 1):
            board = chess.Board()
            if i % 2 != 0:
                white, black = m_eng_a, m_eng_b
                w_name, b_name = m_name_a, m_name_b
            else:
                white, black = m_eng_b, m_eng_a
                w_name, b_name = m_name_b, m_name_a
                
            print(f"\n[Ronda {i:03d}] Blancas: {w_name:10s} vs Negras: {b_name:10s} -> Jugando...", end="", flush=True)
            
            moves = 0
            while not board.is_game_over() and moves < 150:
                if board.turn == chess.WHITE:
                    move = white.get_best_move(board, depth=10)
                else:
                    move = black.get_best_move(board, depth=10)
                board.push(move)
                moves += 1
                if moves % 10 == 0:
                    print(".", end="", flush=True)
                    
            res = board.result()
            if res == "1/2-1/2" and moves >= 150:
                res = "1/2-1/2"
                
            print(f" Fin! Resultado: {res}")
            
            if res == "1-0":
                stats[w_name] += 1.0
            elif res == "0-1":
                stats[b_name] += 1.0
            else:
                stats[w_name] += 0.5
                stats[b_name] += 0.5
                
            print(f"Score Parcial -> {m_name_a}: {stats[m_name_a]} | {m_name_b}: {stats[m_name_b]}")
            
            with open("tournament_triangular_results.txt", "a") as f:
                f.write(f"[{m_name_a} vs {m_name_b}] Ronda {i}: W={w_name}, B={b_name} | Resultado={res} | Score: {m_name_a}={stats[m_name_a]}, {m_name_b}={stats[m_name_b]}\n")
        
        print(f"\n=== FIN MATCH {m_name_a} vs {m_name_b} ===")
        print(f"Resultado final: {m_name_a}: {stats[m_name_a]} - {m_name_b}: {stats[m_name_b]}")
        final_results.append((m_name_a, stats[m_name_a], m_name_b, stats[m_name_b]))

    print("\n" + "="*50)
    print("TABLA DE RESULTADOS FINALES")
    print("="*50)
    print(f"{'Matchup':^30} | {'Score A':^7} | {'Score B':^7}")
    print("-" * 50)
    for a_name, a_score, b_name, b_score in final_results:
        match_str = f"{a_name} vs {b_name}"
        print(f"{match_str:^30} | {a_score:^7.1f} | {b_score:^7.1f}")
    print("=" * 50 + "\n")
    
    engine_base.close()
    engine_nnuecc.close()
    engine_sf.close()

if __name__ == "__main__":
    main()
