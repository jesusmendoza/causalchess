import torch
import chess
import sys
import os
import gc

sys.path.append('..')
sys.path.append('nnue-pytorch')
from prev_move_models import CNNNet
from model.modules.features.halfka_v2_hm import _halfka_idx

# Config
INPUT_TSV = "lichess_15m_evals.tsv"
OUTPUT_DIR = "dataset_15M"
MODEL_PATH = "../models/prev_move_cnn_v2.pt"
CHUNK_SIZE = 1_000_000  # Guardar archivos de 1 millón de posiciones para no saturar la RAM

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

def save_chunk(chunk_id, us_list, them_list, wi_list, wv_list, bi_list, bv_list, outcomes_list, scores_list, embeddings_list, count_in_chunk):
    print(f"\nGuardando chunk {chunk_id:02d} ({count_in_chunk} posiciones)...")
    final_embeddings = torch.cat(embeddings_list, dim=0)
    
    data = {
        'us': torch.tensor(us_list, dtype=torch.float32),
        'them': torch.tensor(them_list, dtype=torch.float32),
        'white_indices': torch.tensor(wi_list, dtype=torch.long),
        'white_values': torch.tensor(wv_list, dtype=torch.float32),
        'black_indices': torch.tensor(bi_list, dtype=torch.long),
        'black_values': torch.tensor(bv_list, dtype=torch.float32),
        'outcome': torch.tensor(outcomes_list, dtype=torch.float32),
        'score': torch.tensor(scores_list, dtype=torch.float32),
        'psqt_indices': torch.zeros((count_in_chunk, 1), dtype=torch.long),
        'layer_stack_indices': torch.zeros((count_in_chunk, 1), dtype=torch.long),
        'causal_embeddings': final_embeddings
    }
    
    out_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id:02d}.pt")
    torch.save(data, out_path)
    print(f"-> {out_path} guardado con éxito. Limpiando memoria RAM...")
    
    # Limpieza extrema de memoria RAM
    del data, final_embeddings
    gc.collect()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Cargando CNNNet para incrustar embeddings causales...")
    cnn = CNNNet(n_moves=1928).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    cnn.load_state_dict(state['model'] if 'model' in state else state)
    cnn.eval()

    us_list, them_list = [], []
    wi_list, bi_list = [], []
    wv_list, bv_list = [], []
    outcomes_list, scores_list = [], []
    embeddings_list = []
    
    print(f"Abriendo {INPUT_TSV}...")
    with open(INPUT_TSV, "r", encoding="utf-8") as f:
        f.readline() # Header
        
        batch_planes = []
        global_count = 0
        chunk_count = 0
        chunk_id = 1
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            fen = parts[0]
            cp = float(parts[1])
            
            board = chess.Board(fen)
            turn = board.turn
            us = 1.0 if turn == chess.WHITE else 0.0
            them = 1.0 - us
            
            wi, bi = process_fen_halfkpa(fen)
            
            outcome = 0.5 # Placeholder, solo aprendemos del score
            
            us_list.append([us])
            them_list.append([them])
            wi_list.append(wi)
            bi_list.append(bi)
            wv_list.append([1.0] * 32)
            bv_list.append([1.0] * 32)
            scores_list.append([cp])
            outcomes_list.append([outcome])
            
            batch_planes.append(fen_to_planes(fen))
            global_count += 1
            chunk_count += 1
            
            if len(batch_planes) == 5000:
                with torch.no_grad():
                    b_planes = torch.stack(batch_planes).to(device)
                    x = b_planes.view(b_planes.size(0), -1)
                    emb = cnn.get_embedding(x).cpu()
                    embeddings_list.append(emb)
                batch_planes = []
                print(f"Chunk {chunk_id:02d} | Procesadas {chunk_count}/{CHUNK_SIZE} posiciones...", end="\r")
            
            # Guardar chunk si se alcanzó el límite
            if chunk_count >= CHUNK_SIZE:
                if len(batch_planes) > 0:
                    with torch.no_grad():
                        b_planes = torch.stack(batch_planes).to(device)
                        x = b_planes.view(b_planes.size(0), -1)
                        emb = cnn.get_embedding(x).cpu()
                        embeddings_list.append(emb)
                    batch_planes = []
                
                save_chunk(chunk_id, us_list, them_list, wi_list, wv_list, bi_list, bv_list, outcomes_list, scores_list, embeddings_list, chunk_count)
                
                # Reiniciar listas para el siguiente chunk
                us_list, them_list = [], []
                wi_list, bi_list = [], []
                wv_list, bv_list = [], []
                outcomes_list, scores_list = [], []
                embeddings_list = []
                
                chunk_id += 1
                chunk_count = 0
                
        # Guardar el último chunk si quedaron elementos rezagados
        if chunk_count > 0:
            if len(batch_planes) > 0:
                with torch.no_grad():
                    b_planes = torch.stack(batch_planes).to(device)
                    x = b_planes.view(b_planes.size(0), -1)
                    emb = cnn.get_embedding(x).cpu()
                    embeddings_list.append(emb)
            save_chunk(chunk_id, us_list, them_list, wi_list, wv_list, bi_list, bv_list, outcomes_list, scores_list, embeddings_list, chunk_count)
            
    print(f"\n¡Tensorización de {global_count} posiciones completada en múltiples chunks! (Sin saturar RAM)")

if __name__ == "__main__":
    main()
