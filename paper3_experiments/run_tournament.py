import chess
import chess.engine
import chess.pgn
import sys
from concurrent.futures import ThreadPoolExecutor

BASELINE_CMD = "stockfish"
NNUECC_CMD = "stockfish"
ROUNDS = 100
TIME_LIMIT = 0.5 # seconds per move (fixed depth/time is safer for strict comparison)
DEPTH_LIMIT = 8

def play_game(round_num):
    print(f"[{round_num}] Arrancando engines...")
    engine_w = chess.engine.SimpleEngine.popen_uci(BASELINE_CMD)
    engine_b = chess.engine.SimpleEngine.popen_uci(NNUECC_CMD)
    print(f"[{round_num}] Engines arrancados. Configurando...")
    
    # Configure engines
    engine_w.configure({"EvalFile": "baseline.nnue"})
    engine_b.configure({"EvalFile": "nnuecc.nnue"})
    print(f"[{round_num}] Configuración lista.")
    
    if round_num % 2 != 0:
        # Swap colors
        engine_w, engine_b = engine_b, engine_w
        white_name = "NNUECC"
        black_name = "Baseline"
    else:
        white_name = "Baseline"
        black_name = "NNUECC"

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Event"] = "NNUECC Ceiling Tournament"
    
    node = game
    limit = chess.engine.Limit(depth=6)
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = engine_w.play(board, limit)
        else:
            result = engine_b.play(board, limit)
            
        board.push(result.move)
        node = node.add_variation(result.move)
        
    game.headers["Result"] = board.result()
    
    engine_w.quit()
    engine_b.quit()
    
    result_str = board.result()
    print(f"Round {round_num:3d} | White: {white_name:10s} vs Black: {black_name:10s} | Result: {result_str}")
    return white_name, black_name, result_str

def main():
    print("=== Iniciando Torneo: Baseline vs NNUECC ===")
    results = {"Baseline": 0.0, "NNUECC": 0.0}
    
    # Run sequentially or parallel
    for i in range(1, ROUNDS + 1):
        w, b, res = play_game(i)
        
        if res == "1-0":
            results[w] += 1.0
        elif res == "0-1":
            results[b] += 1.0
        else:
            results[w] += 0.5
            results[b] += 0.5
            
        print(f"  Score Actual -> Baseline: {results['Baseline']} - NNUECC: {results['NNUECC']}")

    print("\n=== RESULTADO FINAL ===")
    print(f"Baseline: {results['Baseline']}")
    print(f"NNUECC:   {results['NNUECC']}")

if __name__ == "__main__":
    main()
