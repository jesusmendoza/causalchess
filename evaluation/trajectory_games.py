"""Canonical, validated source game data for trajectory analyses."""
from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn


KASPAROV_TOPALOV_PGN = Path(__file__).with_name("kasparov_topalov_1999.pgn")
EXPECTED_PLIES = 87
EXPECTED_RESULT = "1-0"
EXPECTED_LAST_SAN = "Qa7"
EXPECTED_FINAL_FEN = "8/Q6p/6p1/5p2/5P2/2p3P1/3r3P/2K1k3 b - - 3 44"


def load_kasparov_topalov() -> tuple[list[str], list[str], list[str]]:
    """Return (UCI moves, SAN moves, FENs) after validating the complete PGN.

    The FEN sequence includes the initial board, so a complete 87-ply game
    produces 88 positions.  Parsing with python-chess rejects illegal moves.
    """
    with KASPAROV_TOPALOV_PGN.open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
    if game is None:
        raise ValueError(f"No PGN game found in {KASPAROV_TOPALOV_PGN}")
    if game.errors:
        raise ValueError(f"PGN parse errors: {game.errors}")
    if game.headers.get("PlyCount") != str(EXPECTED_PLIES):
        raise ValueError(f"Expected PlyCount {EXPECTED_PLIES}; got {game.headers.get('PlyCount')!r}")
    if game.headers.get("Result") != EXPECTED_RESULT:
        raise ValueError(f"Expected result {EXPECTED_RESULT}; got {game.headers.get('Result')!r}")

    board = game.board()
    fens = [board.fen()]
    uci_moves: list[str] = []
    san_moves: list[str] = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {move.uci()}")
        san_moves.append(board.san(move))
        uci_moves.append(move.uci())
        board.push(move)
        fens.append(board.fen())

    if len(uci_moves) != EXPECTED_PLIES:
        raise ValueError(f"Expected {EXPECTED_PLIES} plies; got {len(uci_moves)}")
    if san_moves[-1] != EXPECTED_LAST_SAN:
        raise ValueError(f"Expected final SAN {EXPECTED_LAST_SAN}; got {san_moves[-1]}")
    if len(fens) != EXPECTED_PLIES + 1:
        raise AssertionError("FEN count must equal plies + initial position")
    if fens[-1] != EXPECTED_FINAL_FEN:
        raise ValueError(f"Unexpected final position: {fens[-1]}")
    return uci_moves, san_moves, fens


def main() -> None:
    uci, san, fens = load_kasparov_topalov()
    with KASPAROV_TOPALOV_PGN.open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
    assert game is not None
    print(f"PGN: {KASPAROV_TOPALOV_PGN}")
    print(f"plies={len(uci)} positions={len(fens)}")
    print(f"final SAN={len(san) // 2 + 1}.{san[-1]} result={game.headers['Result']}")
    print(f"final FEN={fens[-1]}")


if __name__ == "__main__":
    main()
