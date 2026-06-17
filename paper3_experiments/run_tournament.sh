#!/bin/bash
./repo_cchess/c-chess-cli \
    -engine cmd=stockfish name=Baseline option.EvalFile=baseline.nnue \
    -engine cmd=stockfish name=NNUECC option.EvalFile=nnuecc.nnue \
    -each tc=2+0.02 -rounds 100 -concurrency 6 \
    -pgn match_results.pgn
