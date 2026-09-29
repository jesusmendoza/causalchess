# Evaluación de embeddings — catálogo de tests

Los tests se dividen en **dos familias** (paper §4):

- **Concept probes (§A)**: linear / MLP classifiers sobre embedding congelado. Pregunta: *¿está el concepto X decodificable en el embedding?* Tradición BERT/SimCLR. Estándar en la literatura.
- **Chess representation diagnostics (§B a §H)**: tests chess-domain de comportamiento y geometría de la representación. Pregunta: *¿cómo está organizada la representación y cómo se comporta en tareas chess-nativas?* Contribución metodológica del paper.

Todos los tests aceptan `--ckpt <ruta.pt>` y están diseñados para correr contra los 5 modelos del paper (+ untrained como control interno):

1. **Untrained CNN** (random init, control interno dentro de cada test)
2. **Autoencoder** (`models/ae_cnn.pt`) — baseline de reconstrucción
3. **SimCLR contrastivo (color-flip)** (`models/simclr_cnn_final.pt`) — baseline contrastivo sobre simetrías estructurales
4. **SimCLR temporal (same-game pairs)** (`models/simclr_temporal_cnn.pt`) — baseline contrastivo opcional (código listo, no entrenado aún)
5. **PMP-mem** (`models/prev_move_cnn_v2.pt`) — nuestro modelo principal
6. **PMP-no-mem** (`models/prev_move_cnn_v2_dedup.pt`) — modelo sin memoria de aperturas

**Nota:** NextMove CNN fue **descartado como baseline entrenado** — next-move prediction ahora es un **TEST** (B1: frozen embedding + linear head) que se aplica a TODOS los modelos. La comparación contra Maia se hace vía literatura, no re-entrenamiento. Esta decisión ahorró 26h de GPU y evitó reinventar Maia-with-asterisks.

---

## A. Sondeos — ¿qué concepto está codificado en el vector?

### A1. `test_probes.py` — Tier 1 linear, 9 conceptos básicos
**Qué mide:** clasificador lineal sobre el vector predice fase, enroque, turno, jaque, balance material, etc. Si funciona, el concepto está linealmente separable.
**Output clave:** accuracy por concepto + baseline untrained.
**Datos:** `lichess_data/prev_move_2400.tsv` (20K posiciones).
**Tiempo:** ~1 min GPU.

### A2. `test_probes_strategic.py` — Tier 1, 24 conceptos, por fase
**Qué mide:** mismos conceptos linealmente + 15 estratégicos (desarrollo, control del centro, espacio, escudo del rey, pareja de alfiles, peones pasados/doblados, movilidad). Desagregado por fase (apertura/medio/final).
**Output clave:** tabla de lift `trained - untrained` por concepto y por fase.
**Datos:** mismo corpus.
**Tiempo:** ~3 min GPU.

### A3. `test_probes_mlp.py` — Tier 2 no-lineal
**Qué mide:** red pequeña de 2 capas sobre el vector. Si supera al lineal significa que la información está pero codificada de forma no-lineal.
**Output clave:** tabla con R² / accuracy lineal vs MLP por concepto.
**Datos:** mismo corpus.
**Tiempo:** ~5 min GPU.

---

## B. Transferencia — ¿el vector sirve para otra tarea?

### B1. `test_nextmove.py` — transfer lineal
**Qué mide:** congela el vector, entrena clasificador lineal para predecir el movimiento **siguiente** (tarea "opuesta" al pretraining de prev_move). Lift sobre untrained mide la transferencia.
**Output clave:** accuracy top-1 y top-5 vs baselines.
**Datos:** `lichess_data/lichess_2013_01.pgn` (3K partidas).
**Tiempo:** ~5 min GPU (incluye extracción de PGN).

### B2. `test_finetune.py` — Tier 3 fine-tune end-to-end
**Qué mide:** descongela TODO el encoder y lo re-entrena sobre next-move. Compara init pretrained vs init aleatorio. Si pretrained gana, es un **init útil** para tareas downstream (equivalente a ImageNet pretraining).
**Output clave:** val_acc al final del fine-tune para cada init.
**Datos:** `lichess_data/lichess_2013_01.pgn`.
**Tiempo:** ~2-4h GPU por corrida (40K pares × 8 epochs).
**Nota:** el más caro — correrlo solo al final.

---

## C. Geometría — ¿cómo se ve el espacio?

### C1. `test_landscape.py` — proyecciones 2D
**Qué mide:** PCA + t-SNE + UMAP sobre 5K posiciones, coloreadas por fase de juego.
**Output clave:** imagen `paper/figures/test_landscape.png`.
**Tiempo:** ~1 min GPU + tSNE CPU (~5 min).

### C2. `test_trajectories.py` — suavidad de partidas
**Qué mide:** traza partidas completas (Kasparov-Topalov, Scholar's Mate, Morphy's Opera Game) como caminos en el espacio. Mide distancia consecutiva (suavidad) y distancia al inicio (progresión).
**Output clave:** tabla de estadísticas por partida + figura.
**Tiempo:** ~10 seg GPU.

### C3. `eval_neighborhoods.py` — vecinos cercanos a anclas
**Qué mide:** dadas 16 anclas (Najdorf, K+P endgame, Ruy Lopez, mate-in-3, etc.) devuelve los 5 vecinos más cercanos en el corpus.
**Output clave:** listado cualitativo — los vecinos deberían ser estratégicamente parecidos.
**Tiempo:** ~1 min GPU.

### C4. `test_latent_walk.py` — caminata aleatoria
**Qué mide:** partiendo de un ancla, perturba el vector con magnitudes ε∈{0.5, 1, 2, 5, 10, 20} en direcciones aleatorias, busca la posición más cercana en el corpus. Muestra el "radio" de cada concepto.
**Output clave:** ε vs posición encontrada (detecta saturación de corpus).
**Tiempo:** ~1 min GPU.

---

## D. Aritmética — ¿las posiciones se suman?

### D1. `eval_arithmetic_compare.py` — álgebra entrenado vs sin entrenar
**Qué mide:** tres pruebas: puntos medios (`Siciliana + QG = Inglesa?`), simetría de color, aritmética de piezas. Compara contra untrained.
**Output clave:** cualitativo — posiciones encontradas para cada suma/media.
**Tiempo:** ~2 min GPU.

### D2. `test_algebra_by_tier.py` — álgebra por frecuencia
**Qué mide:** la misma aritmética, pero segmentada por cuántas veces aparece cada posición en el entrenamiento (Tier 1: 100+, Tier 4: única). Si funciona solo en Tier 1 es memorización; si funciona en Tier 4 es estructura real.
**Output clave:** mean cosine + self-match rate por tier.
**Datos:** requiere `data/chess_x.bin` (para contar frecuencias).
**Tiempo:** ~3 min GPU.

---

## E. Estilo — ¿reconoce al jugador?

### E1. `test_style.py` — clasificación de 4 GMs
**Qué mide:** dada una posición, ¿quién la jugó? Probe lineal sobre 4 GMs (Carlsen, Tang, Sarin, Blübaum). 25% es azar.
**Output clave:** accuracy + matriz de confusión.
**Datos:** `data_style/gm_games.pgn` (GM games).
**Tiempo:** ~1 min GPU.

### E2. `test_style_endgame.py` — estilo solo en finales
**Qué mide:** diagnóstico con posiciones de finales balanceados (pieces<=20, mate_diff<=2). La partición por posición comparte partidas entre entrenamiento y validación.
**Output clave:** su exactitud no demuestra generalización a partidas nuevas. Para la evaluación válida, usar `test_style_game_split.py` (cinco particiones por partida; exactitud balanceada).
**Tiempo:** ~1 min GPU.

### E3. `test_style_game_split.py` — evaluación por partida
**Qué mide:** identificación del jugador en partidas completamente nuevas, con cinco particiones 80/20 por partida y exactitud balanceada.
**Resultado:** PMP 28.74 ± 3.24% frente a CNN sin entrenar 28.91 ± 2.24%; no se sostiene la conclusión de estilo.

---

## H. Validación con ground truth externo — Elo prediction

### H1. `test_elo_predictor.py` — predicción de Elo desde 40 posiciones
**Qué mide:** el embedding captura la **fuerza del jugador**. Para cada jugador, promedia embeddings de 40 de sus posiciones → entrena Ridge para predecir su Elo real de Lichess. Validación con ground truth externo (no más probes, no más proxies).
**Métrica:** R² y MAE en Elo.
**Datos:** `data_elo/chess_x.bin + chess_player_id.bin + chess_elo.bin` (producido por `extract_elo_dataset.py`).
**Baselines comparados internamente:** material-only (6-d), untrained embedding, material + trained embedding.
**Uso:**
```bash
python3 evaluation/test_elo_predictor.py --ckpt models/prev_move_cnn_v2.pt
```
**Tiempo:** ~2 min GPU por modelo (embedding) + 30 seg CPU (Ridge).
**Resultado paper (§4.8):** Mem R²=0.52, MAE 194 Elo; AE R²=0.49; SimCLR R²=0.35. Material solo R²=0.06 (casi inútil — Elo no depende de material).

---

## G. Ensemble — ¿los pretrainings son complementarios?

### G1. `test_ensemble.py` — concatenación de N embeddings
**Qué mide:** corre probes, style, next-move transfer y strategic distance sobre la **concatenación** de embeddings de N modelos (ej. AE + SimCLR + Mem = 768-d). Compara vs cada modelo individual.
**Pregunta:** ¿un ensemble concatenado bate al mejor modelo individual, o son redundantes?
**Output:** tabla donde cada columna es un modelo + columna ENSEMBLE.
**Uso:**
```bash
python3 evaluation/test_ensemble.py \\
  --ckpts models/ae_cnn.pt models/simclr_cnn.pt models/prev_move_cnn_v2.pt \\
  --labels AE SimCLR Mem --n 20000
```
**Tiempo:** ~5 min GPU para 3 modelos, 15K posiciones.

---

## F. Distancia estratégica — ¿el espacio refleja fuerza?

### F1. `test_strategic_distance.py` — correlación con Stockfish
**Qué mide:** para N pares aleatorios (A, B), calcula:
- `d_emb = ||emb(A) - emb(B)||`
- `d_eval = |eval_SF(A) - eval_SF(B)|` (centipeones)
- Correlación Pearson + Spearman.

Si alta: posiciones "estratégicamente distantes" (evals muy diferentes) también están distantes en el embedding. Fundamental para usar el embedding en búsqueda (FMC, alfa-beta).

**Output clave:** correlación trained, untrained, Hamming-sanity.
**Datos:** requiere `lichess_data/stockfish_evals.tsv` — producido por `compute_stockfish_evals.py` (ver abajo).
**Tiempo:** ~1 min GPU (después de tener los evals).

### Dependencia: `compute_stockfish_evals.py`
Script de un solo uso: toma N FENs del corpus, los evalúa con Stockfish a profundidad fija, guarda TSV `(fen, eval_cp)`.
```bash
python3 compute_stockfish_evals.py --n 10000 --depth 12 \
    --stockfish /usr/games/stockfish \
    --output lichess_data/stockfish_evals.tsv
```
**Tiempo:** ~10-15 min CPU (Stockfish a profundidad 12, ~50-100ms/pos).

---

## Orquestación

### `run_all.sh <ckpt> <outdir>`
Corre A1, A2, A3, C1, C2, C3, C4, D1, D2, E1, E2, B1 en secuencia. Guarda logs en `<outdir>/`.
Usa GPU si está disponible.
**NO incluye:** B2 (fine-tune), F1 (strategic distance). Esos se corren aparte.

### `run_sealed_evaluation.sh <ckpt> <outdir>`
Corre un subconjunto sobre el validation set sellado (Dec 2024). SOLO CORRER UNA VEZ AL FINAL del paper.

---

## Pipeline recomendado (ejecución al final)

1. **Tener todos los checkpoints:** ae_cnn.pt, simclr_cnn.pt, next_move_cnn.pt, prev_move_cnn_v2.pt, prev_move_cnn_v2_dedup.pt.
2. **Generar evals Stockfish una vez:** `compute_stockfish_evals.py --n 10000` (15 min CPU).
3. **Correr la batería contra los 5 modelos:**
   ```bash
   for m in ae_cnn simclr_cnn next_move_cnn prev_move_cnn_v2 prev_move_cnn_v2_dedup; do
       bash evaluation/run_all.sh models/$m.pt evaluation/results_$m
       python3 evaluation/test_strategic_distance.py --ckpt models/$m.pt \
           > evaluation/results_$m/strategic_distance.log 2>&1
   done
   ```
   Tiempo total: ~2h (todo en GPU).
4. **Tier 3 fine-tune** (el más caro) — solo sobre los 2 PMP y baselines seleccionados:
   ```bash
   python3 evaluation/test_finetune.py --ckpt models/prev_move_cnn_v2.pt \
       --label mem > evaluation/results_prev_move_cnn_v2/finetune.log 2>&1
   # idem para no-mem, AE, SimCLR (opcional)
   ```
   Tiempo: ~3-4h por modelo.
5. **Sealed eval** (una sola vez, al final):
   ```bash
   bash evaluation/run_sealed_evaluation.sh models/prev_move_cnn_v2.pt \
       evaluation/results_sealed_mem
   bash evaluation/run_sealed_evaluation.sh models/prev_move_cnn_v2_dedup.pt \
       evaluation/results_sealed_nomem
   ```

---

## Tabla final del paper (7 modelos × 14 tests = 98 celdas)

| Test | Untrained | AE | SimCLR-flip | SimCLR-temp | NextMove | Mem | No-mem |
|---|---|---|---|---|---|---|---|
| A1 probes básicos | | | | | | | |
| A2 probes estratégicos | | | | | | | |
| A3 probes MLP | | | | | | | |
| B1 transfer next-move | | | | | | | |
| B2 fine-tune Tier 3 | | | | | | | |
| C1 landscape | | | | | | | |
| C2 trajectories | | | | | | | |
| C3 neighborhoods | | | | | | | |
| C4 latent walk | | | | | | | |
| D1 arithmetic | | | | | | | |
| D2 algebra by tier | | | | | | | |
| E1 style (all phases) | | | | | | | |
| E2 style endgame | | | | | | | |
| F1 strategic distance | | | | | | | |

---

Última actualización: 2026-04-19
