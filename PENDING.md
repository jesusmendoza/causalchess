# CausalChess — Trabajo Pendiente

Estado honesto al 2026-06-16.

Venue objetivo: **IEEE Transactions on Games (ToG)** — revista IEEE, sin presentación.
Single-blind (autores con nombre), formato journal dos columnas, full paper ≤ 10 págs.
(Se descartó IEEE CoG porque requiere presentación presencial.)

---

## Modelos disponibles

| Modelo | Archivo | Val_acc | Estado |
|--------|---------|---------|--------|
| CausalChess-mem | `models/prev_move_cnn_v2.pt` | 40.4% | ✅ Final |
| CausalChess-no-mem | `models/prev_move_cnn_v2_dedup.pt` | 35.0% | ✅ Final |
| Autoencoder | `models/ae_cnn.pt` | — | ✅ Final |
| SimCLR-flip | `models/simclr_cnn_final.pt` | — | ✅ Final |

Dataset de entrenamiento: Lichess standard rated **junio–agosto 2023** (3 meses, ambos
jugadores ≥ 2400 Elo, desde ply 4), **25,294,743** pares (position, prev_move), 1,928 clases.
Fuente: `lichess_data/prev_move_2400_big.tsv` (coincide exacto con `data/chess_meta.txt`).
Validación OOD sellada sobre Dic-2024.

---

## Suite de evaluación

Scripts en `evaluation/`:

| # | Script | Descripción | Estado |
|---|--------|-------------|--------|
| 1 | `test_probes.py` | 9 conceptos básicos (fase, enroque, material) | ✅ mem, ✅ no-mem, ✅ AE, ✅ SimCLR |
| 2 | `test_probes_strategic.py` | 24 conceptos estratégicos + desglose por fase | ✅ mem, ✅ no-mem |
| 3 | `test_probes_mlp.py` | Tier 2: probes MLP no-lineales | ✅ mem |
| 4 | `test_trajectories.py` | Suavidad de trayectorias en partida | ✅ mem, ✅ no-mem, ✅ AE, ✅ SimCLR |
| 5 | `test_landscape.py` | Proyecciones 2D (PCA, t-SNE) | ✅ figuras generadas |
| 6 | `eval_arithmetic_compare.py` | Chess algebra (midpoint, color-symmetry) | ✅ mem |
| 7 | `eval_neighborhoods.py` | Vecinos más cercanos en el corpus | ✅ mem |
| 8 | `test_nextmove.py` | Transfer a predicción de próximo movimiento | ✅ mem, ✅ no-mem |
| 9 | `test_style.py` | Detección de estilo 4 GMs | ✅ mem, ✅ no-mem |
| 10 | `test_style_endgame.py` | Estilo solo en finales (sin sesgo de apertura) | ✅ mem, ✅ no-mem |
| 11 | `test_algebra_by_tier.py` | Algebra por tier de unicidad FEN | ✅ mem |
| 12 | `test_elo_predictor.py` | Predicción de Elo con datos reales | ✅ mem, ✅ no-mem, ✅ AE, ✅ SimCLR |
| 13 | `test_ensemble.py` | Ensemble AE + SimCLR + Mem (768-d) | ✅ |
| 14 | `test_ensemble_ablation.py` | Leave-one-out ablation del ensemble | ✅ |
| 15 | `test_finetune.py` | Tier 3: fine-tune end-to-end | ✅ mem, no-mem, AE |

---

## Checklist pre-envío — IEEE CoG (Paper 1)

Hallazgos de la revisión 2026-06-16. Orden por prioridad.

### Bloqueantes (impiden el envío)

- [ ] **B1 — Párrafos en español dentro del `.tex` EN.** §4.7 (Complementarity), §4.9 (Elo)
  y parte de trayectorias están sin traducir. Traducir todo al inglés.
- [ ] **B2 — Artefactos Markdown `**...**` en el LaTeX.** No producen negrita; salen como
  asteriscos literales. Reemplazar por `\textbf{...}` (líneas ~92, 205, 225, 297, 331).
- [ ] **B3 — Error aritmético "16×".** 40.4 / 14.9 ≈ 2.7×, no 16×. Corregir el multiplicador
  o el baseline citado, y mantener coherencia abstract ↔ §4.2.
- [x] **B4 — Procedencia del dataset.** Resuelto: era "single-month `lichess_2013_01.pgn`"
  (un piloto de 628K pares, imposible). El real es **junio–agosto 2023** (≥2400 Elo, desde
  ply 4) = 25,294,743 pares. Corregido en el método del `.tex`.

### Consistencia numérica

- [ ] **C1 — Estilo de juego.** Abstract dice 51.7% (all-phase); §Style dice 54.4% (endgame).
  Elegir cifra y declarar la condición explícitamente.
- [ ] **C2 — Comparación con Maia (manzanas/peras).** 14.8% es top-500 clases; Maia 46–52% es
  vocabulario completo. Añadir disclaimer o igualar el vocabulario.

### Rigor / reproducibilidad

- [ ] **R1 — Varianza / semillas.** Resultados son corridas únicas. Añadir CI/desv. estándar
  en métricas clave (mem vs no-mem) o nota honesta de limitación.
- [ ] **R2 — Reproducibilidad.** Falta `README` + `requirements.txt` + declaración de
  disponibilidad de código/modelos.
- [ ] **R3 — `references.bib`.** Varias entradas con `and others`; completar listas de autores.

### Venue / formato

- [x] **V1 — Plantilla y anonimato.** Plantilla journal (`\documentclass[journal]{IEEEtran}`);
  ToG es single-blind → autor/afiliación restaurados. Full paper ≤ 10 págs (vamos en ~7).
- [x] **V2 — Encuadre.** Reencuadrado como contribución representacional ("comprensión más
  humana", no mayor fuerza de juego) en abstract y conclusión.
- [x] **V3 — Cover letter.** Redactada en `paper/cover_letter.md` (falta fecha + URL del repo).
- [ ] **V4 — URL de código.** Insertar el repo público/DOI en la frase de disponibilidad (single-blind permite enlace real).

---

## Pendiente — Paper 3 (NNUECC / LCC)

Diseño en `paper/paper3/`. **Código: iniciado** (ver `paper3_experiments/`, ~23 scripts).

- [x] Fork Stockfish NNUE (`nnue-pytorch`) + implementar NNUECC (input aux. zero-init)
- [x] Entrenar NNUECC desde checkpoint y matchear vs baseline (torneos)
- [ ] Resultado aún **no favorable** (depth-6: baseline 55 – NNUECC 45) — investigar
- [ ] Fork lczero-training + implementar LCC
- [ ] Limpiar placeholders (p. ej. `outcome = 0.5` en `prepare_nnuecc_dataset_tsv.py`)

---

## Notas de convención

- **CausalChess-mem** — entrenado en 25.3M binomios (duplicados incluidos)
- **CausalChess-no-mem** — entrenado en 22.6M binomios deduplicados
- Ambos modelos son independientes — no es validador/validado

---

*Última actualización: 2026-06-16*
