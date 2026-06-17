# faichess_nn — Ideas y direcciones futuras

## Estado actual del proyecto (2026-06)

Este repositorio contiene la implementación y los experimentos de **CausalChess** — un framework de representation learning para ajedrez basado en la tarea de predecir el movimiento previo.

El paper principal (Paper 1) tiene los experimentos completos y está en fase de pulido
pre-envío a **IEEE Conference on Games (CoG)** (ver checklist en `PENDING.md`). Los modelos entrenados son:
- **CausalChess-mem** (`models/prev_move_cnn_v2.pt`) — 25.3M pares, 40.4% val_acc
- **CausalChess-no-mem** (`models/prev_move_cnn_v2_dedup.pt`) — 22.6M pares deduplicados, 35.0% val_acc
- **Autoencoder** (`models/ae_cnn.pt`) — baseline de reconstrucción
- **SimCLR-flip** (`models/simclr_cnn_final.pt`) — baseline contrastivo

---

## Ideas para Paper 1 (pendientes de ejecutar)

### Ablation: class-balanced loss
Entrenar CausalChess con pesos inversamente proporcionales a la frecuencia de cada clase. Hipótesis: val_acc baja (menos "freebies" por frecuencia), pero la calidad del embedding sube. Esto separa "memorización de movimientos frecuentes" de "aprendizaje estructural genuino".

### Más GMs para el test de estilo
Actualmente 4 GMs (Carlsen, Tang, Sarin, msb2). Escalar a 10-20 con más datos de Lichess. La clave es tener GMs con estilos bien diferenciados (Tal vs Karpov vs Fischer vs Kasparov).

### Test cross-era
Evaluar estilo con datos de eras distintas para verificar que el embedding captura el estilo del jugador y no el de la época (1990s vs 2020s).

---

## Ideas para Papers 2-3

### Learned distance para FMC (Paper 2)
Usar la distancia entre embeddings CausalChess como función de diversidad en la búsqueda FMC. La hipótesis: posiciones estratégicamente distintas tienen embeddings lejanos → la distancia en el espacio de embedding es una mejor métrica de diversidad que el popcount de XOR de bitboards.

### NNUECC + LCC (Paper 3)
Augmentar Stockfish NNUE y Leela con el embedding CausalChess como input auxiliar (zero-init). Diseño completo en `paper/paper3/`.

---

## Fuera de scope de este proyecto

- Supervised learning desde Stockfish hacia evaluación posicional — investigación separada.
- Entrenamiento de engine propio desde cero (Misha 2.0) — track de engineering paralelo.
- Redes binarias para ajedrez — proyecto BitFlow, independiente.

---

*Última actualización: 2026-06-16*
