# Comparison table (auto-generated)

| Metric | AE | SimCLR-flip | Mem (40.4%) | No-mem (e1 24.8%) |
|---|---|---|---|---|
| A1 Phase probe | 91.8 | 65.5 | 90.0 | 86.5 |
| A1 Castled W | 98.4 | 70.3 | 93.8 | 85.8 |
| A1 Castled B | 98.5 | 68.4 | 94.8 | 88.7 |
| A1 Turn | 53.8 | 50.0 | 63.5 | 60.9 |
| A1 In-check | 91.9 | 92.1 | 95.3 | 94.4 |
| A1 Material balance R² | +0.095 | +0.016 | +0.215 | +0.084 |
| A1 Piece count R² | +0.953 | +0.371 | +0.922 | +0.905 |
| B1 Next-move top-1 | 9.83% | 2.61% | 14.80% | 12.72% |
| B1 Next-move top-5 | 27.06% | 8.92% | 37.23% | 32.81% |
| E1 Style (all phases) | 55.74% | 42.47% | 51.67% | 46.91% |
| E2 Style (endgame only) | 59.23% | 46.38% | 54.36% | 50.37% |
| D2 Algebra Tier 1 self-match | 27.0% | 65.6% | 58.8% | 50.6% |
| D2 Algebra Tier 4 self-match | 85.0% | 97.6% | 95.6% | 92.4% |
| F1b Embedding-only R² | +0.2238 | +0.1395 | +0.2506 | +0.1525 |
| F1b Material+embed R² | +0.6185 | +0.6045 | +0.6170 | +0.6089 |
| H1 Elo R² | +0.4920 | +0.3505 | +0.5192 | +0.4265 |
| H1 Elo MAE (Elo) | 198.8 | 224.1 | 193.7 | 211.0 |

---
_Regenerate with_: `python3 evaluation/build_comparison_table.py`
