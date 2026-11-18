# Next-move transfer: complete-vocabulary R2 results

All scores are held-out test accuracy (%), mean ± population standard deviation over three paired game splits/training seeds. Each test partition contains 10,000 positions. The corpus and pretrained checkpoints are reused across seeds.

## Frozen representations

| Condition | Top-1 | Top-5 |
|---|---:|---:|
| Uniform | 0.05 ± 0.00 | 0.26 ± 0.00 |
| Training frequency | 1.03 ± 0.09 | 4.24 ± 0.15 |
| Scratch | 4.19 ± 0.04 | 11.88 ± 0.16 |
| PMP-mem | 8.82 ± 0.35 | 24.45 ± 0.47 |
| PMP-no-mem | 9.00 ± 0.25 | 24.61 ± 0.29 |
| AE | 6.17 ± 0.16 | 18.72 ± 0.10 |
| SimCLR | 1.08 ± 0.05 | 4.32 ± 0.25 |
| Ensemble | 7.97 ± 0.16 | 21.99 ± 0.20 |

## End-to-end fine-tuning

| Condition | Top-1 | Top-5 |
|---|---:|---:|
| Uniform | 0.05 ± 0.00 | 0.26 ± 0.00 |
| Training frequency | 1.03 ± 0.09 | 4.24 ± 0.15 |
| Scratch | 2.97 ± 0.59 | 8.20 ± 1.28 |
| PMP-mem | 5.74 ± 0.37 | 14.45 ± 0.53 |
| PMP-no-mem | 5.78 ± 0.11 | 14.65 ± 0.21 |

## Paired top-1 differences from scratch

| Comparison | Mean difference (pp) | SD (pp) |
|---|---:|---:|
| linear:PMP-mem-Scratch | +4.63 | 0.38 |
| linear:PMP-no-mem-Scratch | +4.81 | 0.28 |
| finetune:PMP-mem-Scratch | +2.77 | 0.47 |
| finetune:PMP-no-mem-Scratch | +2.81 | 0.70 |

No model was selected using test accuracy. Fine-tuning selected each checkpoint by validation accuracy; all eight epochs and selected epoch numbers are recorded.
