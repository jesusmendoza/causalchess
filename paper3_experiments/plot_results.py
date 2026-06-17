import matplotlib.pyplot as plt
import numpy as np

with open("history_baseline.txt") as f:
    baseline = [float(x) for x in f.readlines()]
    
with open("history_nnuecc.txt") as f:
    nnuecc = [float(x) for x in f.readlines()]
    
epochs = np.arange(1, 11)

plt.figure(figsize=(10, 6))
plt.plot(epochs, baseline, 'o-', color='tab:red', label='Stockfish NNUE (Baseline)', linewidth=2.5)
plt.plot(epochs, nnuecc, 's-', color='tab:blue', label='Stockfish NNUE-PMP (PMP)', linewidth=2.5)

plt.title("Mini-Scale Fine-Tuning: NNUE vs NNUE-PMP (50k posiciones)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss (Training)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.annotate('NNUE-PMP converge más rápido\nen las primeras 5 epochs', 
             xy=(2, 0.0104), xytext=(3, 0.014),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=11)

plt.tight_layout()
plt.savefig("/home/armando/.gemini/antigravity/brain/b72e7601-d3ad-4ed7-9599-932ccc4a9b47/training_curves.png", dpi=300)
print("Plot saved.")
