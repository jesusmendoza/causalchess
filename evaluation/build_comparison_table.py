#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
build_comparison_table.py — auto-generate the 5×N comparison table from
per-model evaluation logs in evaluation/results_*/.

Scans each results_* directory for known metrics and emits a Markdown
table ready to paste into the paper. Reruns idempotently as new models
are evaluated.

Usage:
    python3 evaluation/build_comparison_table.py
    python3 evaluation/build_comparison_table.py --output paper/tables/comparison.md
"""
import argparse
import os
import re


# Model label → results directory name
MODELS = [
    ("AE",              "results_ae"),
    ("SimCLR-flip",     "results_simclr_final"),
    ("Mem (40.4%)",     "results_mem_at40"),
    ("No-mem (e1 24.8%)", "results_nomem_e1"),
    # Add more as they become available:
    # ("No-mem (final)",  "results_nomem"),
    # ("SimCLR-temp",     "results_simclr_temporal"),
]

RESULTS_ROOT = "evaluation"


def slurp(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def extract_probes(log_text):
    """From probes.log, extract 9 basic concept scores."""
    out = {}
    patterns = [
        ("phase", r"phase \(0/1/2\)\s+([\d.]+)%"),
        ("castled_W", r"white castled\s+([\d.]+)%"),
        ("castled_B", r"black castled\s+([\d.]+)%"),
        ("turn", r"turn \(white/black\)\s+([\d.]+)%"),
        ("in_check", r"in check\s+([\d.]+)%"),
        ("material_bal_R2", r"material balance \(W-B\)\s+R²=([+\-\d.]+)"),
        ("piece_count_R2", r"piece count \(2-32\)\s+R²=([+\-\d.]+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, log_text)
        out[key] = m.group(1) if m else "—"
    return out


def extract_nextmove(log_text):
    out = {}
    m = re.search(r"Trained CNN \+ linear head:\s+val_acc = ([\d.]+)%", log_text)
    out["nextmove_top1"] = m.group(1) + "%" if m else "—"
    top5 = re.findall(r"Top-5 accuracy:.*?Trained:\s+([\d.]+)%", log_text, re.DOTALL)
    out["nextmove_top5"] = top5[0] + "%" if top5 else "—"
    return out


def extract_style(log_text, is_endgame=False):
    out = {}
    m = re.search(r"Trained:\s+([\d.]+)%", log_text)
    out["style_all" if not is_endgame else "style_endgame"] = m.group(1) + "%" if m else "—"
    return out


def extract_algebra_tier(log_text):
    out = {}
    for tier_name, pat in [
        ("algebra_T1", r"Tier 1 \(100\+\)\s+[\d.]+\s+[\d.]+%\s+([\d.]+)%"),
        ("algebra_T4", r"Tier 4 \(unique\)\s+[\d.]+\s+[\d.]+%\s+([\d.]+)%"),
    ]:
        m = re.search(pat, log_text)
        out[tier_name] = m.group(1) + "%" if m else "—"
    return out


def extract_F1b(log_text):
    out = {}
    m = re.search(r"Trained embedding \([^)]+\)\s+train R²=[+\-\d.]+\s+val R²=([+\-\d.]+)", log_text)
    out["F1b_emb_R2"] = m.group(1) if m else "—"
    m2 = re.search(r"Material \+ trained embedding \(262-d\)\s+train R²=[+\-\d.]+\s+val R²=([+\-\d.]+)", log_text)
    out["F1b_mat_emb_R2"] = m2.group(1) if m2 else "—"
    return out


def extract_elo(log_text):
    out = {}
    m = re.search(r"Trained embedding mean.*?R²=([+\-\d.]+)\s+MAE=([\d.]+) Elo", log_text)
    out["elo_R2"] = m.group(1) if m else "—"
    out["elo_MAE"] = m.group(2) if m else "—"
    return out


def collect_model(label, results_dir):
    """Gather all metrics for one model from its results directory."""
    base = os.path.join(RESULTS_ROOT, results_dir)
    metrics = {"label": label}
    metrics.update(extract_probes(slurp(os.path.join(base, "probes.log"))))
    metrics.update(extract_nextmove(slurp(os.path.join(base, "nextmove.log"))))
    metrics.update(extract_style(slurp(os.path.join(base, "style.log")), False))
    metrics.update(extract_style(slurp(os.path.join(base, "style_endgame.log")), True))
    metrics.update(extract_algebra_tier(slurp(os.path.join(base, "algebra_by_tier.log"))))
    # F1b and Elo live in separate dirs
    ckpt_stem = {
        "AE": "ae_cnn",
        "SimCLR-flip": "simclr_cnn_final",
        "Mem (40.4%)": "prev_move_cnn_v2_at40",
        "No-mem (e1 24.8%)": "prev_move_cnn_v2_dedup_e1",
    }.get(label)
    if ckpt_stem:
        metrics.update(extract_F1b(slurp(
            os.path.join(RESULTS_ROOT, "results_F1b", f"{ckpt_stem}.log"))))
        metrics.update(extract_elo(slurp(
            os.path.join(RESULTS_ROOT, "results_elo", f"{ckpt_stem}.log"))))
    return metrics


ROW_ORDER = [
    ("A1 Phase probe",           "phase"),
    ("A1 Castled W",             "castled_W"),
    ("A1 Castled B",             "castled_B"),
    ("A1 Turn",                  "turn"),
    ("A1 In-check",              "in_check"),
    ("A1 Material balance R²",   "material_bal_R2"),
    ("A1 Piece count R²",        "piece_count_R2"),
    ("B1 Next-move top-1",       "nextmove_top1"),
    ("B1 Next-move top-5",       "nextmove_top5"),
    ("E1 Style (all phases)",    "style_all"),
    ("E2 Style (endgame only)",  "style_endgame"),
    ("D2 Algebra Tier 1 self-match", "algebra_T1"),
    ("D2 Algebra Tier 4 self-match", "algebra_T4"),
    ("F1b Embedding-only R²",    "F1b_emb_R2"),
    ("F1b Material+embed R²",    "F1b_mat_emb_R2"),
    ("H1 Elo R²",                "elo_R2"),
    ("H1 Elo MAE (Elo)",         "elo_MAE"),
]


def emit_markdown(results):
    headers = ["Metric"] + [r["label"] for r in results]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric_name, key in ROW_ORDER:
        row = [metric_name]
        for r in results:
            val = r.get(key, "—")
            row.append(val)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="paper/tables/comparison.md")
    args = ap.parse_args()

    results = [collect_model(label, rdir) for label, rdir in MODELS]

    md = emit_markdown(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Comparison table (auto-generated)\n\n")
        f.write(md)
        f.write("\n\n---\n_Regenerate with_: `python3 evaluation/build_comparison_table.py`\n")
    print(f"Wrote {args.output}")
    print()
    print(md)


if __name__ == "__main__":
    main()
