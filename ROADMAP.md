# CausalChess + Misha + Style-Synth — Full Megaproject Roadmap

**Authored:** 2026-04-17 (post honest reality-check)
**Owner:** Jesús Armando Mendoza Ramos / Path-Data / CreAI
**Horizon:** 12-18 months for complete vision

---

## Megaproject summary

Three interlocking tracks:

- **Research track (Papers):** CausalChess representation, FMC distance, Augmented engines
- **Engineering track (Engine):** own chess engine (Misha 2.0), 100% non-derivative
- **Product track (Commercial):** Style-Synth "PlayerScope" bot, chess training market

Each informs the others. Sequencing matters because the engineering track (own engine) is the bottleneck for both Paper 2/3 clean experiments AND the product.

---

## Re-sequenced as of 2026-04-17 (Armando's insight)

Original plan had Misha 2.0 as prerequisite for Papers 2-3. Corrected:
- **Papers 1-3** can be published with academic forks of SF/Lc0 (GPL acceptable for research).
- **Misha 2.0** is only required for the commercial product (Style-Synth), NOT for any paper.
- **Misha 2.0 is a parallel engineering track** — doesn't block papers.

This means LCC/NNUECC experiments (Paper 3) can happen EARLY, producing concrete Elo gains on top of existing strong engines. This is where the biggest PR impact lies — "we improved Leela/Stockfish with a new pretraining trick" is a better headline than anything CausalChess-alone can achieve.

Revised phase ordering:

```
Phase 0:  Paper 1 (CausalChess standalone)       — 1-2 months
Phase 1:  Papers 2-3 on SF/Lc0 forks             — 2-4 months
          (LCC, NNUECC experiments, FMC+distance)
Phase 2:  Misha 2.0 engine build (parallel)      — 4-6 months
Phase 3:  Style-Synth product on Misha 2.0       — 2-3 months after Phase 2
Phase 4:  Commercial scale                        — ongoing
```

Papers can all be submitted within 6 months. Product launches month 8-10.

---

## Phase 0: Close Paper 1 (1-2 months)

**Goal:** CausalChess as representation learning, standalone paper, submission-ready.

### Tasks

- [x] v1 CNN trained (33.15%)
- [x] v2 CNN training clean split (40.4% mem, 35.0% no-mem)
- [x] Refresh all §4 numbers with v2
- [x] Tier 2 MLP probes
- [x] Tier 3 fine-tune evaluations
- [x] Baseline autoencoder training
- [x] Baseline contrastive SimCLR training
- [x] Visualizations: t-SNE by ECO, chess algebra plot
- [ ] Scale style detection to 10-20 GMs (data download + 1 day) — opcional
- [ ] Cross-era style test (1 day) — opcional
- [x] Paper polish: §1, §5 Conclusion, figures
- [x] Final SEALED Dec 2024 evaluation
- [ ] **Pre-submission fixes (ver `PENDING.md` → checklist ToG)** — EN CURSO
- [ ] Submit to **IEEE Transactions on Games (ToG)** — revista, sin presentación

### Deliverable
Paper 1 "Learning Chess Position Representations via Previous-Move Prediction" submitted to IEEE ToG.

### Compute
1 × 4060 for ~4-6 weeks (mostly baselines training sequentially).

### Output
- ~1 paper submission
- HuggingFace weights + Zenodo dataset for reproducibility
- Blog post + arxiv preprint

---

## Phase 1: Papers 2-3 with academic engine forks (2-4 months)

**Goal:** Concrete Elo improvements on Leela/Stockfish demonstrated, two more papers in the pipeline.

This is the **high-value, short-term** path. LCC and NNUECC experiments produce CONCRETE Elo gains — better PR impact than CausalChess-alone (Paper 1). Uses GPL academic forks; fine for research submissions.

### 1A: Paper 3 experiments (the core high-value work)

- [ ] Fork `lczero-training` (academic) — setup 4060 for fine-tune
- [ ] Implement input augmentation (concat CausalChess embedding with zero-init of W_e)
- [ ] Baseline run: fine-tune Lc0 T82 for 1 week on 4060 (control)
- [ ] Augmented run: LCC fine-tune, 1 week, identical setup
- [ ] Match LCC vs baseline, 2000-5000 games with tight CI
- [ ] Fork Stockfish NNUE trainer (`nnue-pytorch`)
- [ ] Implement NNUECC: augment first layer with W_e zero-init
- [ ] Train NNUECC from existing SF NNUE checkpoint, couple days compute
- [ ] Match NNUECC vs SF baseline
- [ ] Write up Paper 3 with concrete Elo numbers

### 1B: Paper 2 experiments (FMC + learned distance)

- [ ] Use existing Misha (SF-fork FMC, GPL academic) — NOT Misha 2.0 yet
- [ ] Integrate CausalChess distance as VR = R × D in Misha
- [ ] Tournament: hand-crafted distance vs learned distance
- [ ] Tight CI Elo lift
- [ ] Cross-framework: add distance as diversity bonus in Leela MCTS (lczero-training fork)
- [ ] Write up Paper 2

### Why academic forks are fine for papers

Reviewers accept "experimental fork of Lc0" or "modified Stockfish NNUE training pipeline." The code derivation is disclosed; results are valid research contributions. This is how most chess AI papers are published (Maia derives from Lc0, many NNUE papers build on Stockfish).

### Compute
- 1 week fine-tune Leela baseline (4060)
- 1 week fine-tune LCC (4060)
- 2-3 days NNUECC training
- 2 weeks tournament matches
- **Total: 4-6 weeks 4060 wall-clock**

### Deliverables
- Paper 2 submitted
- Paper 3 submitted
- Concrete numbers: "CausalChess augmentation gives +X Elo on Lc0, +Y Elo on SF"

### Risk
- **Leela fine-tune in 1 week 4060 may be too little compute to see signal.** Mitigation: ablations with smaller learning rate, longer runs; if signal isn't there, we report honestly.
- **Matching 2000-5000 games takes time.** Mitigation: OpenBench-style SPRT early-stop when CI is tight enough.

---

## Phase 2: Build own engine "Misha 2.0" (4-6 months, PARALLEL)

**Goal:** 2800-3000 Elo engine, 100% non-derivative code, own license, commercial-ready.

This is the **product bottleneck**, NOT the paper bottleneck. Runs in parallel to Phase 1 papers. Can be worked on evenings/weekends while papers advance on main compute track.

### Tasks

- [ ] Week 1: Rust setup, `shakmaty` integration (permissive chess library for movegen)
- [ ] Week 2-4: Alpha-beta search from scratch (textbook algorithms)
- [ ] Week 5-8: Transposition table, move ordering, q-search, late move reduction
- [ ] Week 9-12: Eval integration — CausalChess embedding as eval input, train small NNUE-style head
- [ ] Week 13-16: FMC search implementation (our own algorithm, clean-room)
- [ ] Week 17-18: UCI wrapper, time management, opening book
- [ ] Week 19-24: Self-play tuning + SPSA to reach 2800-3000 Elo

### Deliverable
Misha 2.0 — standalone engine, Rust source, MIT/Apache/proprietary license (our choice).

### Compute
Development intermittent. Self-play tuning: ~1 week continuous 4060 at end.

### Risk
- Reaching 2800+ Elo is non-trivial. Tuning phase may extend.

### Parallel activities during Phase 1
- Submit Paper 1 during Phase 0 → revisions during Phase 1
- Style-Synth design iteration (no code yet, just research into centroids, policy, etc.)

---

## Phase 3: Style-Synth MVP on Misha 2.0 (2-3 months, AFTER Phase 2)

**Goal:** Launch commercial Style-Synth product on our own (non-GPL) engine.

Waits for Misha 2.0 (Phase 2) to be ready.

- [ ] Compute 20 GM style centroids (Carlsen, Kasparov, Tal, Nakamura, ...)
- [ ] Move selector on Misha 2.0 top-K + style filter (1-2 days once engine exists)
- [ ] UCI engine with `--style=<gm>` option
- [ ] Web demo (React + Docker backend)
- [ ] Beta test with 5-10 strong players
- [ ] Monetization tiers design
- [ ] Launch

### Compute
2-3 weeks 4060 for training + matches + self-play tuning of style mixing.

### Output
1 commercial product in beta

### Risk
- Misha 2.0 may not reach 2800 Elo (falls short of Carlsen-target). If so: target 2500-2700 as "intermediate level GM trainer" instead.

---

## Phase 4: Product scale (ongoing)

**Goal:** Style-Synth commercial launch and scale.

- [ ] Pricing tiers validated
- [ ] Landing page + marketing
- [ ] Partnership outreach (chess.com, Lichess, streaming)
- [ ] Community: GM endorsements, YouTube demos
- [ ] Potential acquisition by chess platform

### Revenue targets
- Month 1-3: $0 (beta free)
- Month 4-6: $1-5k MRR (paid beta)
- Month 7-12: $10-50k MRR (public launch)
- Year 2: ??? (platform licensing or acquisition)

---

## Timeline (revised)

```
Month  1      2      3      4      5      6      7      8      9     10     11     12
      [── Phase 0 (P1) ──]
               [────── Phase 1 (P2+P3 papers on forks) ──────]
               [───────── Phase 2 (Misha 2.0 engine, parallel) ────────────]
                                                        [── Phase 3 (Style-Synth on Misha 2.0) ──]
                                                                                       [── Phase 4 ──>
```

**Papers all submitted:** month 6.
**Product MVP ready:** month 8-10.
**Commercial launch:** month 10-12.

Phase 1 (papers on forks) and Phase 2 (own engine build) run IN PARALLEL. Phase 1 uses main compute (GPU), Phase 2 uses developer time (code writing, no GPU needed until tuning phase at the end).

---

## Critical dependencies (revised)

```
Phase 0 (Paper 1) — independent
    ↓
Phase 1 (Papers 2-3 on academic forks) — needs v2 CausalChess from Phase 0
    ↓
Phase 2 (Misha 2.0 own engine) — PARALLEL to Phase 1, independent
    ↓
Phase 3 (Style-Synth MVP) — needs Phase 2 (own engine) for commercial
    ↓
Phase 4 (Scale)
```

**SPOF for PRODUCT:** Phase 2 (own engine). If we can't build 2800+, product doesn't ship.
**NOT SPOF for PAPERS:** Phase 2 is NOT a prerequisite for any paper. Papers 1-3 can all submit with academic forks.

---

## Fork-in-the-road decisions

### D1: Full-time vs part-time
- Full-time: Phase 1 finishes in 4 months
- Part-time (evenings/weekends): 8-10 months

### D2: Solo vs collaborator
- Solo: 12-14 months for full megaproject
- +1 engineer for Phase 1: 8-10 months
- +1 engineer + compute grant: 6-8 months

### D3: Monetization model
- **Premium SaaS** (subscription): Style-Synth direct to users
- **B2B licensing**: sell to chess.com, Lichess, Chess24 as feature
- **Grant-funded research**: no monetization, academic freedom
- **Consulting add-on**: hire out for chess AI consulting based on credibility

### D4: Engine target Elo
- 2800 Elo (Carlsen-level, easier): 4-5 months
- 3000 Elo (world-championship level): 6-9 months
- 3200+ Elo (top 5 engines): 12+ months (probably not feasible solo)

**Recommendation:** target 2800 Elo first, iterate upward. Good enough for Style-Synth "Carlsen" experience.

---

## What we are NOT doing

Avoid scope creep. Explicitly out-of-scope:

- ❌ Beating Stockfish or Leela in head-to-head (they are decades of person-years ahead)
- ❌ Building a non-chess version (unless Phase 4 scales hugely)
- ❌ Training FMC for Go, Shogi, etc. (tempting but tangential)
- ❌ Opening theory research (use existing books)
- ❌ Endgame tablebases (use Syzygy via standard probe)

---

## Exit criteria per phase

- **Phase 0 complete:** Paper 1 submitted to IEEE ToG
- **Phase 1 complete:** Misha 2.0 reaches 2800 CCRL in self-test, own license file committed
- **Phase 2 complete:** Paper 2 shows Elo lift from learned distance, submitted
- **Phase 3 complete:** Paper 3 submitted + Style-Synth MVP playable
- **Phase 4 complete:** Revenue validation ($5k+ MRR or acquisition offer)

---

## Review cadence

- Weekly: task progress against phase
- Monthly: phase status, risks, pivot decisions
- Quarterly: whole roadmap re-examination

---

*Last updated: 2026-06-16 — Paper 1 experiments complete; pre-submission fixes in progress for IEEE CoG.*
