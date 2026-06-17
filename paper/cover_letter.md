# Cover Letter — IEEE Transactions on Games

[Date]

To the Editor-in-Chief,
IEEE Transactions on Games

Dear Editor,

We are pleased to submit our manuscript, **"Learning Chess Position
Representations via Previous-Move Prediction,"** for consideration as a
**full paper (regular research article)** in *IEEE Transactions on Games*.
This is not a candidate for any special issue.

## Summary and contribution

The manuscript introduces **Previous-Move Pretraining (PMP)**, a self-supervised
method for learning chess position representations *without* engine evaluations
or best-move labels. The pretext task is novel: given a board position, predict
the **previous move** that produced it. Solving this temporal retrodiction task
forces the network to infer the causal structure of how a position arose
(castling, pawn breaks, opening identity, game phase) rather than only "who is
winning" or "what is best."

Our main findings are:

- A 6-block CNN (2.3M parameters) trained on 25.3M (position, previous-move)
  pairs from 2400+ Elo Lichess games (June–August 2023) reaches 40.4% top-1
  accuracy on a 1,928-class task. A deduplicated variant reaches 35.0%, showing
  most of the signal is structural rather than opening-theory memorization.
- We derive an empirical-Bayes ceiling for the task (73.4%) to contextualize
  accuracy, and show the model operates well above the moderate-prior baseline,
  which is impossible without genuine structural generalization.
- The frozen 256-d embedding linearly decodes strategic concepts (castling,
  turn, material), identifies player style (54.4% over four GMs on balanced
  endgames), supports interpretable latent arithmetic, and predicts player Elo
  (R² = 0.52, MAE = 194 Elo) — all without identity labels during training.

The contribution is **representational rather than competitive**: the goal is a
richer, more human-like *understanding* of a position, motivating an
auxiliary-input design for engine integration as future work.

## Fit with IEEE Transactions on Games

Chess remains a canonical model system for the Games community, and
representation learning for game states is of broad interest beyond chess. To
our knowledge, previous-move prediction has not been used as a self-supervised
pretext task for game-state representation learning, making this a novel
methodological contribution within the journal's scope.

## Originality and ethics

This manuscript is original, has not been published previously, and is not under
consideration for publication elsewhere, in whole or in part. All authors have
approved the submission. The work does not involve human subjects or sensitive
data; it uses the publicly available Lichess game database. To support
reproducibility, the complete training, dataset-construction, and evaluation
code, together with trained model checkpoints, is publicly available at
[REPOSITORY URL — insert before submission].

## Corresponding author

Jesús Armando Mendoza Ramos
Path-Data
Email: armando@path-data.com

We thank the editors and reviewers for their time and consideration.

Sincerely,

Jesús Armando Mendoza Ramos
