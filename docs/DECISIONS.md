# Decision log — Project Aconcagua

One entry per design decision: what was decided, why, what was rejected. Append-only; a reversed decision gets a new entry that references the old one.

## D-001 · 2026-09-30 · Start from Alpamayo 2 Super, contribute upstream rather than fork
**Decision.** Build the next generation as bricks on top of Alpamayo 2 Super, each targeting a PR to NVIDIA's open repos.
**Why.** Retraining from scratch is out of reach (110k hours of proprietary driving data); post-training adds capabilities the base lacks. A fork is a dead end; an accepted contribution is a foundation.
**Rejected.** Rebuilding a model beside NVIDIA's on their own terrain.

## D-002 · 2026-09-30 · Success is four measurable criteria, not "the next FSD"
**Decision.** Calibrated uncertainty, causal faithfulness, no clean-input regression, long-tail challenge scores — all vs the base model.
**Why.** An ambition needs a verifiable spec; each criterion has an existing measurement in the repo or a public benchmark.

## D-003 · 2026-09-30 · Alpamayo 1.5 first, 2 Super second
**Decision.** Prove the uncertainty pipeline on Alpamayo 1.5 (existing SFT recipe, 10B) before the 2 Super (no recipe, 34B).
**Why.** Verified on 2026-09-30 that no 2 Super training recipe exists in `alpamayo-recipes` or `alpamayo2`. The 1.5 run calibrates the whole pipeline at a fraction of the cost.
**Precondition.** A blackout test on 1.5 must first show whether it shares the defect; either outcome is informative.

## D-004 · 2026-09-30 · LoRA before full fine-tuning on 2 Super
**Decision.** First recipe variant is LoRA on the VLM.
**Why.** Fits on 4×H100, limits catastrophic forgetting, easier to review upstream. Full fine-tuning only if LoRA plateaus.

## D-005 · 2026-09-30 · Parallel agents only on independent tasks
**Decision.** 3–5 agents on separate deliverables (design note, data inventory, tests, runners); never many agents on one training script.
**Why.** The bottleneck is GPU validation and a single coherent design, not code volume. Coordination cost grows faster than output.

## D-006 · 2026-09-30 · 2026 challenges are for learning the workflow; 2027 editions are the target
**Decision.** Register now, push the base 2 Super through the submission pipeline for a reference score; aim trained derivatives at the 2027 editions.
**Why.** AlpaSim Closed-Loop Challenge 2026 froze rules on 2026-09-15 and closes its leaderboard 2026-10-31 — too soon for any trained derivative.

## D-007 · 2026-09-30 · Training targets: text and action never contradict
**Decision.** Inherited from the uncertainty pipeline (PRs #2–#4): the text announces a stop exactly when the trajectory stops; a slowed trajectory is described as slowing; a black front camera is never described as "clearly visible".
**Why.** The harness audits what the model says against what it does; training data must pass the same check, or we teach the defect we audit for.
