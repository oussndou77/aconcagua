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

## D-008 · 2026-09-30 · Alpamayo 2 Super CoC training target is tagged with special tokens (resolved)
**Decision.** The stage-1 target for 2 Super is `<|cot_start|>cot<|cot_end|><|meta_action_start|>…<|meta_action_end|><|traj_future_start|>…`; the per-component label mask of the 1.5 recipe applies unchanged.
**Why.** In `NVlabs/alpamayo2`, `SPECIAL_TOKENS_KEYS` (`models/utils.py`) contains `cot_start`/`cot_end` and `meta_action_start`/`meta_action_end`, added to the tokenizer by `config.py`; `helper.create_messages` asks for `components_prompt=["cot", "traj_future"]` and `get_component_str` emits only the opening token of a requested component, so a prompt ending in `<|cot_start|>` makes the model generate `cot <|cot_end|> <|meta_action_start|>…<|meta_action_end|> <|traj_future_start|>…`. Note for the record: the release's `create_messages` passes `components_order=["image", "traj_history", "prompt"]`, so its inference prompt drops the empty assistant turn; the tagged form is what `build_conversation` produces in training mode with `cot` in `components_order`.
**Rejected.** An untagged "assistant span" target, which would have required a different label mask and left the meta-action block unmarked.

## D-010 · 2026-09-30 · `afh` reasons in camera identifiers, not tensor positions (adopted)
**Decision.** Degradations, target severities and target texts are keyed by the loader's `camera_indices`; a tensor position is never taken for a camera. Fix in alpamayo-faithfulness PR #7 (`afh/cameras.py`, `apply_degradation(..., camera_indices=)`), `camera_indices` required for manifests in PR #8.
**Why.** The 1.5 loader returns 4 cameras `[0, 1, 2, 6]` and the 2 Super task profiles 6 cameras `[0, 1, 2, 3, 5, 6]`; with the former position convention, position 3 of the 1.5 tensor (front telephoto) was named "rear-left" and missed the front-camera floors, which is exactly the "black front camera described as visible" defect D-007 forbids.
**Rejected.** Remapping in each runner (duplicated logic, silent when forgotten).
