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

## D-008 · 2026-09-30 · Alpamayo 2 Super CoC training target uses the tagged component format (working hypothesis, pending GPU observation)
**Decision.** Stage-1 targets follow the format `build_conversation` produces in training mode with `cot` in `components_order`: `<|cot_start|>cot<|cot_end|><|meta_action_start|>…<|meta_action_end|><|traj_future_start|>…`; the per-component label mask of the 1.5 recipe applies unchanged.
**Why.** In `NVlabs/alpamayo2` (`5e7975f`), `SPECIAL_TOKENS_KEYS` (`models/utils.py`) contains `cot_start`/`cot_end` and `meta_action_start`/`meta_action_end`, added to the tokenizer by `config.py`, and `build_conversation` emits them for assistant components in training mode. The release's inference path is different: `helper.create_messages` passes `components_order=["image","traj_history","prompt"]`, so no assistant component is built; `components_prompt=["cot","traj_future"]` only phrases the user request in natural language, the empty assistant turn is dropped, and generation starts with `add_generation_prompt=True`, without `<|cot_start|>`.
**Open until GPU.** Whether the released model emits `<|cot_start|>` itself as the first generated token, or free text followed directly by `<|traj_future_start|>`. One rollout with `raw_outputs` settles it (unknown 1). If the model does not emit the tag, either the targets drop the CoC tags to match inference, or inference is switched to training-mode prompting; that choice becomes D-011.
**Rejected for now.** An untagged assistant-span target, which would need a different label mask and leave the meta-action block unmarked — reconsidered if the GPU observation goes that way.
**Update 2026-10-11 — what the official recipe says (not settled, stays a working hypothesis).** Review in `docs/recipe_a2_sft_review.md` §1 and §7.3. (a) `alpamayo-recipes` #39 does not train CoC or meta-actions (`recipes/alpamayo2_sft/configs/preprocess/trajectory.yaml:2-4`; README: "no reasoning parquet, so CoC/meta-action SFT is not enabled"); its template, if `cot` were added to `components_order`, would produce the tagged form above (`build_conversation` → `construct_cot`). (b) Its label mask is the whole assistant span (`recipes/alpamayo2_sft/processor.py:147-158`, `label_mask.py`), so the "Rejected for now" reason no longer holds: the official recipe uses exactly that mask, and it is compatible with tagged targets. (c) `NVlabs/alpamayo2` #18 tests model the release's trajectory generation as untagged text (`tests/test_text_extraction.py:170-181`: `"A red light is ahead.\nLongitudinal: brake…<|traj_future_start|>"`), which points against the hypothesis that the release emits `<|cot_start|>`. A test fixture is not an observation: unknown 1 stays GPU, and D-011 stays unwritten until block 1 of `docs/gpu_session_01.md` has run.

## D-010 · 2026-09-30 · `afh` reasons in camera identifiers, not tensor positions (adopted)
**Decision.** Degradations, target severities and target texts are keyed by the loader's `camera_indices`; a tensor position is never taken for a camera. Fix in alpamayo-faithfulness PR #7 (`afh/cameras.py`, `apply_degradation(..., camera_indices=)`), `camera_indices` required for manifests in PR #8.
**Why.** The 1.5 loader returns 4 cameras `[0, 1, 2, 6]` and the 2 Super task profiles 6 cameras `[0, 1, 2, 3, 5, 6]`; with the former position convention, position 3 of the 1.5 tensor (front telephoto) was named "rear-left" and missed the front-camera floors, which is exactly the "black front camera described as visible" defect D-007 forbids.
**Rejected.** Remapping in each runner (duplicated logic, silent when forgotten).

## D-014 · 2026-10-11 · Adopt the official Alpamayo 2 Super recipe as the W1 base (proposed)
**Proposal.** W1 builds on `NVlabs/alpamayo-recipes` `recipes/alpamayo2_sft` (#39, commit `c29dc61`; environment pins `NVlabs/alpamayo2` `9596749`) instead of porting the 1.5 recipe. Our work becomes layers on top of it, each a candidate upstream PR: (1) a PEFT/LoRA option on `TrainableAlpamayo2Super` (D-004); (2) a preprocessing config with `cot` (and later `meta_action`) in `components_order`, fed by our targets; (3) a dataset adapter from the `afh` uncertainty manifests to the recipe's `PhysicalAIDataset` sample contract. The 1.5 port plan of `docs/finetune_design.md` §3 is superseded where the recipe covers the same step (processor, collator, stage-1/stage-2 forwards, ZeRO-3 launch, evaluation).
**Why.** The recipe is release-native (loads `nvidia/Alpamayo2-Super` without conversion), already implements and smoke-tests both stages (stage-1 loss 1.81 → 1.73 in two steps, held-out `eval_min_ade` 0.8288 m), and its forward is the release's own loss with separate `future_traj_ce` / `others_ce`. ROADMAP §8 lists "NVIDIA ships a 2 Super recipe during W1" with the response "adopt it". Review with file and line references: `docs/recipe_a2_sft_review.md`.
**What it does not cover.** No LoRA/PEFT (no `peft` anywhere in the recipe); no CoC or meta-action supervision; hardware is 32 H100 for stage 1 and 16+ for stage 2 in full fine-tuning (ZeRO-3), single-node only as a 1-camera, 1-frame, CPU-offload smoke.
**Re-examination of D-003 (1.5 first) — for Ousseynou to decide.** Its stated premise ("no 2 Super training recipe exists") is now false. Its other reasons still stand as written: 10B vs 34B cost, and the precondition that the 1.5 blackout test shows whether 1.5 shares the defect. What changes: the pipeline-calibration argument weakens, since the 2 Super training path now exists and is NVIDIA-maintained, while a 1.5 run would still go through the A1-format conversion and the 1.5 recipe. Options: (a) keep D-003 as is; (b) keep 1.5 only as the cheap blackout/W3 pilot and move W1 to 2 Super now; (c) drop 1.5. Block 3 of `docs/gpu_session_01.md` (1.5 blackout) and block 2 (2 Super memory on one GPU) give the inputs for that choice.
**Re-examination of D-004 (LoRA first on 2 Super) — unchanged in intent.** The recipe is full fine-tuning; LoRA is not provided, so D-004 now means adding PEFT to `TrainableAlpamayo2Super` (LM `q/k/v/o_proj` + `gate/up/down_proj`, `docs/finetune_design.md` §3) rather than to a ported recipe. Whether LoRA on 34B fits the ROADMAP's 4×H100 is unknown 6, measured in block 2.
**Contradictions to resolve, not decided here.** (1) ROADMAP W1 still says "No official recipe exists (verified 2026-09-30)". (2) ROADMAP W0 / §6 budget assume 4–8 H100 and LoRA at ~$12/h; the recipe as shipped needs 32 H100. (3) D-008's rejection argument is obsolete (see its 2026-10-11 update). (4) W3 needs CoC supervision, which the recipe leaves disabled; its 0.8288 m minADE is measured without generated CoC and is not comparable to `afh` measurements.
**Rejected (for now).** Continuing the hand-written 1.5 → 2 Super port: it would duplicate maintained NVIDIA code and drift from it (principle 5, "contribute, don't fork").
