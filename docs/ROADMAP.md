# Project Aconcagua — roadmap

*Alpamayo peaks at 5,947 m. Aconcagua, in the same range, is the next summit: 6,961 m. Same chain, higher.*

**Living document.** Every milestone updates it. Decisions and their reasons live in `DECISIONS.md`.
Started 2026-09-30.

## 1. Goal and definition of success

Build, from NVIDIA Alpamayo 2 Super, a derived model that beats its origin on four measurable
points, with its building blocks contributed upstream.

| Criterion | Measure | Baseline (Alpamayo 2 Super) |
|---|---|---|
| Calibrated uncertainty under sensor degradation | AUROC + severity/uncertainty correlation (`afh/eval_uncertainty.py`) | ≈ 0.5 — narrates "clear path ahead" under blackout ([NVlabs/alpamayo2#9](https://github.com/NVlabs/alpamayo2/issues/9)) |
| Causal faithfulness of the reasoning | Citation contrast, causal vs negative control (`afh/axes/counterfactual.py`) | Faithful on tested scenes; never optimised for |
| No regression on clean input | Axes 1–4 + minADE vs base, same clips | Reference |
| Long tail | PAI-AV OOD Reasoning Challenge, AlpaSim Closed-Loop Challenge | Base model score |

## 2. Construction principles

1. **Every brick stands alone.** Stopping after any stage leaves a publishable result.
2. **Everything is measured with controls** — null, negative, held-out. Visual checks become automated tests as soon as possible.
3. **Clean input never regresses.** A capability that degrades nominal behaviour is rejected.
4. **Text and action never contradict** — enforced on training targets by tests, on the model by evaluation.
5. **Contribute, don't fork.** Each brick targets a PR to `NVlabs/alpamayo-recipes` or a sibling repo.

## 3. Assets at start

- Causal-audit harness (4 axes, controls, validated on Alpamayo-R1 and 2 Super; methodology acknowledged by the Alpamayo team).
- Uncertainty data pipeline (`afh/degradation.py`, `afh/uncertainty_dataset.py`, `afh/eval_uncertainty.py`): 7 single + compound degradations, camera-aware severity, held-out splits, two-axis evaluation, 30 cold tests.
- Open channel with the Alpamayo team on issue #9, with an explicit request to share results and protocol.
- NVIDIA open ecosystem: `alpamayo-recipes` (1.5 SFT/RL/quant), AlpaSim, AlpaGym, CoC autolabeler, 1,607 NuRec scenes, Cosmos synthetic data.
- Tooling: Claude Code (cloud) for cold work, parallel agents for independent tasks, RunPod/cluster for GPU.

## 4. Workstreams

### W0 — Infrastructure
Rented cluster (4–8×H100), 500 GB–1 TB volume, experiment tracking (MLflow or W&B), artifact bucket, reproducible setup script.
**Done when:** a smoke training runs on 4 GPUs and resumes after interruption.

### W1 — Alpamayo 2 Super training recipe
No official recipe exists (verified 2026-09-30: `alpamayo-recipes` holds 1 / 1.5 SFT, 1.x RL, 1.5 quant only; `alpamayo2` has no training code). Port `recipes/alpamayo1_5_sft` (stage 1 VLM cross-entropy, stage 2 diffusion flow matching) to Cosmos 3 backbone, 7 cameras, meta-actions. **LoRA first.**
Deliverables: design note (exhaustive 1.5 vs 2 diff), data adapter, training script, 10-example smoke test, PR upstream.
**Done when:** a clean-input fine-tune reproduces base behaviour with no regression.

### W2 — Data
1. Targeted synthetic: NuRec (remove/insert/move agents, weather, time) — also the photometric counterfactual the audit needs.
2. Convertible public datasets: nuPlan first (AlpaSim challenge dev kit supports joint PAI-AV + nuPlan training), then Waymo Open, Argoverse 2, ZOD, nuScenes. Camera remapping to the 7 Alpamayo slots, calibration, ego-motion.
3. CoC labels via `NVlabs/alpamayo-coc-autolabeler`.
Deliverables: costed inventory (hours, licence, conversion effort), converters.
**Done when:** 10,000 labelled examples outside the NVIDIA dataset, in training format.

### W3 — Capability 1: uncertainty under degradation
Pipeline ready. Train on Alpamayo 1.5 first (existing recipe, 3× smaller; blackout test first to confirm the defect), then on 2 Super once W1 works.
Deliverables: before/after video, two-axis evaluation report, results on issue #9.
**Done when:** calibration AUROC clearly above base, zero clean-input regression, generalisation to held-out families and compounds.

### W4 — Capability 2: causal faithfulness via RL
The harness becomes the reward: a rollout is rewarded when its explanation survives the negative control and vanishes under causal occlusion. AlpaGym custom reward, GRPO.
Risk: reward hacking → composite reward (faithfulness + trajectory quality + non-regression), negative control inside the reward, held-out scenes.
**Done when:** citation contrast up on held-out scenes with no minADE loss.

### W5 — Long tail in closed loop
AlpaSim + AlpaGym on NuRec then OmniDreams scenes, with W3 degradations injected: the model must *drive* cautiously under sensor failure, not only say so.
**Done when:** fewer collisions / off-road on a long-tail scenario set at equal progress.

### W6 — Distillation
2 Super as teacher, 1.5 Nano as natural student, for DRIVE Thor deployment.
**Done when:** the student keeps the teacher's calibration and faithfulness.

### W7 — Evaluation, publication, contribution (continuous)
Harness extensions, upstream PRs, issue #9 updates, challenge submissions, workshop paper.
**Done when:** ≥1 accepted contribution to an NVlabs repo and a submission to each challenge.

## 5. Schedule

| Phase | Period | Content | Exit milestone |
|---|---|---|---|
| 0 — Foundations | Oct 2026 | W0; W1 design note; W2 inventory; 1.5 blackout test; first uncertainty fine-tune on 1.5; **register for both 2026 challenges and push base 2 Super through the submission pipeline (reference score)** | Pipeline proven on 1.5 |
| 1 — The recipe | Nov–Dec 2026 | W1 working; W3 on 2 Super; results on issue #9; recipe PR | Results shared with NVIDIA |
| 2 — Causality | Jan–Feb 2027 | W4; W2 extension | First VLA trained for causal faithfulness |
| 3 — Closed loop | Mar–May 2027 | W5 | Closed-loop metrics |
| 4 — The student | Jun 2027 + | W6; paper; **2027 challenge editions** | Deployable model + publication |

**Calendar constraint (2026-09-30):** the AlpaSim Closed-Loop Challenge 2026 froze rules on 2026-09-15, closes its public leaderboard on 2026-10-31, releases results 2026-11-15 (NeurIPS 2026 workshop). A trained derivative cannot make it; the 2026 editions are for learning the submission workflow, the 2027 editions are the target.

## 6. Budget (orders of magnitude)
LoRA on 2 Super: 4×H100 ≈ $12/h; full fine-tune: 8×H100 ≈ $25/h. W3 ≈ $3–6k; W4–W5 ≈ $15–30k (closed-loop RL is the heavy item). Six months ≈ **$30–60k** compute, plus storage.

## 7. Organisation
- Ousseynou decides; every design decision is logged in `DECISIONS.md`.
- Claude (chat): architecture, review, strategy, project memory.
- Claude Code (cloud): all cold work on this repo.
- Parallel agents: independent tasks only, 3–5 at a time, never the same file.
- GPU: RunPod for short sessions, rented cluster for long training.

## 8. Risks
| Risk | Mitigation |
|---|---|
| NVIDIA ships a 2 Super recipe during W1 | Good news: adopt it, redirect effort to W3–W4. Watch the repo weekly. |
| NVIDIA AV Dataset licence on redistributing derived weights | **Verify in Phase 0**, before sharing any checkpoint. |
| Catastrophic forgetting | LoRA, 40% clean examples, non-regression as a blocking criterion. |
| Reward hacking (W4) | Composite reward, negative control in the reward, held-out scenes. |
| Closed-loop RL cost | Start with tens of NuRec scenes, measure cost per epoch, extrapolate before committing. |
| One person, lab-scale project | Every brick publishable alone; never depend on the end. |
| Loss of momentum | One visible result per month, minimum. |
