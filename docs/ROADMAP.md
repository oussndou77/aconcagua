# Project Aconcagua — roadmap

*Alpamayo peaks at 5,947 m. Aconcagua, in the same range, is the next summit: 6,961 m. Same chain, higher.*

**Living document.** Every milestone updates it. Decisions and their reasons live in `DECISIONS.md`.
Started 2026-09-30.

## 1. Goal and definition of success

Build, from NVIDIA's Alpamayo models, a derived model that beats its origin on four measurable
points, with its building blocks contributed upstream. Since D-015 (2026-11-22) the base is
**Alpamayo 2 Edge** (2B); Alpamayo 2 Super (34B) is the reference model and the future teacher.

| Criterion | Measure | Baseline |
|---|---|---|
| Calibrated uncertainty under sensor degradation | AUROC + severity/uncertainty correlation (`afh/eval_uncertainty.py`) | 2 Super: ≈ 0.5 — narrates "clear path ahead" under blackout ([NVlabs/alpamayo2#9](https://github.com/NVlabs/alpamayo2/issues/9)). Edge: not measured yet (GPU session 02, blocks 1–2) |
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
One 80 GB GPU first (RunPod, 100 GB container disk), then whatever GPU session 02 block 4 says a training run needs: the Edge recipe is written for 8 GPUs (ZeRO-3) and names no GPU model. Experiment tracking (the recipe logs to W&B), artifact bucket, reproducible setup script (`uv sync --locked` of the recipe).
**Done when:** a smoke training runs and resumes after interruption.

### W1 — Run the Edge recipe, measure it, plug our targets in
Base: Alpamayo 2 Edge (D-015) on the official recipe `alpamayo-recipes/recipes/alpamayo2edge_sft`, native configuration `sft_joint_nav_cot` (D-016, proposed). The recipe already has the processor, collator, label mask, joint loss and evaluation; W1 no longer writes a recipe. Review: `docs/edge_review.md`. The 2 Super recipe review (`docs/recipe_a2_sft_review.md`) is kept for the reference model.
1. **Run it**: the recipe's own evaluation on the base checkpoint, then one joint training step on one GPU (GPU session 02, blocks 4–5).
2. **Measure it**: what the base model generates, with and without cameras, and whether it follows the navigation instruction (blocks 1–3); memory and time per step (block 4). LoRA only if full fine-tuning does not fit on 80 GB (D-004, suspended).
3. **Plug our targets in**: the three extension points listed under W3.
**Done when:** a clean-input fine-tune with the native configuration reproduces base behaviour with no regression.

### W2 — Data
1. Targeted synthetic: NuRec (remove/insert/move agents, weather, time) — also the photometric counterfactual the audit needs.
2. Convertible public datasets: nuPlan first (AlpaSim challenge dev kit supports joint PAI-AV + nuPlan training), then Waymo Open, Argoverse 2, ZOD, nuScenes. Camera remapping to the model's slots (7 for 2 Super; 6 for Edge, with 22 images at irregular time steps and a per-camera pixel budget), calibration, ego-motion.
3. CoC labels via `NVlabs/alpamayo-coc-autolabeler`.
Deliverables: costed inventory (hours, licence, conversion effort), converters.
**Done when:** 10,000 labelled examples outside the NVIDIA dataset, in training format.

### W3 — Capability 1: uncertainty under degradation
Pipeline ready on the `afh` side; trained on Alpamayo 2 Edge (D-015). Three extension points are missing between our targets and the recipe (`docs/edge_review.md` §6):
1. **A navigation instruction per event.** It is a mandatory input of Edge's native task; public PAI does not provide it, and neither do our manifests.
2. **An entry for degraded images.** The recipe's dataset decodes frames straight from PAI and hands them to preprocessing; `afh` degradations also assume a rectangular camera × time tensor, which Edge does not use (D-010 update).
3. **A replacement of the target trajectory.** The recipe recomputes the future from ego-motion; a damped target must replace `ego_future_xyz` / `ego_future_rot`, which feed both the discrete trajectory tokens and the expert.
The baseline is the base Edge model measured with `afh` (reasoning + trajectory, same clips, same controls). minADE has three definitions in play (recipe: 6.4 s, 6 candidates; model card: 5 s; `afh`: shortest common horizon) and none of them reads the reasoning: never compare across them.
Deliverables: before/after video, two-axis evaluation report, results on issue #9.
**Done when:** calibration AUROC clearly above base, zero clean-input regression, generalisation to held-out families and compounds.

### W4 — Capability 2: causal faithfulness via RL
The harness becomes the reward: a rollout is rewarded when its explanation survives the negative control and vanishes under causal occlusion. AlpaGym custom reward, GRPO.
Risk: reward hacking → composite reward (faithfulness + trajectory quality + non-regression), negative control inside the reward, held-out scenes.
**Done when:** citation contrast up on held-out scenes with no minADE loss.

### W5 — Long tail in closed loop
AlpaSim + AlpaGym on NuRec then OmniDreams scenes, with W3 degradations injected: the model must *drive* cautiously under sensor failure, not only say so.
**Done when:** fewer collisions / off-road on a long-tail scenario set at equal progress.

### W6 — Distillation: 2 Super → Edge
2 Super as teacher, Edge as student. **No official recipe covers it**: `recipes/alpamayo_distillation` accepts only Alpamayo 1 / 1.5 as teachers and `Cosmos-Reason2-2B` as student, and recommends 128 H100 (`docs/edge_review.md` §9). This workstream stays ours to design. Deployment on DRIVE AGX Thor goes through `recipes/alpamayo2edge_edgellm` (ONNX → TensorRT, FP8), which only documents the released checkpoint.
**Done when:** the student keeps the teacher's calibration and faithfulness.

### W7 — Evaluation, publication, contribution (continuous)
Harness extensions, upstream PRs, issue #9 updates, challenge submissions, workshop paper.
**Done when:** ≥1 accepted contribution to an NVlabs repo and a submission to each challenge.

## 5. Schedule

| Phase | Period | Content | Exit milestone |
|---|---|---|---|
| 0 — Foundations | Oct–Nov 2026 | W0; W1 design note and recipe reviews; W2 inventory; **GPU session 02 on Edge** (base behaviour, blackout, navigation, memory); **register for both 2026 challenges and push base 2 Super through the submission pipeline (reference score)** | Edge measured; recipe running |
| 1 — The recipe | Dec 2026–Jan 2027 | W1 done; W3 on Edge; results on issue #9; upstream PRs for the extension points | Results shared with NVIDIA |
| 2 — Causality | Jan–Feb 2027 | W4; W2 extension | First VLA trained for causal faithfulness |
| 3 — Closed loop | Mar–May 2027 | W5 | Closed-loop metrics |
| 4 — The student | Jun 2027 + | W6; paper; **2027 challenge editions** | Deployable model + publication |

**Calendar constraint (2026-09-30):** the AlpaSim Closed-Loop Challenge 2026 froze rules on 2026-09-15, closes its public leaderboard on 2026-10-31, releases results 2026-11-15 (NeurIPS 2026 workshop). A trained derivative cannot make it; the 2026 editions are for learning the submission workflow, the 2027 editions are the target.

## 6. Budget (orders of magnitude)
Revised 2026-11-22 for Edge (D-015). Working unit: **one 80 GB GPU**, about $3/h at the H100 rate used so far ($12/h for 4). Inference and probes fit on it (NVIDIA measures 12 GiB at inference with 6 candidates). Training cost is **not known yet**: the recipe is written for 8 GPUs (about $25/h at the same rate) and gives no memory figure; GPU session 02 block 4 measures memory and time per step on one GPU, and the W3 estimate follows from it. Storage is a line of its own: the recipe's default PAI subset is 181 GiB, all public CoC chunks 7.7 TiB. Earlier envelope, kept until re-estimated: W3 ≈ $3–6k; W4–W5 ≈ $15–30k (closed-loop RL is the heavy item); six months ≈ **$30–60k** compute, plus storage. The 2 Super figures (32 H100 for its recipe) now only matter for W6, where 2 Super is the teacher.

## 7. Organisation
- Ousseynou decides; every design decision is logged in `DECISIONS.md`.
- Claude (chat): architecture, review, strategy, project memory.
- Claude Code (cloud): all cold work on this repo.
- Parallel agents: independent tasks only, 3–5 at a time, never the same file.
- GPU: RunPod for short sessions, rented cluster for long training.

## 8. Risks
| Risk | Mitigation |
|---|---|
| NVIDIA ships a recipe during W1 | **Happened twice**: 2 Super on 2026-09-30 (`alpamayo-recipes` #39) and Edge on 2026-11-10 (#50). Response: D-015 and D-016 (adopt the Edge recipe); keep watching the repo weekly. |
| Loss of CoC when training on trajectory only | Edge's default config (`sft_joint`) is trajectory-only and changes the prompt format; only `sft_joint_nav_cot` supervises the reasoning, with a token weight of 0.2 against 1.0 for the trajectory. Always train with the native config; 40% clean examples whose target is the model's own CoC; `afh` axes 1–4 on clean input as a blocking non-regression check. |
| Navigation is mandatory and absent from PAI | Edge's native task needs a navigation instruction per event; public PAI has none, and inference refuses to run without one. Measure how much the output depends on it (GPU session 02, block 3) before choosing how to produce instructions for our events. |
| Public CoC labels differ from the release's | The recipe says PAI's public CoC labels "differ from route-conditioned reasoning and nominal labels used by the release". Do not assume they are in-distribution targets; compare them with what the base model generates (block 1) before training on them. |
| Thor export keeps a 2,211-token output vocabulary | `alpamayo2edge_edgellm` reduces the LM head to the token ids the CoC uses. New uncertainty wording may fall outside it: check target texts against that list, or plan to rebuild it, before W6. |
| minADE has three definitions | Recipe (6.4 s, 6 candidates), model card (5 s, 1,392 samples), `afh` (shortest common horizon). State the definition with every number; compare only like with like. |
| Licences on redistributing derived weights | Edge weights are under OpenMDW-1.1 (text in `docs/configs/alpamayo2-edge/LICENSE`); the PAI dataset has its own access terms. **Verify both in Phase 0**, before sharing any checkpoint. |
| Catastrophic forgetting | Low learning rates as in the recipe (LoRA if D-004 is re-opened), 40% clean examples, non-regression as a blocking criterion. |
| Reward hacking (W4) | Composite reward, negative control in the reward, held-out scenes. |
| Closed-loop RL cost | Start with tens of NuRec scenes, measure cost per epoch, extrapolate before committing. |
| One person, lab-scale project | Every brick publishable alone; never depend on the end. |
| Loss of momentum | One visible result per month, minimum. |
