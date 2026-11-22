# Aconcagua

Next summit after Alpamayo: a derived model that beats its NVIDIA origin on calibrated
uncertainty, causal faithfulness, clean-input non-regression and the long tail, built as bricks
contributed upstream. Base model since D-015: Alpamayo 2 Edge (2B); Alpamayo 2 Super is the
reference and future teacher. Plan in `docs/ROADMAP.md`, decisions in `docs/DECISIONS.md`.

## Layout

| Path | Content |
|---|---|
| `docs/ROADMAP.md`, `docs/DECISIONS.md` | living plan and decision log |
| `docs/finetune_design.md` | W1 design note: the Alpamayo 1.5 SFT recipe, the exhaustive 1.5 vs 2 Super diff, the LoRA port plan, the unknowns table |
| `docs/recipe_a2_sft_review.md` | review of NVIDIA's official Alpamayo 2 Super SFT recipe (alpamayo-recipes #39) and alpamayo2 #18 |
| `docs/edge_review.md` | review of Alpamayo 2 Edge: inference repository, official SFT recipe, checkpoint config (`docs/configs/alpamayo2-edge/`) |
| `docs/gpu_session_02.md`, `docs/observations/TEMPLATE-session02.md` | GPU session on Edge (one 80 GB GPU), block by block, and its observation template |
| `docs/gpu_session_01.md`, `docs/observations/TEMPLATE-session.md` | earlier session plan for 2 Super and 1.5; never run, kept |
| `runners/` | GPU scripts; each docstring names the NVlabs environment it runs in |
| `tests/` | cold tests (no GPU, no torch) |
| `scripts/install_afh.sh` | pinned install of the `afh` harness from GitHub |
| `scripts/fetch_edge_sources.sh` | pinned source checkouts of `NVlabs/alpamayo2-edge` and `NVlabs/alpamayo-recipes`, read as text by the cold API check |

## Setup (cold)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/install_afh.sh        # afh harness from github.com/oussndou77/alpamayo-faithfulness, pinned
bash scripts/fetch_edge_sources.sh # optional: pinned NVlabs sources for tests/test_edge_pinned_api_cold.py
python -m pytest tests
```

The harness is a pip dependency installed from GitHub at a pinned commit, never copied. Its
fixtures (the reference layout the tests compare against) come from the pinned checkout the
script leaves under `.deps/`, which is used for nothing else. The NVlabs checkouts are source
only: nothing is installed from them here, and the tests that read them skip when they are absent.

## GPU runners

`runners/probe_blackout_a15.py` reproduces the blackout test of `NVlabs/alpamayo2#9` on
Alpamayo 1.5 (baseline, then every camera black, K = 5 rollouts) and writes a JSON in the
layout of the harness fixture `fixtures/cf_0ea6fd88_a2_blackout.json`. It runs inside the
`NVlabs/alpamayo1.5` environment with `afh` installed.

`runners/probe_tokens_a2.py` runs the Alpamayo 2 Super release inference path unchanged and
reports what it generates (first tokens, whether `<|cot_start|>` appears, meta-actions).
`runners/probe_train_memory_a2.py` measures one stage-1 training step on one GPU through the
official recipe path, falling back to fewer cameras then fewer frames on OOM or on a sequence
longer than 6 144 tokens (recorded, never truncated). Both run in the
`alpamayo2_sft` recipe environment; `runners/probe_common.py` holds their torch-free logic.

### Alpamayo 2 Edge (GPU session 02)

Four probes, all in the environment of the official recipe (`uv sync --locked` in
`recipes/alpamayo2edge_sft`), on the 19 annotated events of the recipe's demo asset:

| Runner | Block | What it records |
|---|---|---|
| `runners/probe_tokens_edge.py` | 1 | raw generation of the release path: first tokens, where the trajectory anchor is, length of the reasoning before it (zero included), trajectories |
| `runners/probe_blackout_edge.py` | 2 | the same with every camera black, then the front camera only, and the trajectory deviation from the clean run |
| `runners/probe_nav_edge.py` | 3 | the same with the navigation instruction changed (mirrored, then "Continue straight"), validated by the recipe's own validator |
| `runners/probe_train_memory_edge.py` | 4 | one full joint training step through the recipe's model, processor and collator on one GPU: real sequence length, peak memory, seconds per step, with the same record-then-fall-back logic as the 2 Super probe and no truncation |

`runners/probe_edge_common.py` holds their shared logic; its torch-free half is tested cold.
