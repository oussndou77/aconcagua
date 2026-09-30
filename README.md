# Aconcagua

Next summit after Alpamayo: a derived model of NVIDIA Alpamayo 2 Super that beats its origin on
calibrated uncertainty, causal faithfulness, clean-input non-regression and the long tail, built
as bricks contributed upstream. Plan in `docs/ROADMAP.md`, decisions in `docs/DECISIONS.md`.

## Layout

| Path | Content |
|---|---|
| `docs/ROADMAP.md`, `docs/DECISIONS.md` | living plan and decision log |
| `docs/finetune_design.md` | W1 design note: the Alpamayo 1.5 SFT recipe, the exhaustive 1.5 vs 2 Super diff, the LoRA port plan |
| `runners/` | GPU scripts; each docstring names the NVlabs environment it runs in |
| `tests/` | cold tests (no GPU, no torch) |
| `scripts/install_afh.sh` | pinned install of the `afh` harness from GitHub |

## Setup (cold)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/install_afh.sh        # afh harness from github.com/oussndou77/alpamayo-faithfulness, pinned
python -m pytest tests
```

The harness is a pip dependency installed from GitHub at a pinned commit, never copied. Its
fixtures (the reference layout the tests compare against) come from the pinned checkout the
script leaves under `.deps/`, which is used for nothing else.

## GPU runners

`runners/probe_blackout_a15.py` reproduces the blackout test of `NVlabs/alpamayo2#9` on
Alpamayo 1.5 (baseline, then every camera black, K = 5 rollouts) and writes a JSON in the
layout of the harness fixture `fixtures/cf_0ea6fd88_a2_blackout.json`. It runs inside the
`NVlabs/alpamayo1.5` environment with `afh` installed.
