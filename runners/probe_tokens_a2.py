#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_tokens_a2.py — what does Alpamayo 2 Super actually generate? (gpu_session_01, block 1)

Runs the release inference path unchanged on clip 0ea6fd88 at t0 = 5.1 s:

    load_physical_aiavdataset -> select_task_input(data, "trajectory")
    -> helper.prepare_model_inputs -> Alpamayo2Super.sample_trajectories_from_data(return_extra=True)

and keeps the raw output. The only addition is a recorder around `model.vlm.generate` that
copies the prompt length and the returned sequences BEFORE the release overwrites the tokens
after <|traj_future_start|> with padding. Nothing in the prompt, sampling or model is touched.

It presupposes nothing about the format (D-008 is a working hypothesis; NVlabs/alpamayo2 #18
tests model an untagged generation). For every run it reports:
  * the last prompt tokens (does the prompt end with <|im_start|>assistant\\n? with a tag?);
  * the first 10 generated tokens (ids, token strings, decoded text);
  * whether <|cot_start|> appears, at which position, or that it is absent;
  * the position of every structural marker (cot, meta_action, traj_future, im_end);
  * the full decoded text, the <|meta_action_start|>…<|meta_action_end|> block if any, and
    the meta-action lines found (tagged or untagged, source stated);
  * the release's own text extraction (extra[...]) for comparison;
  * `cot_for_training`: the CoC that block 2 (probe_train_memory_a2.py) puts in its
    training example.

Optionally (default on) it also runs the release text task "meta_action"
(text_tasks.prepare_text_generation_inputs + generate_text) for unknown 4.

Environment: the alpamayo2_sft recipe venv (NVlabs/alpamayo-recipes, pins NVlabs/alpamayo2
9596749; helper.py and conversation.py are identical to alpamayo2 main 5977692), with afh not
required. See docs/gpu_session_01.md, block 0.

Usage:
    python runners/probe_tokens_a2.py --out outputs/session01/tokens_a2_0ea6fd88.json
    python runners/probe_tokens_a2.py --n-runs 3 --repro-check     # unknown 11 as well
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_common import (  # noqa: E402
    DEFAULT_CLIP,
    DEFAULT_MODEL_ID,
    DEFAULT_T0_US,
    EXPECTED_TOKEN_IDS,
    MARKER_TOKENS,
    analyze_generation,
    check_token_ids,
    package_provenance,
    write_json,
)

PROMPT_TAIL = 24


class GenerateRecorder:
    """Wrap `vlm.generate`: record prompt length and a CPU copy of the sequences, return as is."""

    def __init__(self, generate):
        self._generate = generate
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args, **kwargs):
        out = self._generate(*args, **kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        seqs = out.sequences if hasattr(out, "sequences") else out
        self.calls.append({
            "prompt_len": int(input_ids.shape[1]) if input_ids is not None else None,
            "prompt_ids": input_ids[0].detach().cpu().tolist() if input_ids is not None else None,
            "sequences": seqs.detach().clone().cpu(),
        })
        return out


def _env() -> dict[str, Any]:
    import torch
    import transformers
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "alpamayo2_super": package_provenance("alpamayo2_super"),
        "alpamayo2_sft": package_provenance("alpamayo2-sft"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {"name": p.name, "total_memory_gib": round(p.total_memory / 2**30, 1)}
    return env


def _marker_ids(tokenizer) -> dict[str, Any]:
    ids = {}
    for key, tok in MARKER_TOKENS.items():
        i = tokenizer.convert_tokens_to_ids(tok)
        ids[key] = None if i is None or i == tokenizer.unk_token_id else int(i)
    return ids


def _analyze_call(call: dict, tokenizer, marker_ids: dict, first_n: int) -> dict[str, Any]:
    seq = call["sequences"][0].tolist()
    gen = seq[call["prompt_len"]:]
    pad = tokenizer.pad_token_id
    trailing_pad = 0
    while gen and pad is not None and gen[-1] == pad:
        gen.pop()
        trailing_pad += 1
    report = analyze_generation(
        gen,
        id_to_token=lambda t: tokenizer.convert_ids_to_tokens(int(t)),
        decode=lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False),
        marker_ids=marker_ids,
        first_n=first_n,
    )
    report["trailing_pad_tokens_removed"] = trailing_pad
    report["generated_ids"] = gen
    return report


def _prompt_report(prompt_ids: list[int], tokenizer) -> dict[str, Any]:
    tail = prompt_ids[-PROMPT_TAIL:]
    return {
        "prompt_len": len(prompt_ids),
        "tail_ids": tail,
        "tail_tokens": tokenizer.convert_ids_to_tokens(tail),
        "tail_text": tokenizer.decode(tail, skip_special_tokens=False),
        "contains": {k: tok in tokenizer.decode(prompt_ids, skip_special_tokens=False)
                     for k, tok in MARKER_TOKENS.items()
                     if k in ("cot_start", "meta_action_start", "traj_future_start")},
        "n_image_pad": sum(1 for t in prompt_ids if t == tokenizer.convert_tokens_to_ids("<|image_pad|>")),
    }


def _step(payload: dict, name: str, out: str, fn):
    """Run one step; record duration, result or error; write the JSON either way."""
    t = time.time()
    try:
        result = fn()
        payload["steps"][name] = {"status": "ok", "seconds": round(time.time() - t, 1)}
        return result
    except Exception as exc:  # noqa: BLE001
        payload["steps"][name] = {"status": "error", "seconds": round(time.time() - t, 1),
                                  "error": f"{type(exc).__name__}: {str(exc)[:800]}",
                                  "traceback": traceback.format_exc()[-3000:]}
        print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        write_json(out, payload)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    ap.add_argument("--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--seed", type=int, default=42, help="seed of run 0 (release smoke default)")
    ap.add_argument("--n-runs", type=int, default=1, help="independent runs, seeds seed..seed+n-1")
    ap.add_argument("--repro-check", action="store_true",
                    help="re-run the first seed and compare generated ids (unknown 11)")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--max-generation-length", type=int, default=None,
                    help="default: the release's own (max(256, tokens_per_future_traj))")
    ap.add_argument("--first-n", type=int, default=10)
    ap.add_argument("--skip-meta-action-task", action="store_true")
    ap.add_argument("--out", default="outputs/session01/tokens_a2_0ea6fd88.json")
    args = ap.parse_args(argv)

    payload: dict[str, Any] = {
        "probe": "probe_tokens_a2", "clip_id": args.clip, "t0_us": args.t0_us,
        "model_id": args.model_id, "args": vars(args), "steps": {}, "runs": [],
    }
    out = args.out

    import numpy as np
    import torch

    payload["env"] = _step(payload, "env", out, _env)
    if not torch.cuda.is_available():
        payload["steps"]["env"]["error"] = "no CUDA device"
        write_json(out, payload)
        raise SystemExit("CUDA GPU required")

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    def load_data():
        src = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
        return src, select_task_input(src, "trajectory")

    source, data = _step(payload, "load_data", out, load_data)
    payload["input"] = {"camera_indices": data["camera_indices"].tolist(),
                        "image_frames_shape": list(data["image_frames"].shape)}

    model = _step(payload, "load_model", out, lambda: Alpamayo2Super.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"))
    model.eval()
    payload["memory_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    tok = model.tokenizer
    marker_ids = _marker_ids(tok)
    resolved = {t: (None if (i := tok.convert_tokens_to_ids(t)) == tok.unk_token_id else int(i))
                for t in EXPECTED_TOKEN_IDS}
    payload["token_ids"] = check_token_ids(resolved)
    payload["token_ids"]["config_traj_ids"] = dict(model.config.traj_ids)
    write_json(out, payload)

    inputs = _step(payload, "prepare_model_inputs", out,
                   lambda: helper.prepare_model_inputs(data, model.config, model.tokenizer))
    inputs = helper.to_device(inputs, "cuda")

    recorder = GenerateRecorder(model.vlm.generate)
    model.vlm.generate = recorder
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()

    seeds = [args.seed + k for k in range(args.n_runs)]
    if args.repro_check:
        seeds.append(args.seed)
    for k, seed in enumerate(seeds):
        def run(seed=seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            kw: dict[str, Any] = {"diffusion_kwargs": {"inference_step": args.diffusion_steps}}
            if args.max_generation_length is not None:
                kw["max_generation_length"] = args.max_generation_length
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _rot, _lp, extra = model.sample_trajectories_from_data(
                    data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                    return_extra=True, **kw)
            return pred_xyz, extra

        n_before = len(recorder.calls)
        pred_xyz, extra = _step(payload, f"run_{k}_seed_{seed}", out, run)
        call = recorder.calls[n_before]
        rep = _analyze_call(call, tok, marker_ids, args.first_n)
        if k == 0:
            payload["prompt"] = _prompt_report(call["prompt_ids"], tok)
        xy = pred_xyz[0, 0, 0, :, :2].float().cpu().numpy()
        rep.update({
            "seed": seed,
            "release_extraction": {key: np.asarray(v).reshape(-1).tolist() for key, v in extra.items()
                                   if key != "raw_outputs"},
            "pred_xy": xy.tolist(),
            "ade_vs_logged_future_m": float(np.linalg.norm(xy - gt_xy, axis=1).mean()),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        })
        payload["runs"].append(rep)
        write_json(out, payload)
        c = rep["cot_start"]
        print(f"[run {k} seed {seed}] {c['statement']} | first tokens: "
              f"{[t['token'] for t in rep['first_tokens']]}")
        print(f"    text: {rep['text_before_traj_future_start'][:160]!r}")
        print(f"    meta-actions: {rep['meta_actions']}")

    if args.repro_check:
        a, b = payload["runs"][0]["generated_ids"], payload["runs"][-1]["generated_ids"]
        payload["reproducibility"] = {"seed": args.seed, "identical_generated_ids": a == b,
                                      "len_a": len(a), "len_b": len(b)}

    first = payload["runs"][0]
    payload["summary"] = {
        "cot_start_verdict": first["cot_start"]["verdict"],
        "cot_start_statement": first["cot_start"]["statement"],
        "verdicts_all_runs": [r["cot_start"]["verdict"] for r in payload["runs"]],
        "cot_for_training": first["cot_for_training"],
        "cot_source": first["cot_source"],
        "meta_actions_found": first["meta_actions"],
    }
    write_json(out, payload)

    if not args.skip_meta_action_task:
        from alpamayo2_super.text_tasks import generate_text, prepare_text_generation_inputs

        def meta_task():
            mdata = select_task_input(source, "meta_action")
            ti = prepare_text_generation_inputs(data=mdata, model_config=model.config,
                                                tokenizer=model.tokenizer, task="meta_action")
            ti = helper.to_device(ti, "cuda")
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = generate_text(model, ti, top_p=0.98, temperature=0.6, max_new_tokens=512)
            return res

        n_before = len(recorder.calls)
        try:
            res = _step(payload, "meta_action_task", out, meta_task)
            call = recorder.calls[n_before]
            rep = _analyze_call(call, tok, marker_ids, args.first_n)
            rep["prompt"] = _prompt_report(call["prompt_ids"], tok)
            rep["release_extraction"] = {k: list(v) for k, v in res.items()
                                         if k in ("cot", "meta_action")}
            payload["meta_action_task"] = rep
            payload["summary"]["meta_action_task_meta_actions"] = rep["meta_actions"]
            print(f"[meta_action task] {rep['cot_start']['statement']} | meta-actions: {rep['meta_actions']}")
        except Exception:  # noqa: BLE001 — block 1's main answer is already saved
            pass
        write_json(out, payload)

    print(f"\n>>> {payload['summary']['cot_start_statement']}")
    print(f"Saved -> {out}")
    return payload


if __name__ == "__main__":
    main()
