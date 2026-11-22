#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_train_memory_edge.py — one joint training step of Alpamayo 2 Edge on one GPU
(gpu_session_02, block 4)

Question (D-004, suspended): does full joint fine-tuning, as the official recipe does it,
fit on one 80 GB GPU with the native input (6 cameras, 22 images)? If yes, W1 needs no LoRA.

Path: the recipe's own classes, nothing rewritten.
    model     = hydra instantiate of recipes/alpamayo2edge_sft/configs/model/release.yaml
                (TrainableAlpamayo2Edge.from_pretrained, training_stage joint, FP32 weights,
                 SDPA, loss weights 1.0 / 1.0, token weights 1.0 / 0.2), as train_hf.py:41
    tokens    = Alpamayo2EdgeProcessor.preprocess with configs/preprocess/nav_cot.yaml
                (image, traj_history, nav_instruction, prompt | cot, traj_future; whole
                 assistant span supervised)
    batch     = Alpamayo2EdgeCollator(max_length 8192, padding to the longest sequence)
    step      = forward under BF16 autocast, backward, gradient clipping at 1.0, fused AdamW
                step on FP32 weights (configs/sft_base.yaml:88-114), no DeepSpeed, no
                gradient checkpointing (forbidden in joint training, models/sft.py:344-355)

The example is one annotated event of the recipe's demo asset: its 22 images and history
from the inference loader, its navigation and reasoning from the asset, its future poses
from the dataset (probe_edge_common.load_future_poses). It is assembled here because the
recipe's dataset class needs a local PAI download; preprocessing, collation and the forward
are the recipe's.

Ground truth stays out of the JSON: the future poses feed the loss and nothing else (no
pose value is written, and the cached poses live in the sample cache, which is not part of
the session archive). The losses are numbers computed against that ground truth; they are
written because a non-finite loss must be visible, and docs/gpu_session_02.md lists them
as not publishable.

Measured: real sequence length (before any padding), visual tokens against the cold
prediction, supervised tokens, losses, gradient norm, peak memory after forward, backward
and optimizer step, seconds per step.

Fallback, same logic as the 2 Super probe (probe_common.run_with_fallback): on OOM or on a
sequence above max_length the reason is recorded (a too-long sequence is a finding, with its
measured length), memory is freed, and the next input is tried — fewer images, then the
recipe's debug input (front camera, one frame). Nothing is ever truncated. The configuration
that finally fits is marked reduced, with the reasons. Any other error stops the probe.

The optimizer steps use the recipe's learning rate on the loaded model; the model is not
saved. Environment: the recipe venv. See docs/gpu_session_02.md.

Usage:
    python runners/probe_train_memory_edge.py --events $EVENTS --model-dir $EDGE_MODEL \\
        --out outputs/session02/train_memory_edge.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_common import check_sequence_length, run_with_fallback  # noqa: E402
from runners.probe_edge_common import (  # noqa: E402
    EDGE_FALLBACK_CONFIGS,
    EVENTS_ASSET,
    PAI_REVISION,
    PINS,
    EdgeInputConfig,
    env_report,
    gpu_gate,
    load_events,
    load_future_poses,
    load_sample,
    make_dataset_interface,
    predict_visual_tokens,
    sample_report,
    select_positions,
    step,
    write_probe_json,
)

MIN_GPU_GIB = 75.0          # an "80 GB" card reports about 79 to 80 GiB
MAX_LENGTH = 8192           # recipes/alpamayo2edge_sft/configs/sft_base.yaml:62
# recipes/alpamayo2edge_sft/configs/preprocess/nav_cot.yaml:2-4 (read live; this copy is
# what the pinned file contains and is compared with it)
NATIVE_PREPROCESS = {
    "components_order": ["image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"],
    "components_prompt": ["cot", "traj_future"],
    "label_components": ["assistant"],
}
# recipes/alpamayo2edge_sft/configs/sft_base.yaml:93-108
OPTIMIZER = {"lr": 3.5e-6, "betas": (0.9, 0.95), "eps": 1.0e-8, "weight_decay": 0.1}
MAX_GRAD_NORM = 1.0


def summarize(result: dict, weights_gib: float | None, gpu_total_gib: float | None) -> dict:
    """One line per attempt, the retained configuration and the answer to D-004's question
    (pure; tested cold)."""
    rows = []
    for a in result["attempts"]:
        s = a["steps"]
        seq = s.get("sequence_before_padding") or {}
        train = s.get("train_steps") or []
        rows.append({
            "config": a["config"]["name"], "n_images": (s.get("images") or {}).get("n_images"),
            "status": a["status"], "failed_step": a["failed_step"],
            "reduced": a.get("reduced", False), "reduced_because": a.get("reduced_because", []),
            "real_len": seq.get("real_len"), "visual_tokens": seq.get("visual_tokens"),
            "predicted_visual_tokens": (s.get("images") or {}).get("predicted_visual_tokens"),
            "steps_completed": len([t for t in train if t.get("optimizer_seconds") is not None]),
            "peak_memory_gib": s.get("peak_memory_gib"),
            "seconds_per_step": s.get("seconds_per_step"),
            "loss_first_step": train[0].get("loss") if train else None,
            "error": a["error"],
        })
    native = EDGE_FALLBACK_CONFIGS[0].name
    retained = result["retained_config"]
    return {
        "weights_gib": weights_gib, "gpu_total_gib": gpu_total_gib,
        "retained_config": retained,
        "retained_reduced": result.get("retained_reduced", False),
        "reduced_because": result.get("reduced_because", []),
        "abandoned": result.get("abandoned", []),
        "findings": result.get("findings", []),
        "stopped_on_error": result["stopped_on_error"],
        "native_full_finetune_fits": bool(retained == native),
        "native_config": native,
        "attempts": rows,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help=f"annotation JSON; the recipe's {EVENTS_ASSET}")
    ap.add_argument("--event-index", type=int, default=0, help="row of --events used as the example")
    ap.add_argument("--model-dir", default=os.environ.get("EDGE_MODEL"),
                    help="local model directory ($EDGE_MODEL); the tokenizer is read from it")
    ap.add_argument("--pai-revision", default=PAI_REVISION)
    ap.add_argument("--sample-cache", default="outputs/session02/samples")
    ap.add_argument("--cot-text", default=None, help="reasoning target; default: the event's own `cot`")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--steps", type=int, default=3, help="optimizer steps per configuration")
    ap.add_argument("--sweep", action="store_true", help="measure every configuration, not only the first that fits")
    ap.add_argument("--min-gpu-gib", type=float, default=MIN_GPU_GIB)
    ap.add_argument("--allow-small-gpu", action="store_true",
                    help="run below --min-gpu-gib; recorded in the JSON")
    ap.add_argument("--out", default="outputs/session02/train_memory_edge.json")
    args = ap.parse_args(argv)
    if not args.model_dir or not os.path.isdir(args.model_dir):
        ap.error("--model-dir (or $EDGE_MODEL) must be a local model directory")
    if args.steps < 1:
        ap.error("--steps must be at least 1")

    payload: dict[str, Any] = {"probe": "probe_train_memory_edge", "pins": PINS, "args": vars(args),
                               "steps": {}, "optimizer": {**OPTIMIZER, "fused": True,
                                                          "max_grad_norm": MAX_GRAD_NORM}}
    out = args.out

    import torch

    payload["env"] = step(payload, "env", out, env_report)
    total = (payload["env"].get("gpu") or {}).get("total_memory_gib")
    gate = gpu_gate(total, args.min_gpu_gib, args.allow_small_gpu)
    payload["env"]["gpu_requirement"] = gate
    write_probe_json(out, payload)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if not gate["proceed"]:
        raise SystemExit(f"GPU has {total} GiB, block 4 asks for {args.min_gpu_gib} GiB "
                         "(--allow-small-gpu to run anyway; it is recorded)")

    # ---- the example
    event = step(payload, "load_event", out,
                 lambda: load_events(args.events, str(args.event_index))[0])
    cot = (args.cot_text or event.cot or "").strip()
    if not cot:
        raise SystemExit("the event has no `cot` and --cot-text is empty: no reasoning target")
    avdi = step(payload, "dataset_interface", out, lambda: make_dataset_interface(args.pai_revision))
    sample, source = step(payload, "load_sample", out, lambda: load_sample(event, avdi, args.sample_cache))
    future, future_source = step(payload, "load_future_poses", out,
                                 lambda: load_future_poses(event, avdi, args.sample_cache))
    height, width = (int(x) for x in sample["image_frames"].shape[-2:])
    payload["example"] = {
        "event_index": event.index, "clip_id": event.clip_id, "t0_us": event.t0_us,
        "nav_text": event.nav_text,
        "cot": {"text": cot, "words": len(cot.split()),
                "source": "--cot-text" if args.cot_text else "event `cot` field of --events "
                          "(demo asset: generated by Alpamayo 2 Edge, reviewed)"},
        "sample_source": source, **sample_report(sample),
        # the target trajectory is dataset ground truth: its values stay out of this JSON
        "target_trajectory": {"n_waypoints": int(future["ego_future_xyz"].shape[1]),
                              "source": future_source, "values_written": False},
    }
    write_probe_json(out, payload)

    # ---- the recipe's model, processor and collator
    from importlib.resources import files

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from alpamayo2edge_sft.processor import Alpamayo2EdgeCollator

    recipe_configs = files("alpamayo2edge_sft") / "configs"
    model_cfg = OmegaConf.load(str(recipe_configs / "model" / "release.yaml"))
    model_cfg.pretrained_model_name_or_path = args.model_dir
    preprocess_cfg = OmegaConf.to_container(OmegaConf.load(str(recipe_configs / "preprocess" / "nav_cot.yaml")))
    components = {k: preprocess_cfg[k] for k in NATIVE_PREPROCESS}
    payload["recipe"] = {
        "model_config": OmegaConf.to_container(model_cfg),
        "preprocess": preprocess_cfg,
        "preprocess_matches_pinned_copy": components == NATIVE_PREPROCESS,
    }

    def load_model():
        m = instantiate(model_cfg, _convert_="partial")       # train_hf.py:41
        m.to("cuda")
        m.train()
        return m

    model = step(payload, "load_model", out, load_model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    weights_gib = round(torch.cuda.memory_allocated() / 2**30, 2)
    payload["model"] = {
        "training_stage": getattr(model, "training_stage", None),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable": sum(p.numel() for p in trainable),
        "trainable_dtypes": sorted({str(p.dtype) for p in trainable}),
        "gradient_checkpointing": bool(getattr(model.vlm, "is_gradient_checkpointing", False)),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "memory_after_load_gib": weights_gib,
    }
    collator = step(payload, "collator", out, lambda: Alpamayo2EdgeCollator(
        model_config=model.config, max_length=args.max_length, pad_to_fixed_length=False))
    processor = collator.data_processor
    tok = processor.processor.tokenizer
    image_token_id = int(model.config.vlm_config.image_token_id)
    future_pad = int(model.config.traj_ids["future_pad"])
    payload["example"]["cot"]["tokens"] = len(tok.encode(cot, add_special_tokens=False))
    write_probe_json(out, payload)

    cams = sample["camera_indices"]
    stamps = sample["absolute_timestamps"].reshape(-1)
    state: dict[str, Any] = {}

    def to_cuda(x):
        if torch.is_tensor(x):                       # floats stay FP32 (trainer.py:149-166)
            return x.to("cuda", dtype=torch.float32 if torch.is_floating_point(x) else None)
        if isinstance(x, dict):
            return {k: to_cuda(v) for k, v in x.items()}
        return x

    def sync() -> float:
        torch.cuda.synchronize()
        return time.time()

    def peak() -> float:
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)

    def attempt(cfg: EdgeInputConfig, steps: dict) -> None:
        steps["_current"] = "select_images"
        pos = select_positions(cams.tolist(), stamps.tolist(), cfg)
        sub = dict(sample)
        sub.update(future)
        sub["image_frames"] = sample["image_frames"][pos].contiguous()
        sub["camera_indices"] = cams[pos]
        sub["absolute_timestamps"] = stamps[pos]
        sub["nav_text"] = [event.nav_text]
        sub["cot"] = cot
        steps["images"] = {"n_images": len(pos), "positions": pos,
                           "predicted_visual_tokens": predict_visual_tokens(height, width, cfg)}

        steps["_current"] = "preprocess"
        tokenized = processor.preprocess(data=sub, generation_mode=False, **components)
        unpadded = tokenized["input_ids"][0]
        steps["sequence_before_padding"] = {
            "real_len": int(unpadded.shape[0]), "max_length": args.max_length,
            "visual_tokens": int((unpadded == image_token_id).sum()),
        }
        steps["_current"] = "length_check"
        check_sequence_length(int(unpadded.shape[0]), args.max_length)     # no truncation, ever

        steps["_current"] = "collate"
        sub["tokenized_data"] = tokenized
        batch = collator([sub])
        ids = batch["tokenized_data"]["input_ids"][0]
        attn = batch["tokenized_data"]["attention_mask"][0]
        mask = batch["labels_mask"][0]
        real = ids[attn.bool()].tolist()
        if real != unpadded.tolist():
            raise RuntimeError(f"collated batch lost tokens: {len(real)} real tokens after collation, "
                               f"{int(unpadded.shape[0])} before (truncation or reordering)")
        supervised = ids[mask].tolist()
        n_visual = int((ids == image_token_id).sum())
        steps["sequence"] = {
            "padded_len": int(ids.shape[0]), "real_len": len(real),
            "visual_tokens": n_visual, "visual_share_of_real": round(n_visual / max(1, len(real)), 4),
            "supervised_tokens": len(supervised),
            "future_placeholders": int((ids == future_pad).sum()),
            "supervised_first_tokens": tok.convert_ids_to_tokens(supervised[:24]),
            "supervised_text_head": tok.decode(supervised[:48], skip_special_tokens=False),
            "supervised_last_tokens": tok.convert_ids_to_tokens(supervised[-3:]),
        }

        steps["_current"] = "to_device"
        batch = to_cuda(batch)
        state["optimizer"] = torch.optim.AdamW(trainable, fused=True, **OPTIMIZER)
        torch.cuda.reset_peak_memory_stats()
        steps["train_steps"] = []
        try:
            _train(batch, steps)
        finally:
            # also on failure: the peak reached before an OOM is a measurement
            steps["peak_memory_gib"] = peak()

    def _train(batch: dict, steps: dict) -> None:
        for k in range(args.steps):
            rep: dict[str, Any] = {"step": k}
            steps["train_steps"].append(rep)
            steps["_current"] = f"step{k}_forward"
            t0 = sync()
            with torch.autocast("cuda", dtype=torch.bfloat16):          # trainer.bf16: true
                output = model(tokenized_data=batch["tokenized_data"],
                               traj_data=batch["traj_data"], labels_mask=batch["labels_mask"])
            loss = output.loss
            t1 = sync()
            rep.update(forward_seconds=round(t1 - t0, 3), peak_after_forward_gib=peak(),
                       loss=float(loss.detach().float()),
                       **{name: float(getattr(output, name).detach().float())
                          for name in ("vlm_ce", "future_traj_ce", "others_ce", "expert_diffusion")
                          if getattr(output, name, None) is not None})
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {k}: {rep['loss']}")
            steps["_current"] = f"step{k}_backward"
            loss.backward()
            t2 = sync()
            rep.update(backward_seconds=round(t2 - t1, 3), peak_after_backward_gib=peak())
            steps["_current"] = f"step{k}_optimizer"
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            state["optimizer"].step()
            state["optimizer"].zero_grad(set_to_none=True)
            t3 = sync()
            rep.update(grad_norm_before_clipping=float(grad_norm), optimizer_seconds=round(t3 - t2, 3),
                       step_seconds=round(t3 - t0, 3), peak_after_optimizer_gib=peak())
            del output, loss
        times = [r["step_seconds"] for r in steps["train_steps"]]
        # the first step pays one-off initialisation: report the later ones when there are any
        later = times[1:] or times
        steps["seconds_per_step"] = round(sorted(later)[len(later) // 2], 3)
        steps["seconds_first_step"] = times[0]

    def cleanup() -> None:
        state.pop("optimizer", None)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def on_update(attempts: list[dict]) -> None:
        payload["attempts"] = attempts
        write_probe_json(out, payload)

    result = run_with_fallback(EDGE_FALLBACK_CONFIGS, attempt, on_update=on_update,
                               cleanup=cleanup, sweep=args.sweep)
    payload["attempts"] = result["attempts"]
    payload["summary"] = summarize(result, weights_gib, total)
    write_probe_json(out, payload)

    s = payload["summary"]
    for row in s["attempts"]:
        print(f"[{row['config']}] {row['status']} real_len={row['real_len']} "
              f"peak={row['peak_memory_gib']} GiB step={row['seconds_per_step']} s "
              f"{row['error'] or ''}")
    print(f"\n>>> native full fine-tuning fits on this GPU ({total} GiB): {s['native_full_finetune_fits']}"
          f" | retained: {s['retained_config']}"
          + (f" (reduced: {s['reduced_because']})" if s["retained_reduced"] else ""))
    print(f"Saved -> {out}")
    return payload


if __name__ == "__main__":
    main()
