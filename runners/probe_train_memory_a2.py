#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_train_memory_a2.py — one stage-1 training step of Alpamayo 2 Super on ONE GPU, through the
official recipe path (gpu_session_01, block 2; unknowns 5, 6, 8; D-009, D-014).

Training path = NVlabs/alpamayo-recipes recipes/alpamayo2_sft (#39), used as is:
    TrainableAlpamayo2Super.from_pretrained(training_stage="vlm")      models/trainable_alpamayo2_super.py
    Alpamayo2SuperProcessor.preprocess(...)  -> build_conversation      processor.py
    Alpamayo2SuperCollator (assistant-span label mask, fixed padding)   processor.py, label_mask.py
    model(tokenized_data, traj_data, labels_mask) -> loss, future_traj_ce, others_ce
The sample comes from the recipe's PhysicalAIDataset when --pai-dir is given; otherwise from the
same release loader the dataset calls (load_physical_aiavdataset, streamed), with the dataset's
own post-processing (ego tensors squeezed).

Where the CoC comes from (probe_common.choose_cot, recorded under training_path.cot):
  1. --cot-text if given;
  2. otherwise the CoC the model itself generated in block 1 (summary.cot_for_training of
     the block-1 JSON) — a real CoC, the normal case;
  3. otherwise, and only then, a placeholder built from real CoC sentences, whose length in
     tokens is the maximum of a real distribution: block 1's CoCs if it produced any, else the
     Alpamayo 2 Super CoCs of the afh fixtures (--cot-reference). The distribution, the
     statistic and the measured token length are written in the JSON.

The real sequence length (before padding) is measured right after preprocessing. Above the
recipe's max_length (6 144) the probe raises SequenceTooLong and stops: it never truncates,
and it does not shrink the input to make a too-long sequence fit. After collation it checks
that the padded batch still holds every real token.

GPU: block 2 needs an H200 (141 GB) or larger; the probe refuses a GPU under 120 GiB unless
--allow-small-gpu is passed, and records it (such a run does not count for D-014).

Three deliberate departures from the recipe, all stated in the JSON:
  1. one GPU, no DeepSpeed: the 34B model sits whole on cuda:0 (the recipe shards it with ZeRO-3
     over 32 GPUs; nothing here measures that);
  2. gradients only on q_proj of the last 4 LM layers (a LoRA-sized proxy, D-004), not the whole
     VLM that `training_stage="vlm"` unfreezes;
  3. `cot` is added to components_order (the recipe ships trajectory-only targets; --no-cot runs
     the shipped preprocessing instead).

Per configuration it reports: sequence length (padded and real), number and share of visual
tokens, CoC tokens, assistant-mask tokens and its first tokens (training format evidence for
D-008), losses, q_proj gradient norms, and torch.cuda.max_memory_allocated / memory_allocated
after load, after the forward and after the backward.

On OOM it frees memory and retries with fewer cameras, then fewer frames
(probe_common.FALLBACK_CONFIGS), and reports the configuration that passes. Any other error,
including a too-long sequence, stops the probe with its message.
The JSON is rewritten after every step, so a crash still leaves a measurement.

Usage (recipe venv, see docs/gpu_session_01.md block 0):
    python runners/probe_train_memory_a2.py \\
        --tokens-json outputs/session01/tokens_a2_0ea6fd88.json \\
        --out outputs/session01/train_memory_a2.json
    ... --sweep        # measure every configuration, not only the first that passes
    ... --no-cot       # the recipe's shipped trajectory-only targets
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_common import (  # noqa: E402
    DEFAULT_CLIP,
    DEFAULT_MODEL_ID,
    DEFAULT_T0_US,
    FALLBACK_CONFIGS,
    InputConfig,
    check_sequence_length,
    choose_cot,
    count_span,
    package_provenance,
    reference_cots,
    run_with_fallback,
    select_last_layer_modules,
    write_json,
)

COMPONENTS_WITH_COT = ["image", "traj_history", "prompt", "cot", "traj_future"]
COMPONENTS_RECIPE = ["image", "traj_history", "prompt", "traj_future"]   # trajectory.yaml:2
RECIPE_MAX_LENGTH = 6144                                                 # sft_base.yaml:46
MIN_GPU_GIB = 120.0          # H200 = ~140 GiB usable; H100 80 GB (~79 GiB) is refused for block 2
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COT_REFERENCE = REPO_ROOT / ".deps/alpamayo-faithfulness/fixtures/records_a2.json"


def gib(x: int) -> float:
    return round(x / 2**30, 3)


def read_block1(tokens_json: str | None) -> tuple[dict | None, str]:
    """Block-1 JSON if readable, with a note on why not otherwise."""
    if not tokens_json:
        return None, "no --tokens-json"
    try:
        with open(tokens_json) as fh:
            return json.load(fh), tokens_json
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable {tokens_json}: {type(exc).__name__}"


def gpu_gate(total_gib: float, allow_small: bool) -> dict:
    """Block 2 needs >= MIN_GPU_GIB; below that, refuse unless explicitly overridden."""
    ok = total_gib >= MIN_GPU_GIB
    if not ok and not allow_small:
        raise SystemExit(f"block 2 requires an H200 (>= {MIN_GPU_GIB:.0f} GiB); this GPU has "
                         f"{total_gib:.1f} GiB. Run blocks 1, 3, 4 here and block 2 on an H200, "
                         "or pass --allow-small-gpu (the run will not count for D-014).")
    return {"min_gib": MIN_GPU_GIB, "total_gib": total_gib, "meets_requirement": ok,
            "overridden": (not ok) and allow_small}


def load_source_sample(args) -> tuple[dict, str]:
    """7-camera x 4-frame sample, exactly as the recipe dataset returns it (before preprocess)."""
    if args.pai_dir:
        from alpamayo2_sft.dataset import PhysicalAIDataset
        ds = PhysicalAIDataset(local_dir=args.pai_dir, chunk_ids=args.chunk_ids,
                               preprocess=None, use_default_keyframe=True, num_frames=4,
                               camera_features=None, include_calibration=False)
        idx = ds.clip_ids.index(args.clip) if args.clip in ds.clip_ids else 0
        return ds[idx], f"alpamayo2_sft.dataset.PhysicalAIDataset[{idx}] ({ds.clip_ids[idx]})"
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    sample = load_physical_aiavdataset(args.clip, t0_us=args.t0_us, num_frames=4,
                                       include_calibration=False)
    for key in tuple(sample):                     # alpamayo2_sft/dataset.py:85-87
        if key.startswith("ego_"):
            sample[key] = sample[key].squeeze(0)
    return sample, "alpamayo2_super.load_physical_aiavdataset (streamed; same loader as the recipe dataset)"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    ap.add_argument("--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--tokens-json", default="outputs/session01/tokens_a2_0ea6fd88.json",
                    help="block 1 output; its summary.cot_for_training is the example's CoC")
    ap.add_argument("--cot-text", default=None, help="override the CoC text")
    ap.add_argument("--cot-reference", default=str(DEFAULT_COT_REFERENCE),
                    help="afh fixture with real A2 CoCs, used only if block 1 gave no CoC")
    ap.add_argument("--allow-small-gpu", action="store_true",
                    help="run under 120 GiB anyway (recorded; does not count for D-014)")
    ap.add_argument("--no-cot", action="store_true", help="recipe's shipped trajectory-only targets")
    ap.add_argument("--pai-dir", default=None, help="local PAI root: use the recipe's PhysicalAIDataset")
    ap.add_argument("--chunk-ids", default=None, help="with --pai-dir, e.g. '0-1'")
    ap.add_argument("--n-last-layers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=RECIPE_MAX_LENGTH)
    ap.add_argument("--no-fixed-padding", action="store_true",
                    help="pad to the sample length instead of the recipe's fixed max_length")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="try every configuration")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of configuration names (default: all, in order)")
    ap.add_argument("--out", default="outputs/session01/train_memory_a2.json")
    args = ap.parse_args(argv)
    out = args.out

    components = COMPONENTS_RECIPE if args.no_cot else COMPONENTS_WITH_COT
    block1, block1_note = (None, "not used (--no-cot)") if args.no_cot else read_block1(args.tokens_json)

    payload: dict[str, Any] = {
        "probe": "probe_train_memory_a2", "clip_id": args.clip, "t0_us": args.t0_us,
        "model_id": args.model_id, "args": vars(args),
        "training_path": {
            "model": "alpamayo2_sft.models.trainable_alpamayo2_super.TrainableAlpamayo2Super (training_stage=vlm)",
            "preprocess": "alpamayo2_sft.processor.Alpamayo2SuperProcessor.preprocess",
            "collator": "alpamayo2_sft.processor.Alpamayo2SuperCollator",
            "components_order": components, "block1_json": block1_note,
            "cot": None,          # filled once the tokenizer is loaded (probe_common.choose_cot)
            "departures_from_recipe": [
                "single GPU, no DeepSpeed ZeRO-3 (recipe: 4 x 8 H100)",
                f"gradients only on q_proj of the last {args.n_last_layers} LM layers (recipe: whole VLM)",
                "cot added to components_order" if not args.no_cot else "none on targets",
            ],
        },
        "steps": {},
    }
    write_json(out, payload)

    import torch
    import transformers

    payload["env"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "alpamayo2_super": package_provenance("alpamayo2_super"),
        "alpamayo2_sft": package_provenance("alpamayo2-sft"),
        "alpamayo_recipes": package_provenance("alpamayo-recipes"),
    }
    if not torch.cuda.is_available():
        payload["steps"]["env"] = {"status": "error", "error": "no CUDA device"}
        write_json(out, payload)
        raise SystemExit("CUDA GPU required")
    prop = torch.cuda.get_device_properties(0)
    payload["env"]["gpu"] = {"name": prop.name, "total_memory_gib": gib(prop.total_memory)}
    try:
        payload["env"]["gpu_requirement"] = gpu_gate(gib(prop.total_memory), args.allow_small_gpu)
    except SystemExit as exc:
        payload["steps"]["gpu_requirement"] = {"status": "refused", "error": str(exc)}
        write_json(out, payload)
        raise

    from alpamayo2_sft.models.trainable_alpamayo2_super import TrainableAlpamayo2Super
    from alpamayo2_sft.processor import Alpamayo2SuperCollator, Alpamayo2SuperProcessor
    from alpamayo2_super.input_profiles import InputProfile, select_input_profile

    # ---- load (once; a failure here leaves no attempt to make)
    t = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        model = TrainableAlpamayo2Super.from_pretrained(
            args.model_id, dtype=torch.bfloat16, training_stage="vlm", device_map="cuda:0")
        payload["steps"]["load_model"] = {
            "status": "ok", "seconds": round(time.time() - t, 1),
            "memory_allocated_gib": gib(torch.cuda.memory_allocated()),
            "max_memory_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        }
    except Exception as exc:  # noqa: BLE001
        payload["steps"]["load_model"] = {"status": "oom" if "out of memory" in str(exc).lower() else "error",
                                          "seconds": round(time.time() - t, 1),
                                          "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
        write_json(out, payload)
        raise SystemExit(f"model load failed: {exc}")
    write_json(out, payload)

    # ---- trainable set: q_proj of the last N LM layers only
    for p in model.parameters():
        p.requires_grad_(False)
    names = select_last_layer_modules([n for n, _ in model.named_modules()],
                                      n_last=args.n_last_layers, proj="q_proj")
    modules = dict(model.named_modules())
    trainable = []
    for n in names:
        for pn, p in modules[n].named_parameters():
            p.requires_grad_(True)
            trainable.append((f"{n}.{pn}", p))
    payload["trainable"] = {"modules": names,
                            "n_params": int(sum(p.numel() for _, p in trainable))}
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable({"use_reentrant": False})   # recipe: sft_base.yaml:84-86
    model.train()

    # ---- data (once, 7 cameras x 4 frames)
    t = time.time()
    try:
        source, source_desc = load_source_sample(args)
        processor = Alpamayo2SuperProcessor(model.config)
        if not args.no_cot:
            tk = processor.processor.tokenizer
            refs = reference_cots(args.cot_reference) if os.path.isfile(args.cot_reference) else []
            cot_info = choose_cot(args.cot_text, block1, refs,
                                  encode=lambda t: tk.encode(t, add_special_tokens=False),
                                  decode=lambda ids: tk.decode(list(ids), skip_special_tokens=False))
            cot_info["reference_file"] = args.cot_reference if refs else None
            payload["training_path"]["cot"] = cot_info
            source["cot"] = cot_info["text"]
            print(f"[cot] {cot_info['kind']} ({cot_info['source']}), {cot_info['tokens']} tokens: "
                  f"{cot_info['text'][:100]!r}")
        collator = Alpamayo2SuperCollator(model.config, max_length=args.max_length,
                                          pad_to_fixed_length=not args.no_fixed_padding)
        payload["steps"]["load_data"] = {"status": "ok", "seconds": round(time.time() - t, 1),
                                         "source": source_desc}
    except Exception as exc:  # noqa: BLE001
        payload["steps"]["load_data"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
        write_json(out, payload)
        raise SystemExit(f"data load failed: {exc}")
    write_json(out, payload)

    tok = processor.processor.tokenizer
    ids = {k: tok.convert_tokens_to_ids(v) for k, v in {
        "image_pad": "<|image_pad|>", "cot_start": "<|cot_start|>", "cot_end": "<|cot_end|>",
        "traj_future_start": "<|traj_future_start|>"}.items()}
    image_token_id = getattr(model.config.vlm_config, "image_token_id", None) or ids["image_pad"]

    def to_cuda(x):
        if torch.is_tensor(x):
            return x.to("cuda")
        if isinstance(x, dict):
            return {k: to_cuda(v) for k, v in x.items()}
        return x

    def attempt(cfg: InputConfig, steps: dict) -> None:
        steps["_current"] = "preprocess"
        sample = select_input_profile(source, InputProfile(cfg.camera_ids, cfg.frame_indices))
        sample["tokenized_data"] = processor.preprocess(
            data=sample, components_order=components, components_prompt=components[3:],
            label_components=components[3:], generation_mode=False)
        unpadded = sample["tokenized_data"]["input_ids"][0]
        steps["sequence_before_padding"] = {
            "real_len": int(unpadded.shape[0]), "max_length": args.max_length,
            "visual_tokens": int((unpadded == image_token_id).sum()),
        }
        steps["_current"] = "length_check"
        check_sequence_length(int(unpadded.shape[0]), args.max_length)   # no truncation, ever
        steps["_current"] = "collate"
        batch = collator([sample])
        input_ids = batch["tokenized_data"]["input_ids"][0]
        attn = batch["tokenized_data"]["attention_mask"][0]
        mask = batch["labels_mask"][0]
        real = input_ids[attn.bool()].tolist()
        if real != unpadded.tolist():
            raise RuntimeError(f"collated batch lost tokens: {len(real)} real tokens after padding, "
                               f"{int(unpadded.shape[0])} before (truncation or reordering)")
        n_visual = int((input_ids == image_token_id).sum())
        assistant_ids = input_ids[mask].tolist()
        steps["sequence"] = {
            "padded_len": int(input_ids.shape[0]), "real_len": len(real),
            "visual_tokens": n_visual, "visual_share_of_real": round(n_visual / max(1, len(real)), 4),
            "cot_tokens_between_tags": count_span(real, ids["cot_start"], ids["cot_end"]),
            "assistant_mask_tokens": len(assistant_ids),
            "assistant_first_tokens": tok.convert_ids_to_tokens(assistant_ids[:20]),
            "assistant_text_head": tok.decode(assistant_ids[:60], skip_special_tokens=False),
            "pixel_values_shape": list(batch["tokenized_data"]["pixel_values"].shape),
            "image_grid_thw": batch["tokenized_data"]["image_grid_thw"].tolist(),
        }
        batch = to_cuda(batch)

        steps["_current"] = "forward"
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(tokenized_data=batch["tokenized_data"], traj_data=batch["traj_data"],
                           labels_mask=batch["labels_mask"])
        torch.cuda.synchronize()
        steps["forward"] = {
            "max_memory_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "memory_allocated_gib": gib(torch.cuda.memory_allocated()),
            "loss": float(output.loss), "future_traj_ce": float(output.future_traj_ce),
            "others_ce": float(output.others_ce),
        }

        steps["_current"] = "backward"
        torch.cuda.reset_peak_memory_stats()
        output.loss.backward()
        torch.cuda.synchronize()
        steps["backward"] = {
            "max_memory_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "memory_allocated_gib": gib(torch.cuda.memory_allocated()),
            "grad_norms": {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in trainable},
        }
        del output, batch

    def cleanup() -> None:
        for _, p in trainable:
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()

    configs = FALLBACK_CONFIGS
    if args.configs:
        wanted = args.configs.split(",")
        configs = tuple(c for c in FALLBACK_CONFIGS if c.name in wanted)

    def on_update(attempts):
        payload["attempts"] = attempts
        write_json(out, payload)

    result = run_with_fallback(configs, attempt, on_update=on_update, cleanup=cleanup, sweep=args.sweep)
    payload.update(result)
    payload["summary"] = summarize(result, payload["steps"]["load_model"])
    write_json(out, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved -> {out}")
    return payload


def summarize(result: dict, load_step: dict) -> dict:
    """One line per attempt plus the retained configuration (pure; tested cold)."""
    rows = []
    for a in result["attempts"]:
        s = a["steps"]
        rows.append({
            "config": a["config"]["name"], "status": a["status"], "failed_step": a["failed_step"],
            "real_len": ((s.get("sequence_before_padding") or {}).get("real_len")
                         or (s.get("sequence") or {}).get("real_len")),
            "visual_share": (s.get("sequence") or {}).get("visual_share_of_real"),
            "peak_forward_gib": (s.get("forward") or {}).get("max_memory_allocated_gib"),
            "peak_backward_gib": (s.get("backward") or {}).get("max_memory_allocated_gib"),
            "error": a["error"],
        })
    return {"weights_gib": load_step.get("memory_allocated_gib"),
            "retained_config": result["retained_config"],
            "stopped_on_error": result["stopped_on_error"], "attempts": rows}


if __name__ == "__main__":
    main()
