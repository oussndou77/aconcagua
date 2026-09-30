#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_blackout_a15.py — the blackout test on Alpamayo 1.5 (ROADMAP W3 precondition, D-003).

Reproduces, on Alpamayo 1.5, the experiment whose Alpamayo 2 Super result lives in
`fixtures/cf_0ea6fd88_a2_blackout.json` of oussndou77/alpamayo-faithfulness: K independent
rollouts on the original frames (baseline), then the same K rollouts with EVERY camera
black, scored with the harness's counterfactual axis and written in the same JSON layout.

The black frames come from `afh.degradation` (family "blackout", severity 1, all cameras),
so the probe uses exactly the degradation the uncertainty fine-tune will train on. The
harness also gives the target the model SHOULD produce for that spec (`target_text`,
`target_severity`); both are recorded next to what the model actually said.

Either outcome is informative (D-003): if 1.5 narrates "clear path ahead" with no image,
it shares the defect reported in NVlabs/alpamayo2#9 and W3 can start on 1.5; if it does
not, the defect is specific to 2 Super and W3 needs the 2 Super recipe first.

GPU only for the rollouts. Everything else (degradation, scoring, payload) is pure Python
and is covered by tests/test_probe_blackout_a15_cold.py.

Environment on the pod (see docs/finetune_design.md §1.5 and the 1.5 README):
    uv venv a1_5_venv --python 3.12 && source a1_5_venv/bin/activate
    uv sync --active [--no-install-package flash-attn]      # in a clone of NVlabs/alpamayo1.5
    pip install "alpamayo-faithfulness @ git+https://github.com/oussndou77/alpamayo-faithfulness.git"
    export HF_TOKEN=...   # approved access to nvidia/Alpamayo-1.5-10B and the PhysicalAI-AV dataset

Usage:
    python runners/probe_blackout_a15.py                                   # defaults below
    python runners/probe_blackout_a15.py --k-rollouts 5 --attn sdpa \
        --out outputs/cf_0ea6fd88_a15_blackout.json --dump-frames outputs/frames
    python runners/probe_blackout_a15.py --null-control                    # + noise floor

Design notes:
  * K INDEPENDENT rollouts (one seed each, num_traj_samples=1), never one call with
    num_traj_samples=K: a multi-sample call shares one Chain-of-Causation across the K
    trajectories (same lesson as runners/run_inference.py in the harness).
  * The 1.5 loader returns 4 cameras with camera ids [0, 1, 2, 6]; afh.degradation works on
    TENSOR indices, so `spec.cameras` are [0, 1, 2, 3] and the payload records the id
    mapping under "camera_indices". For a total blackout this changes nothing physically,
    but afh's camera names / front-camera floors assume the 7-camera order (see the design
    note, proposed D-010).
  * `--attn sdpa` is the default: the 1.5 README documents it as the flash-attn-free path,
    and the diffusion expert is forced to sdpa anyway. Pass `--attn flash_attention_2` on a
    pod where flash-attn is built.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Iterable

import numpy as np

from afh.axes.consistency import summarize_trajectory
from afh.axes.counterfactual import score_counterfactual
from afh.degradation import (
    DegradationSpec,
    apply_degradation,
    target_severity,
    target_text,
    uncertainty_score,
)
from afh.parser import parse_trace
from afh.trace import CoCTrace, TrajectorySummary

DEFAULT_CLIP = "0ea6fd88-dcdd-434e-9fa3-56ce0fb35bf2"
DEFAULT_T0_US = 5_100_000
DEFAULT_K = 5
DEFAULT_MODEL_ID = "nvidia/Alpamayo-1.5-10B"
TARGET_AGENT = "vehicle"          # the agent the A2 baseline cites on this clip (parked vehicle)
BLACKOUT_TRACK_ID = "BLACKOUT"    # the A2 fixture's marker for "no track, whole image black"
DT = 0.1                          # 64 waypoints @ 10 Hz

# Keys of fixtures/cf_0ea6fd88_a2_blackout.json (alpamayo-faithfulness), in order. The payload
# written here starts with exactly these keys so the two files are comparable field by field.
FIXTURE_KEYS = (
    "clip_id", "target_agent", "track_id",
    "baseline_traces", "cf_traces",
    "baseline_behaviors", "cf_behaviors",
    "score", "verdict", "baseline_citation", "cf_citation",
)

# camera id -> name, as in the loaders of both NVlabs releases
CAMERA_ID_TO_NAME = {
    0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov", 3: "camera_rear_left_70fov",
    4: "camera_rear_tele_30fov", 5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}


# --------------------------------------------------------------------------- cold helpers

def blackout_spec(n_cam: int, seed: int = 0) -> DegradationSpec:
    """Family blackout, severity 1, every tensor index affected (the NVlabs/alpamayo2#9 case)."""
    return DegradationSpec(family="blackout", severity=1.0, cameras=list(range(n_cam)), seed=seed)


def blackout_frames(frames: np.ndarray, seed: int = 0) -> tuple[np.ndarray, DegradationSpec]:
    """
    Black out every camera of `frames` (n_cam, n_t, 3, H, W) uint8 through afh.degradation.

    Returns the degraded frames and the resolved spec (cameras and params filled). The
    result is asserted to be all-zero: the probe must never run on a partially black input.
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        raise TypeError(f"frames must be uint8 (loader output), got {frames.dtype}")
    if frames.ndim != 5:
        raise ValueError(f"frames must be (n_cam, n_t, 3, H, W), got shape {frames.shape}")
    degraded, spec = apply_degradation(frames, blackout_spec(frames.shape[0], seed=seed))
    if degraded.any():
        raise AssertionError("blackout left non-zero pixels; refusing to run a partial blackout")
    return degraded, spec


def flatten_reasoning(cot_entry: Any) -> list[str]:
    """extra['cot'] (numpy array / nested lists / str) -> ordered, de-duplicated non-empty strings."""
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, (str, bytes)):
            out.append(x.decode() if isinstance(x, bytes) else x)
        elif hasattr(x, "tolist"):
            walk(x.tolist())
        elif isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
        else:
            out.append(str(x))

    walk(cot_entry)
    seen: set[str] = set()
    res: list[str] = []
    for s in out:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def side_records(clip_id: str, sentences_per_rollout: Iterable[list[str]],
                 xys: Iterable[np.ndarray], dt: float = DT) -> tuple[list[CoCTrace], list[TrajectorySummary]]:
    """Turn K rollouts (reasoning sentences, (T, 2) waypoints) into harness traces and summaries."""
    traces, trajs = [], []
    for k, (sentences, xy) in enumerate(zip(sentences_per_rollout, xys)):
        claims = []
        for s in sentences:
            claims.extend(parse_trace(clip_id, k, s).claims)
        traces.append(CoCTrace(clip_id=clip_id, sample_index=k,
                               raw_text="\n".join(sentences), claims=claims))
        pts = [tuple(map(float, p[:2])) for p in np.asarray(xy, dtype=float)]
        trajs.append(summarize_trajectory(pts, dt=dt))
    return traces, trajs


def summarize_uncertainty(traces: list[CoCTrace]) -> dict:
    """
    Metric-A view of what the model said: per-rollout afh.degradation.uncertainty_score
    (-1 = no lexicon match), the share of rollouts that claim a clear road (level 0.0) and
    the share that admit having no usable input (level >= 0.95).
    """
    scores = [uncertainty_score(t.raw_text) for t in traces]
    n = max(1, len(scores))
    return {
        "uncertainty_scores": scores,
        "clear_path_rate": sum(1 for s in scores if s == 0.0) / n,
        "no_input_admitted_rate": sum(1 for s in scores if s >= 0.95) / n,
    }


def build_payload(clip_id: str, base_traces: list[CoCTrace], base_trajs: list[TrajectorySummary],
                  cf_traces: list[CoCTrace], cf_trajs: list[TrajectorySummary],
                  target_agent: str = TARGET_AGENT, extras: dict | None = None) -> dict:
    """
    Score baseline vs blackout with afh.axes.counterfactual and lay the result out exactly
    like fixtures/cf_0ea6fd88_a2_blackout.json (FIXTURE_KEYS first, then `extras`).
    """
    result = score_counterfactual(clip_id, target_agent, base_traces, base_trajs, cf_traces, cf_trajs)
    payload = {
        "clip_id": clip_id,
        "target_agent": target_agent,
        "track_id": BLACKOUT_TRACK_ID,
        "baseline_traces": [t.raw_text for t in base_traces],
        "cf_traces": [t.raw_text for t in cf_traces],
        "baseline_behaviors": [sorted(t.behaviors()) for t in base_trajs],
        "cf_behaviors": [sorted(t.behaviors()) for t in cf_trajs],
        "score": result.score,
        "verdict": result.verdict,
        "baseline_citation": result.baseline_citation,
        "cf_citation": result.cf_citation,
    }
    assert tuple(payload) == FIXTURE_KEYS
    payload["behavior_change"] = result.behavior_change
    payload["notes"] = list(result.notes)
    payload["baseline_uncertainty"] = summarize_uncertainty(base_traces)
    payload["cf_uncertainty"] = summarize_uncertainty(cf_traces)
    for k, v in (extras or {}).items():
        payload[k] = v
    return payload


def _to_xy(arr: Any) -> np.ndarray:
    """(..., T, >=2) tensor/array -> (T, 2) float ndarray, squeezing leading singleton dims."""
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    while a.ndim > 2:
        a = a[0]
    return a[:, :2].astype(float)


def _min_ade(xys: list[np.ndarray], gt_xy: np.ndarray) -> float | None:
    if not xys:
        return None
    return float(min(np.linalg.norm(np.asarray(xy) - gt_xy, axis=1).mean() for xy in xys))


# --------------------------------------------------------------------------- GPU side

def load_model(model_id: str, attn: str | None):
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if attn:
        kwargs["attn_implementation"] = attn
    model = Alpamayo1_5.from_pretrained(model_id, **kwargs).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor, helper


def run_side(model, processor, helper, data: dict, frames, k_rollouts: int, seed_offset: int = 0,
             top_p: float = 0.98, temperature: float = 0.6, max_gen: int = 256,
             diffusion_steps: int | None = None, label: str = "") -> tuple[list[list[str]], list[np.ndarray]]:
    """
    K independent Alpamayo 1.5 rollouts on `frames` (torch uint8 (n_cam, n_t, 3, H, W)).

    The 1.5 pipeline (test_inference.py): messages carry the image tensors and the camera
    ids, the processor tokenises them, and the model fuses the ego history itself.
    Returns the reasoning sentences and the (T, 2) waypoints of every rollout.
    """
    import torch

    messages = helper.create_message(frames.flatten(0, 1), camera_indices=data["camera_indices"])
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    base_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    kwargs: dict[str, Any] = {}
    if diffusion_steps is not None:
        kwargs["diffusion_kwargs"] = {"inference_step": diffusion_steps}

    sentences_per_rollout, xys = [], []
    for k in range(k_rollouts):
        seed = k + seed_offset
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mi = helper.to_device(copy.deepcopy(base_inputs), "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=top_p, temperature=temperature, num_traj_samples=1,
                max_generation_length=max_gen, return_extra=True, **kwargs)
        sentences = flatten_reasoning(extra["cot"]) if isinstance(extra, dict) and "cot" in extra else []
        xy = _to_xy(pred_xyz)
        sentences_per_rollout.append(sentences)
        xys.append(xy)
        print(f"  [{label}] rollout {k} (seed {seed}): "
              f"{sentences[0][:90] if sentences else '(empty reasoning)'}")
    return sentences_per_rollout, xys


def dump_frames(frames, camera_indices: list[int], out_dir: str, tag: str) -> None:
    """Save the last frame of every camera as PNG (verify the blackout before trusting numbers)."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    arr = frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)
    for i, cam_id in enumerate(camera_indices):
        img = np.transpose(arr[i, -1], (1, 2, 0)).astype(np.uint8)
        path = os.path.join(out_dir, f"{tag}_cam{cam_id}_{CAMERA_ID_TO_NAME.get(cam_id, 'unknown')}.png")
        Image.fromarray(img).save(path)
        print(f"  saved {path}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    ap.add_argument("--k-rollouts", type=int, default=DEFAULT_K)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--attn", default="sdpa",
                    help="attention implementation for the VLM: sdpa (default, no flash-attn build) "
                         "or flash_attention_2; 'none' keeps the checkpoint's setting")
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-generation-length", type=int, default=256)
    ap.add_argument("--diffusion-steps", type=int, default=None,
                    help="expert Euler steps (default: the checkpoint's, 10)")
    ap.add_argument("--target-agent", default=TARGET_AGENT)
    ap.add_argument("--null-control", action="store_true",
                    help="also run a second UNMASKED baseline with other seeds: the sampling-noise "
                         "floor any blackout effect must exceed")
    ap.add_argument("--dump-frames", default=None, metavar="DIR",
                    help="save the last frame of every camera (baseline and blackout) as PNG")
    ap.add_argument("--out", default="outputs/cf_0ea6fd88_a15_blackout.json")
    args = ap.parse_args(argv)

    import torch
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    print(f"=== Alpamayo 1.5 blackout probe: clip {args.clip} t0={args.t0_us} us K={args.k_rollouts}")
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]
    camera_indices = [int(c) for c in data["camera_indices"].tolist()]
    print(f"[cams] tensor order = {camera_indices} -> "
          f"{[CAMERA_ID_TO_NAME.get(c, '?') for c in camera_indices]}; frames {tuple(frames.shape)} {frames.dtype}")

    degraded_np, spec = blackout_frames(frames.cpu().numpy())
    black = torch.from_numpy(degraded_np)
    ts = target_severity(spec)
    expected_text = target_text(spec)
    print(f"[blackout] spec={spec.to_dict()}")
    print(f"[blackout] afh target severity {ts:.2f}; expected wording: {expected_text!r}")

    if args.dump_frames:
        dump_frames(frames, camera_indices, args.dump_frames, "baseline")
        dump_frames(black, camera_indices, args.dump_frames, "blackout")

    model, processor, helper = load_model(args.model_id, None if args.attn == "none" else args.attn)
    gt_xy = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]
    common = dict(top_p=args.top_p, temperature=args.temperature,
                  max_gen=args.max_generation_length, diffusion_steps=args.diffusion_steps)

    print("=== BASELINE (original frames) ===")
    base_sent, base_xy = run_side(model, processor, helper, data, frames, args.k_rollouts,
                                  label="baseline", **common)
    print("=== BLACKOUT (all cameras black) ===")
    cf_sent, cf_xy = run_side(model, processor, helper, data, black, args.k_rollouts,
                              label="blackout", **common)

    base_tr, base_tj = side_records(args.clip, base_sent, base_xy)
    cf_tr, cf_tj = side_records(args.clip, cf_sent, cf_xy)

    extras = {
        "model_id": args.model_id,
        "t0_us": args.t0_us,
        "k_rollouts": args.k_rollouts,
        "seeds": list(range(args.k_rollouts)),
        "attn": args.attn,
        "sampling": {"top_p": args.top_p, "temperature": args.temperature,
                     "max_generation_length": args.max_generation_length,
                     "diffusion_steps": args.diffusion_steps, "num_traj_samples": 1},
        "camera_indices": camera_indices,
        "camera_names": [CAMERA_ID_TO_NAME.get(c, "unknown") for c in camera_indices],
        "degradation": spec.to_dict(),
        "target_severity": ts,
        "target_text": expected_text,
        "baseline_xy": [xy.tolist() for xy in base_xy],
        "cf_xy": [xy.tolist() for xy in cf_xy],
        "baseline_min_ade": _min_ade(base_xy, gt_xy),
        "cf_min_ade": _min_ade(cf_xy, gt_xy),
    }
    payload = build_payload(args.clip, base_tr, base_tj, cf_tr, cf_tj,
                            target_agent=args.target_agent, extras=extras)

    if args.null_control:
        print("=== NULL CONTROL (original frames, other seeds) ===")
        null_sent, null_xy = run_side(model, processor, helper, data, frames, args.k_rollouts,
                                      seed_offset=1000, label="null", **common)
        null_tr, null_tj = side_records(args.clip, null_sent, null_xy)
        null_result = score_counterfactual(args.clip, args.target_agent, base_tr, base_tj, null_tr, null_tj)
        payload["null_control_traces"] = [t.raw_text for t in null_tr]
        payload["null_xy"] = [xy.tolist() for xy in null_xy]
        payload["null_control_behavior_change"] = null_result.behavior_change
        payload["exceeds_noise_floor"] = bool(payload["behavior_change"] > null_result.behavior_change)
        print(f">>> noise floor (baseline vs baseline): {null_result.behavior_change:.0%} | "
              f"blackout: {payload['behavior_change']:.0%}")

    print()
    print(score_counterfactual(args.clip, args.target_agent, base_tr, base_tj, cf_tr, cf_tj).format_report())
    cfu = payload["cf_uncertainty"]
    print(f">>> under blackout, {cfu['clear_path_rate']:.0%} of rollouts claim a clear road, "
          f"{cfu['no_input_admitted_rate']:.0%} admit having no usable input "
          f"(afh target: {expected_text!r})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {args.out}")
    return payload


if __name__ == "__main__":
    main()
