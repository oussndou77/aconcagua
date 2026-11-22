# SPDX-License-Identifier: Apache-2.0
"""
probe_edge_common.py — helpers shared by the Alpamayo 2 Edge probes of docs/gpu_session_02.md.

The top half is pure Python + numpy and is tested cold (tests/test_probes_edge_cold.py):

  * the 19 annotated events of the recipe's demo asset (clip, time, navigation, reasoning);
  * navigation counterfactuals in the recipe's validated format;
  * a description of a generated token sequence that presupposes no format: where the
    trajectory anchor is, how long the reasoning before it is (zero included), what else is
    there (<think> tags, EOS, discrete trajectory tokens, meta-action lines);
  * image selection by camera identifier (blackout, memory fallback): Edge feeds 22 flat
    images with one identifier per image, so nothing here assumes a camera x time tensor;
  * trajectory deviation between two conditions, with the within-condition spread as floor;
  * the expected number of visual tokens of a configuration (docs/edge_review.md §5).

The bottom half talks to the GPU and to the pinned packages; it imports torch lazily so the
module itself imports cold. Every call into alpamayo2_edge mirrors a call site of the pinned
inference repository (NVlabs/alpamayo2-edge 02f55a7), cited next to it.

Pins (D-016): alpamayo-recipes 622f9d2, alpamayo2-edge 02f55a7, nvidia/Alpamayo2-Edge
f8951d0, PhysicalAI-AV dataset 33f9bf4 (the recipe's).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from runners.probe_common import package_provenance, write_json

EDGE_REPO_COMMIT = "02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e"
RECIPES_COMMIT = "622f9d23ff9e4961ba8c6aac71367776cdf6b016"
HF_MODEL_ID = "nvidia/Alpamayo2-Edge"
HF_REVISION = "f8951d095de3589b3face1dec3f4394c8ecedb38"
PAI_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"   # recipes/alpamayo2edge_sft/README.md:88
PINS = {"alpamayo2_edge": EDGE_REPO_COMMIT, "alpamayo_recipes": RECIPES_COMMIT,
        "hf_model": HF_MODEL_ID, "hf_revision": HF_REVISION, "pai_revision": PAI_REVISION}

DEFAULT_SEED = 43                 # alpamayo2_edge/inference.py:278
DEFAULT_NUM_CANDIDATES = 6        # the model card's measured configuration
DEFAULT_MAX_GENERATION_LENGTH = 256   # inference.py:284
DEFAULT_DIFFUSION_STEPS = 10      # inference.py:285
EVENTS_ASSET = "recipes/alpamayo2edge_sft/assets/nav_demo_samples.json"

# docs/configs/alpamayo2-edge/config.json (traj_ids, vlm_config). Informational: the probes
# read the live values from model.config and report any disagreement.
EDGE_EXPECTED_IDS = {
    "history_id0": 131072, "future_id0": 132072,
    "history_start": 135077, "history_end": 135079,
    "future_start": 135084, "future_end": 135086,
    "history_pad": 135087, "future_pad": 135088,
    "image_token_id": 19, "eos_token_id": 11, "pad_token_id": 0,
}
HISTORY_VOCAB, FUTURE_VOCAB = 1000, 3000
CLOSED_THINKING = "<think></think>"           # alpamayo2_edge/chat_template.py:24
ANCHOR_TOKEN = "<|traj_future_start|>"

# Clip-relative window in which an event keeps 2 s of camera history and 6.4 s of future
# inside a 20 s PAI clip (recipes/alpamayo2edge_sft/dataset.py:451-463).
MIN_T0_US, MAX_T0_US = 2_000_000, 13_600_000

# Camera identifiers and the native frame profile (alpamayo2_edge/sample.py:13-17,
# load_physical_aiavdataset.py:21-30). Same identifiers as afh/cameras.py.
EDGE_CAMERA_NAMES = {
    0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov", 3: "camera_rear_left_70fov",
    5: "camera_rear_right_70fov", 6: "camera_front_tele_30fov",
}
EDGE_CAMERA_FRAME_COUNTS = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
EDGE_PIXEL_BOUNDS = {cam: (163840, 196608) if cam in (1, 6) else (81920, 98304)
                     for cam in EDGE_CAMERA_FRAME_COUNTS}
FRONT_WIDE_CAMERA, FRONT_TELE_CAMERA = 1, 6

# Run in this order: from no camera at all to a single camera missing.
BLACKOUT_CONDITIONS = {
    "blackout_all": tuple(EDGE_CAMERA_FRAME_COUNTS),                  # every camera, every frame
    "blackout_front_both": (FRONT_WIDE_CAMERA, FRONT_TELE_CAMERA),    # both forward cameras
    "blackout_front": (FRONT_WIDE_CAMERA,),                           # front wide only; telephoto stays
}


# --------------------------------------------------------------------------- events

@dataclass(frozen=True)
class Event:
    index: int
    clip_id: str
    t0_us: int
    nav_text: str
    cot: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.clip_id}_{self.t0_us}"


def parse_indices(spec: Optional[str], n: int) -> list[int]:
    """'0,3,5-7' -> [0, 3, 5, 6, 7]; None or 'all' -> every index. Out-of-range is an error."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad or not out:
        raise ValueError(f"event indices {bad or spec!r} outside [0, {n})")
    return list(dict.fromkeys(out))


def load_events(path: str, indices: Optional[str] = None) -> list[Event]:
    """Rows of an annotation JSON in the recipe's schema: clip_id, t0_relative (integer
    clip-relative microseconds), nav_text, optional cot (recipes/alpamayo2edge_sft/README.md:121-126)."""
    with open(path) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: annotations must be a nonempty JSON list")
    events = []
    for i, row in enumerate(rows):
        clip, t0, nav = row.get("clip_id"), row.get("t0_relative"), row.get("nav_text")
        if isinstance(nav, list) and len(nav) == 1:
            nav = nav[0]
        if not isinstance(clip, str) or not clip.strip():
            raise ValueError(f"{path}: row {i}: clip_id must be a nonempty string")
        if type(t0) is not int:
            raise ValueError(f"{path}: row {i}: t0_relative must be integer microseconds")
        if not isinstance(nav, str) or not nav.strip():
            raise ValueError(f"{path}: row {i}: nav_text must be a nonempty string")
        cot = row.get("cot")
        events.append(Event(i, clip, t0, nav, cot if isinstance(cot, str) and cot.strip() else None))
    return [events[i] for i in parse_indices(indices, len(events))]


_SIDE_RE = re.compile(r"\b(left|right)\b")


def nav_counterfactuals(nav_text: str) -> list[dict[str, str]]:
    """Counterfactual instructions for one event, in the release's own syntax:
    'mirror' swaps left and right and keeps everything else (distance included);
    'straight' is the literal 'Continue straight'. Both forms are named by the recipe's
    validator (src/alpamayo/data/pai_metadata.py:84-100), which the probe runs on each."""
    out = []
    if _SIDE_RE.search(nav_text):
        mirrored = _SIDE_RE.sub(lambda m: "right" if m.group(1) == "left" else "left", nav_text)
        out.append({"label": "mirror", "nav_text": mirrored})
    if nav_text != "Continue straight":
        out.append({"label": "straight", "nav_text": "Continue straight"})
    return out


def direction_mentions(text: Optional[str]) -> dict[str, int]:
    """Occurrences of the words left / right / straight in a text (whole words, any case)."""
    t = (text or "").lower()
    return {w: len(re.findall(rf"\b{w}\b", t)) for w in ("left", "right", "straight")}


# --------------------------------------------------------------------------- generation analysis

# Same axis heads as the release's splitter (alpamayo2_edge/models/token_utils.py:122).
_AXIS_HEAD_RE = re.compile(r"(Longitudinal|Lateral|Lane)\s*:")


def parse_meta_actions(text: Optional[str]) -> list[dict[str, str]]:
    """Every `Longitudinal|Lateral|Lane: value` found in `text`, in order; a value runs to
    the next axis head or the end of the line, so several axes on one line are separated."""
    if not text:
        return []
    heads = list(_AXIS_HEAD_RE.finditer(text))
    out = []
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        out.append({"axis": m.group(1), "value": text[m.end():end].split("\n", 1)[0].strip()})
    return out


def analyze_edge_generation(
    token_ids: Sequence[int],
    id_to_token: Callable[[int], str],
    decode: Callable[[Sequence[int]], str],
    traj_ids: dict[str, int],
    eos_ids: Iterable[int],
    pad_id: Optional[int] = None,
    first_n: int = 12,
    history_vocab: int = HISTORY_VOCAB,
    future_vocab: int = FUTURE_VOCAB,
) -> dict[str, Any]:
    """
    Describe what was generated after the prompt, without assuming a format.

    token_ids:  generated ids (prompt removed), as returned by generate().
    traj_ids:   model.config.traj_ids (future_start is the trajectory anchor).
    eos_ids:    text EOS ids of the generation config (<|im_end|>).

    The "reasoning span" is everything before the first anchor or EOS, whichever comes
    first. Its token count is the reasoning length; zero is a valid observation. <think>
    tags are looked up in the decoded text because their tokenisation is not known here.
    """
    ids = [int(t) for t in token_ids]
    trailing_pad = 0
    while ids and pad_id is not None and ids[-1] == pad_id:
        ids.pop()
        trailing_pad += 1
    eos = {int(e) for e in eos_ids}
    anchor = int(traj_ids["future_start"])
    anchor_pos = [i for i, t in enumerate(ids) if t == anchor]
    eos_pos = [i for i, t in enumerate(ids) if t in eos]
    end_pos = [i for i, t in enumerate(ids) if t == int(traj_ids["future_end"])]

    def in_range(t: int, key: str, size: int) -> bool:
        return int(traj_ids[key]) <= t < int(traj_ids[key]) + size

    cuts = [p[0] for p in (anchor_pos, eos_pos) if p]
    span_end = min(cuts) if cuts else len(ids)
    span = ids[:span_end]
    raw = decode(span)
    stripped = raw.strip()
    think_open, think_close = "<think>" in raw, "</think>" in raw
    think_block = None
    cot = stripped
    if stripped.startswith(CLOSED_THINKING):
        cot = stripped[len(CLOSED_THINKING):].strip()
    elif think_open and think_close:
        head, _, tail = stripped.partition("</think>")
        think_block = head.split("<think>", 1)[-1].strip()
        cot = tail.strip()

    if anchor_pos and (not eos_pos or anchor_pos[0] < eos_pos[0]):
        stop = "anchor"
    elif eos_pos:
        stop = "eos_before_anchor"
    else:
        stop = "no_anchor_no_eos"

    after = ids[anchor_pos[0] + 1:] if anchor_pos else []
    return {
        "n_generated": len(ids),
        "trailing_pad_tokens_removed": trailing_pad,
        "first_tokens": [{"position": i, "id": t, "token": id_to_token(t), "text": decode([t])}
                         for i, t in enumerate(ids[:first_n])],
        "stop": stop,
        "anchor_position": anchor_pos[0] if anchor_pos else None,
        "n_anchors": len(anchor_pos),
        "eos_positions": eos_pos,
        "traj_future_end_positions": end_pos,
        "reasoning_tokens": len(span),
        "reasoning_text_raw": raw,
        "cot": cot,
        "cot_words": len(cot.split()),
        "cot_chars": len(cot),
        "cot_empty": cot == "",
        "think": {"open_tag_generated": think_open, "close_tag_generated": think_close,
                  "starts_with_closed_block": stripped.startswith(CLOSED_THINKING),
                  "block_text": think_block},
        "meta_actions": parse_meta_actions(cot),
        "n_future_trajectory_tokens": sum(1 for t in ids if in_range(t, "future_id0", future_vocab)),
        "n_history_trajectory_tokens": sum(1 for t in ids if in_range(t, "history_id0", history_vocab)),
        "after_anchor": {"n_tokens": len(after),
                         "first_tokens": [id_to_token(t) for t in after[:first_n]]},
        "text_full": decode(ids),
        "generated_ids": ids,
    }


def summarize_reasoning(candidates: Sequence[dict]) -> dict[str, Any]:
    """Reasoning-length distribution over candidates (each an analyze_edge_generation dict)."""
    n = len(candidates)
    if not n:
        return {"n_candidates": 0}
    tokens = sorted(c["reasoning_tokens"] for c in candidates)
    words = sorted(c["cot_words"] for c in candidates)
    stops: dict[str, int] = {}
    for c in candidates:
        stops[c["stop"]] = stops.get(c["stop"], 0) + 1
    return {
        "n_candidates": n,
        "stop_counts": stops,
        "n_empty_cot": sum(1 for c in candidates if c["cot_empty"]),
        "n_think_open_generated": sum(1 for c in candidates if c["think"]["open_tag_generated"]),
        "n_with_meta_actions": sum(1 for c in candidates if c["meta_actions"]),
        "n_with_future_tokens_generated": sum(1 for c in candidates if c["n_future_trajectory_tokens"]),
        "reasoning_tokens": {"min": tokens[0], "median": float(np.median(tokens)), "max": tokens[-1],
                             "mean": round(float(np.mean(tokens)), 2)},
        "cot_words": {"min": words[0], "median": float(np.median(words)), "max": words[-1]},
    }


# --------------------------------------------------------------------------- images by camera id

def positions_for_cameras(camera_indices: Sequence[int], camera_ids: Iterable[int]) -> list[int]:
    """Positions of the flat image list whose camera identifier is in camera_ids. A requested
    camera that is absent is an error: a blackout that hits nothing must not pass silently."""
    wanted = [int(c) for c in camera_ids]
    cams = [int(c) for c in camera_indices]
    missing = [c for c in wanted if c not in cams]
    if missing:
        raise ValueError(f"camera ids {missing} not in the sample (present: {sorted(set(cams))})")
    return [i for i, c in enumerate(cams) if c in set(wanted)]


@dataclass(frozen=True)
class EdgeInputConfig:
    """Images of a training attempt: for each camera, its last `frames_per_camera` captures."""
    name: str
    camera_ids: tuple[int, ...]
    frames_per_camera: tuple[int, ...]

    @property
    def n_images(self) -> int:
        return sum(self.frames_per_camera)


_ALL = tuple(EDGE_CAMERA_FRAME_COUNTS)
# Native first, then fewer images (older frames dropped), then the recipe's debug input
# (configs/sft_joint_single_node_debug.yaml: front wide camera, one frame).
EDGE_FALLBACK_CONFIGS = (
    EdgeInputConfig("6cam_22img_native", _ALL, tuple(EDGE_CAMERA_FRAME_COUNTS[c] for c in _ALL)),
    EdgeInputConfig("6cam_12img_last2", _ALL, (2,) * 6),
    EdgeInputConfig("6cam_6img_last1", _ALL, (1,) * 6),
    EdgeInputConfig("2cam_2img_front", (1, 6), (1, 1)),
    EdgeInputConfig("1cam_1img_recipe_debug", (1,), (1,)),
)


def select_positions(camera_indices: Sequence[int], timestamps: Sequence[int],
                     cfg: EdgeInputConfig) -> list[int]:
    """Positions kept for `cfg`: per camera, the last k images by capture time. Never drops
    part of an image and never reorders: positions come back ascending."""
    cams = [int(c) for c in camera_indices]
    ts = [int(t) for t in timestamps]
    if len(cams) != len(ts):
        raise ValueError("camera_indices and timestamps must have the same length")
    keep: list[int] = []
    for cam, k in zip(cfg.camera_ids, cfg.frames_per_camera, strict=True):
        own = sorted((i for i, c in enumerate(cams) if c == cam), key=lambda i: (ts[i], i))
        if len(own) < k:
            raise ValueError(f"camera {cam}: {len(own)} images available, {k} requested by {cfg.name}")
        keep.extend(own[len(own) - k:])
    return sorted(keep)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Same arithmetic as transformers 5.17.0 qwen2_vl.smart_resize, which both the inference
    package (helper.py:111-116) and the recipe (processor.py:237-242) call with factor 32."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def predict_visual_tokens(height: int, width: int, cfg: EdgeInputConfig = EDGE_FALLBACK_CONFIGS[0],
                          patch: int = 16, merge: int = 2) -> int:
    """Image placeholder tokens expected for source frames of height x width: one token per
    (patch*merge)^2 pixels after resizing to the camera's pixel budget. 2 892 for the native
    profile at 1080 x 1920 (docs/edge_review.md §5). A prediction to compare with the measure."""
    total = 0
    for cam, k in zip(cfg.camera_ids, cfg.frames_per_camera, strict=True):
        lo, hi = EDGE_PIXEL_BOUNDS[cam]
        h, w = smart_resize(height, width, patch * merge, lo, hi)
        total += k * (h // patch) * (w // patch) // merge ** 2
    return total


# --------------------------------------------------------------------------- trajectories

def _xy(traj: Any) -> np.ndarray:
    a = np.asarray(traj, dtype=float)
    if a.ndim < 2 or a.shape[-1] < 2:
        raise ValueError(f"expected [..., T, >=2], got {a.shape}")
    return a[..., :2]


def traj_summary(traj: Any, dt: float = 0.1) -> dict[str, float]:
    """End point, path length and speed over the first and last second of one trajectory
    [T, >=2] in the ego frame at t0 (origin = ego at t0, waypoints every dt from dt)."""
    xy = _xy(traj)
    pts = np.vstack([np.zeros((1, 2)), xy])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    n = max(1, int(round(1.0 / dt)))
    return {"end_x_m": round(float(xy[-1, 0]), 3), "end_y_m": round(float(xy[-1, 1]), 3),
            "path_length_m": round(float(seg.sum()), 3),
            "speed_first_second_mps": round(float(seg[:n].sum() / (n * dt)), 3),
            "speed_last_second_mps": round(float(seg[-n:].sum() / (n * dt)), 3)}


def ade(a: Any, b: Any) -> float:
    """Mean XY distance between two trajectories of the same length."""
    a, b = _xy(a), _xy(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b, axis=-1).mean())


def within_spread(trajs: Any) -> Optional[float]:
    """Mean pairwise ADE among the K candidates of one condition: the sampling noise floor.
    None with fewer than two candidates."""
    xy = _xy(trajs)
    k = xy.shape[0]
    if xy.ndim != 3 or k < 2:
        return None
    pairs = [ade(xy[i], xy[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(pairs))


def deviation(base: Any, cond: Any) -> dict[str, Any]:
    """How far condition trajectories [K, T, >=2] are from baseline ones [K', T, >=2].

    ade_between_means compares the two mean trajectories; ade_paired pairs candidate i with
    candidate i (meaningful when both runs used the same seed, hence the same noise);
    exceeds_baseline_spread says whether the deviation between means is larger than the
    spread among baseline candidates. Facts only: no threshold beyond that comparison."""
    b, c = _xy(base), _xy(cond)
    if b.ndim != 3 or c.ndim != 3 or b.shape[1] != c.shape[1]:
        raise ValueError(f"expected [K, T, 2] with equal T, got {b.shape} and {c.shape}")
    bm, cm = b.mean(axis=0), c.mean(axis=0)
    spread = within_spread(b)
    between = ade(bm, cm)
    out: dict[str, Any] = {
        "ade_between_means_m": round(between, 3),
        "fde_between_means_m": round(float(np.linalg.norm(bm[-1] - cm[-1])), 3),
        "ade_paired_m": (round(float(np.mean([ade(b[i], c[i]) for i in range(b.shape[0])])), 3)
                         if b.shape[0] == c.shape[0] else None),
        "baseline_within_spread_m": None if spread is None else round(spread, 3),
        "condition_within_spread_m": (None if within_spread(c) is None
                                      else round(within_spread(c), 3)),
        "exceeds_baseline_spread": None if spread is None else bool(between > spread),
        "end_y_baseline_m": round(float(bm[-1, 1]), 3), "end_y_condition_m": round(float(cm[-1, 1]), 3),
        "path_length_baseline_m": traj_summary(bm)["path_length_m"],
        "path_length_condition_m": traj_summary(cm)["path_length_m"],
    }
    return out


# --------------------------------------------------------------------------- what a probe may write

# Dataset ground truth never goes into a probe JSON (docs/gpu_session_02.md, "Ce qui est
# publiable"): no ego pose from the dataset, past or future, and no score computed against
# the logged future. The probes write model outputs and measurements about the run.
# Keys are matched whole or by prefix after lower-casing; the guard walks the whole payload.
_FORBIDDEN_KEY_RE = re.compile(
    r"^(ego_(future|history)_(xyz|rot).*|future_(end_)?(xyz|rot|pose).*|gt_.*|.*ground_truth.*"
    r"|logged_future.*|.*_vs_logged_future.*|min_ade.*|eval_min_ade.*|(m|f)?ade_vs_(gt|truth).*)$")


class GroundTruthInPayload(ValueError):
    """A probe tried to write dataset ground truth, or a score against it, to its JSON."""


def forbidden_keys(payload: Any, path: str = "") -> list[str]:
    """Paths of every key of a nested payload that names dataset ground truth."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if _FORBIDDEN_KEY_RE.match(str(key).lower()):
                found.append(here)
            found += forbidden_keys(value, here)
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            found += forbidden_keys(value, f"{path}[{i}]")
    return found


def write_probe_json(path: str, payload: dict) -> None:
    """The only way an Edge probe writes its JSON: refuse ground-truth keys, then write
    atomically. A refusal is a bug in the probe, never something to work around."""
    bad = forbidden_keys(payload)
    if bad:
        raise GroundTruthInPayload(f"refusing to write dataset ground truth to {path}: {bad[:5]}")
    write_json(path, payload)


# --------------------------------------------------------------------------- GPU gate

def gpu_gate(total_gib: Optional[float], min_gib: float, allow_small: bool) -> dict[str, Any]:
    """Requirement on the GPU the probe runs on; an override is recorded, never silent."""
    meets = total_gib is not None and total_gib >= min_gib
    return {"min_gib": min_gib, "total_gib": total_gib, "meets_requirement": bool(meets),
            "overridden": bool(allow_small and not meets),
            "proceed": bool(meets or allow_small)}


# =========================================================================== GPU side
# Everything below needs torch and the recipe environment (docs/gpu_session_02.md, block 0).

class GenerateRecorder:
    """Wrap `vlm.generate`: keep the prompt length and a CPU copy of every returned row,
    then return the output untouched. Recorded before the release looks for the anchor, so a
    generation without anchor is still on record."""

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


def env_report() -> dict[str, Any]:
    import torch
    import transformers
    env: dict[str, Any] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available(),
        "alpamayo2_edge": package_provenance("alpamayo2-edge"),
        "alpamayo2edge_sft": package_provenance("alpamayo2edge-sft"),
        "physical_ai_av": package_provenance("physical-ai-av"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {"name": p.name, "total_memory_gib": round(p.total_memory / 2**30, 1),
                      "bf16_supported": bool(torch.cuda.is_bf16_supported())}
    return env


def step(payload: dict, name: str, out: str, fn: Callable[[], Any]) -> Any:
    """Run one step; record duration and result or error; write the JSON either way."""
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
        write_probe_json(out, payload)


def token_report(model) -> dict[str, Any]:
    """Live ids against docs/configs/alpamayo2-edge/config.json; mismatches reported, not fatal."""
    tok = model.tokenizer
    live = {k: int(v) for k, v in dict(model.config.traj_ids).items()}
    live["image_token_id"] = int(model.config.vlm_config.image_token_id)
    live["pad_token_id"] = None if tok.pad_token_id is None else int(tok.pad_token_id)
    eos = model.vlm.generation_config.eos_token_id
    live["eos_token_id"] = eos if isinstance(eos, int) else list(eos or [])
    mismatches = {k: {"expected": v, "live": live.get(k)} for k, v in EDGE_EXPECTED_IDS.items()
                  if live.get(k) != v and not (k == "eos_token_id" and live.get(k) == [v])}
    return {"live": live, "mismatches": mismatches, "all_match": not mismatches,
            "think_open_ids": tok.encode("<think>", add_special_tokens=False),
            "think_close_ids": tok.encode("</think>", add_special_tokens=False),
            "anchor_token": tok.convert_ids_to_tokens(live["future_start"])}


def eos_ids_of(model) -> list[int]:
    eos = model.vlm.generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def make_dataset_interface(pai_revision: Optional[str]):
    """One PhysicalAI-AV interface for the whole run (load_physical_aiavdataset accepts it
    through `avdi`, README of alpamayo2-edge, "Python API"). Streams from the Hub."""
    import physical_ai_av
    return physical_ai_av.PhysicalAIAVDatasetInterface(revision=pai_revision)


def load_sample(event: Event, avdi: Any, cache_dir: Optional[str]) -> tuple[dict, str]:
    """The raw Edge sample of an event: 22 uint8 frames, camera ids, capture timestamps,
    20 history poses, navigation. Cached on disk so blocks 2-4 do not stream again."""
    import torch
    path = os.path.join(cache_dir, f"{event.key}.pt") if cache_dir else None
    if path and os.path.exists(path):
        try:
            sample = torch.load(path, weights_only=True)
            sample["nav_text"] = [event.nav_text]
            return sample, f"cache:{path}"
        except Exception as exc:  # noqa: BLE001 — an unreadable cache entry is rebuilt, not fatal
            print(f"[cache] {path} unreadable ({type(exc).__name__}: {exc}); streaming again",
                  file=sys.stderr)
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
    sample = load_physical_aiavdataset(event.clip_id, t0_us=event.t0_us,
                                       nav_text=event.nav_text, avdi=avdi)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.tmp"
        torch.save(sample, tmp)
        os.replace(tmp, path)
    return sample, "physical_ai_av stream"


def sample_report(sample: dict) -> dict[str, Any]:
    cams = sample["camera_indices"].tolist()
    ts = sample["absolute_timestamps"].reshape(-1).tolist()
    h, w = (int(x) for x in sample["image_frames"].shape[-2:])
    return {"image_frames_shape": list(sample["image_frames"].shape),
            "camera_indices": cams,
            "images_per_camera": {str(c): cams.count(c) for c in sorted(set(cams))},
            "capture_offsets_s": [round((t - max(ts)) / 1e6, 3) for t in ts],
            "source_height_width": [h, w],
            "predicted_visual_tokens_native": predict_visual_tokens(h, w),
            "provenance": sample.get("provenance")}


def with_blackout(sample: dict, camera_ids: Iterable[int]) -> tuple[dict, list[int]]:
    """Copy of the sample whose images from `camera_ids` are black (all zeros, uint8)."""
    positions = positions_for_cameras(sample["camera_indices"].tolist(), camera_ids)
    out = dict(sample)
    frames = sample["image_frames"].clone()
    frames[positions] = 0
    out["image_frames"] = frames.contiguous()
    return out, positions


def with_navigation(sample: dict, nav_text: str) -> dict:
    out = dict(sample)
    out["nav_text"] = [nav_text]
    return out


def prompt_report(prompt_ids: list[int], model, tail: int = 16) -> dict[str, Any]:
    """End of the prompt as the model received it. The prompt carries the ego history as
    discrete tokens; those are dataset data, so any of them falling in the reported tail is
    withheld (the tail normally holds the task sentence and the assistant header only)."""
    tok = model.tokenizer
    image_id = int(model.config.vlm_config.image_token_id)
    h0 = int(model.config.traj_ids["history_id0"])
    h1 = h0 + int(getattr(model.config, "history_vocab_size", HISTORY_VOCAB))
    raw_tail = prompt_ids[-tail:]
    withheld = sum(1 for t in raw_tail if h0 <= t < h1)
    kept = [t for t in raw_tail if not h0 <= t < h1]
    tail_text = tok.decode(kept, skip_special_tokens=False)
    return {"prompt_len": len(prompt_ids),
            "n_image_tokens": sum(1 for t in prompt_ids if t == image_id),
            "tail_ids": kept, "tail_tokens": tok.convert_ids_to_tokens(kept),
            "tail_text": tail_text, "history_tokens_withheld_from_tail": withheld,
            "ends_with_closed_thinking": tail_text.endswith(CLOSED_THINKING)}


def _analyze_rows(call: dict, model, first_n: int) -> list[dict]:
    tok = model.tokenizer
    rows = []
    for seq in call["sequences"].tolist():
        rows.append(analyze_edge_generation(
            seq[call["prompt_len"]:],
            id_to_token=lambda t: tok.convert_ids_to_tokens(int(t)),
            decode=lambda ids: tok.decode(list(ids), skip_special_tokens=False),
            traj_ids=dict(model.config.traj_ids), eos_ids=eos_ids_of(model),
            pad_id=tok.pad_token_id, first_n=first_n))
    return rows


def run_edge_inference(model, sample: dict, recorder: GenerateRecorder, *, seed: int,
                       num_traj_samples: int, max_generation_length: int,
                       diffusion_steps: int, first_n: int = 12) -> dict[str, Any]:
    """
    The release path, unchanged, on one raw sample:
        helper.prepare_model_inputs -> helper.to_device
        -> Alpamayo2Edge.sample_trajectories_from_data(return_extra=True)
    seeded like alpamayo2_edge/inference.py:38-61 (torch.manual_seed + a seeded generator
    for diffusion). A generation that never reaches the anchor raises in the release
    (expert_utils.py:80-85); it is recorded here as status "no_anchor" with the raw tokens.
    """
    import torch
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    device = next(model.parameters()).device
    prepared = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device=device)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    n_before = len(recorder.calls)
    t = time.time()
    result: dict[str, Any] = {"seed": seed, "status": "ok", "error": None}
    xyz = extra = None
    try:
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=model.dtype, enabled=model.dtype != torch.float32):
            xyz, _rot, _placeholder, extra = model.sample_trajectories_from_data(
                prepared, num_traj_sets=1, num_traj_samples=num_traj_samples,
                top_p=0.98, top_k=None, temperature=0.6,
                max_generation_length=max_generation_length,
                diffusion_kwargs={"inference_step": diffusion_steps, "generator": generator},
                return_extra=True)
    except ValueError as exc:
        if not str(exc).startswith("No <traj_future_start> token found"):
            raise
        result.update(status="no_anchor", error=str(exc))
    torch.cuda.synchronize()
    result["seconds"] = round(time.time() - t, 2)
    result["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    calls = recorder.calls[n_before:]
    if calls:
        call = calls[0]
        result["prompt"] = prompt_report(call["prompt_ids"], model)
        result["candidates"] = _analyze_rows(call, model, first_n)
        result["reasoning_summary"] = summarize_reasoning(result["candidates"])
    if xyz is not None:
        traj = xyz[0, 0].float().cpu().numpy()                     # [K, 64, 3]
        result["pred_xyz"] = np.round(traj, 3).tolist()
        result["trajectory_summaries"] = [traj_summary(tr) for tr in traj]
        result["mean_trajectory_summary"] = traj_summary(traj.mean(axis=0))
        result["within_spread_m"] = (None if within_spread(traj) is None
                                     else round(within_spread(traj), 3))
        result["release_extraction"] = {
            key: np.asarray(extra[key]).reshape(-1).tolist()
            for key in ("cot", "meta_action", "answer") if key in extra}
    return result


def run_unconstrained(model, sample: dict, recorder: GenerateRecorder, *, seed: int,
                      max_new_tokens: int, first_n: int = 12) -> dict[str, Any]:
    """
    One direct `vlm.generate` on the release prompt, with the release sampling settings but
    WITHOUT the release's logits processors (which forbid trajectory tokens and forbid EOS
    before the anchor, alpamayo2_edge.py:273-285) and with the default EOS. Shows what the
    model emits when nothing steers it: does it reach the anchor by itself, does it write
    discrete trajectory tokens after it, where does it stop. Called like
    text_tasks.generate_vqa_answer calls it (text_tasks.py:26-54), on input ids fused with
    the history tokens like inference.run_data does (inference.py:94-102).
    """
    import copy

    import torch
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.models.utils import fuse_traj_tokens

    device = next(model.parameters()).device
    prepared = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device=device)
    tokenized = dict(prepared["tokenized_data"])
    histories = {key: prepared[key] for key in ("ego_history_xyz", "ego_history_rot")}
    generation = copy.deepcopy(model.vlm.generation_config)
    generation.do_sample = True
    generation.top_p = 0.98
    generation.top_k = None
    generation.temperature = 0.6
    generation.num_beams = 1
    generation.num_return_sequences = 1
    generation.max_new_tokens = max_new_tokens
    generation.pad_token_id = model.tokenizer.pad_token_id
    generation.return_dict_in_generate = False
    generation.output_scores = False
    generation.output_logits = False
    torch.manual_seed(seed)
    n_before = len(recorder.calls)
    t = time.time()
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=model.dtype, enabled=model.dtype != torch.float32):
        tokenized["input_ids"] = fuse_traj_tokens(
            model.history_traj_tokenizer, model.future_traj_tokenizer,
            tokenized["input_ids"], histories, model.config.traj_ids)
        model.vlm.generate(**tokenized, generation_config=generation)
    call = recorder.calls[n_before]
    return {"seed": seed, "max_new_tokens": max_new_tokens, "seconds": round(time.time() - t, 2),
            "logits_processors": "none", "eos_token_id": eos_ids_of(model),
            "candidates": _analyze_rows(call, model, first_n)}


def load_edge_model(model_id: str, revision: Optional[str], attn_implementation: str = "sdpa"):
    """checkpoint.load_model: strict loading, BF16, eval mode (checkpoint.py:92-137)."""
    from alpamayo2_edge.checkpoint import load_model
    return load_model(model_id, device="cuda", dtype="bfloat16",
                      attn_implementation=attn_implementation, revision=revision)


def validate_navigation_live(text: str) -> dict[str, Any]:
    """Run the recipe's own navigation validator (src/alpamayo/data/pai_metadata.py:84-100)."""
    try:
        from alpamayo.data.pai_metadata import validate_navigation
    except Exception as exc:  # noqa: BLE001 — recorded: the probe must say it was not validated
        return {"validated": False, "reason": f"validator unavailable: {type(exc).__name__}: {exc}"}
    try:
        validate_navigation(text, "probe")
        return {"validated": True, "valid": True}
    except ValueError as exc:
        return {"validated": True, "valid": False, "reason": str(exc)}


def add_common_args(ap) -> None:
    ap.add_argument("--events", required=True,
                    help=f"annotation JSON; the recipe's {EVENTS_ASSET}")
    ap.add_argument("--event-indices", default="all", help="e.g. 0,3,5-7 (default: all rows)")
    ap.add_argument("--model-id", default=os.environ.get("EDGE_MODEL", HF_MODEL_ID),
                    help="local model directory ($EDGE_MODEL) or HF repository id")
    ap.add_argument("--revision", default=HF_REVISION, help="HF revision when --model-id is a repo id")
    ap.add_argument("--pai-revision", default=PAI_REVISION)
    ap.add_argument("--sample-cache", default="outputs/session02/samples",
                    help="directory of cached raw samples, shared by blocks 1-4")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-traj-samples", type=int, default=DEFAULT_NUM_CANDIDATES,
                    help="candidates per event, each with its own reasoning")
    ap.add_argument("--max-generation-length", type=int, default=DEFAULT_MAX_GENERATION_LENGTH)
    ap.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    ap.add_argument("--first-n", type=int, default=12, help="generated tokens listed one by one")


def open_session(args, probe: str) -> tuple[dict, Any, GenerateRecorder, Any, list[Event]]:
    """Common start of blocks 1-3: payload, model, recorder, dataset interface, events."""
    import torch
    payload: dict[str, Any] = {"probe": probe, "pins": PINS, "args": vars(args), "steps": {},
                               "events": []}
    out = args.out
    events = step(payload, "load_events", out, lambda: load_events(args.events, args.event_indices))
    payload["n_events"] = len(events)
    payload["env"] = step(payload, "env", out, env_report)
    if not torch.cuda.is_available():
        payload["steps"]["env"]["error"] = "no CUDA device"
        write_probe_json(out, payload)
        raise SystemExit("CUDA GPU required")
    model, loading_info = step(payload, "load_model", out,
                               lambda: load_edge_model(args.model_id, args.revision))
    payload["model"] = {"path": getattr(model.config, "_name_or_path", None),
                        "dtype": str(model.dtype),
                        "memory_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                        "loading_problems": {k: v for k, v in dict(loading_info).items()
                                             if k in ("missing_keys", "unexpected_keys",
                                                      "mismatched_keys", "error_msgs") and v}}
    payload["token_ids"] = token_report(model)
    recorder = GenerateRecorder(model.vlm.generate)
    model.vlm.generate = recorder
    avdi = step(payload, "dataset_interface", out, lambda: make_dataset_interface(args.pai_revision))
    write_probe_json(out, payload)
    return payload, model, recorder, avdi, events


def load_event_sample(event: Event, avdi: Any, cache_dir: Optional[str]) -> tuple[Optional[dict], dict]:
    """Sample of an event, or None with the error on record (an event that cannot be loaded
    is a finding about that event, not a reason to stop the block)."""
    entry: dict[str, Any] = {**asdict(event), "conditions": {}}
    t = time.time()
    try:
        sample, source = load_sample(event, avdi, cache_dir)
    except Exception as exc:  # noqa: BLE001
        entry["load"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                         "seconds": round(time.time() - t, 1)}
        print(f"[event {event.index}] load FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, entry
    entry["load"] = {"status": "ok", "source": source, "seconds": round(time.time() - t, 1),
                     **sample_report(sample)}
    return sample, entry


def run_condition(model, sample: dict, recorder: GenerateRecorder, args) -> dict[str, Any]:
    """One condition of one event; an unexpected error is recorded and the block goes on."""
    try:
        return run_edge_inference(
            model, sample, recorder, seed=args.seed, num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            diffusion_steps=args.diffusion_steps, first_n=args.first_n)
    except Exception as exc:  # noqa: BLE001
        print(f"    condition FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                "traceback": traceback.format_exc()[-2000:]}


def baseline_trajectories(baseline: Optional[dict]) -> dict[str, list]:
    """event key -> clean pred_xyz [K, 64, 3] from a block-1 JSON (or any payload whose
    events carry a "clean" condition)."""
    out: dict[str, list] = {}
    for ev in (baseline or {}).get("events", []):
        clean = (ev.get("conditions") or {}).get("clean") or {}
        if clean.get("pred_xyz"):
            out[f"{ev['clip_id']}_{ev['t0_us']}"] = clean["pred_xyz"]
    return out


def baseline_cots(baseline: Optional[dict]) -> dict[str, list]:
    out: dict[str, list] = {}
    for ev in (baseline or {}).get("events", []):
        clean = (ev.get("conditions") or {}).get("clean") or {}
        if clean.get("candidates"):
            out[f"{ev['clip_id']}_{ev['t0_us']}"] = [c["cot"] for c in clean["candidates"]]
    return out


def read_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


def load_future_poses(event: Event, avdi: Any, cache_dir: Optional[str],
                      n_future: int = 64, dt_us: int = 100_000) -> tuple[dict, str]:
    """
    Future ego poses of an event in the ego frame at t0: ego_future_xyz [1, n, 3] and
    ego_future_rot [1, n, 3, 3], waypoints every dt from t0 + dt. The inference loader does
    not load them (load_physical_aiavdataset.py:147-148); this follows the same SDK calls and
    the same change of frame as its history block (:162-180), with t0 as the reference pose.

    The recipe's own dataset anchors poses at the last captured image and interpolates
    linearly (recipes/alpamayo2edge_sft/dataset.py:309-325); the values differ slightly from
    these, which changes the loss by a hair and the memory not at all.
    """
    import torch
    path = os.path.join(cache_dir, f"{event.key}_future.pt") if cache_dir else None
    if path and os.path.exists(path):
        try:
            return torch.load(path, weights_only=True), f"cache:{path}"
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] {path} unreadable ({type(exc).__name__}: {exc}); streaming again",
                  file=sys.stderr)
    egomotion = avdi.get_clip_feature(event.clip_id, avdi.features.LABELS.EGOMOTION,
                                      maybe_stream=True)
    stamps = event.t0_us + np.arange(0, n_future + 1, dtype=np.int64) * dt_us
    lower, upper = egomotion.time_range
    if stamps[0] < lower or stamps[-1] > upper:
        raise ValueError(f"future poses [{stamps[0]}, {stamps[-1]}] outside egomotion range "
                         f"[{lower}, {upper}] microseconds")
    poses = egomotion(stamps)
    world_xyz = np.asarray(poses.pose.translation)
    world_rot = poses.pose.rotation
    t0_inverse = world_rot[0].inv()
    local_xyz = t0_inverse.apply(world_xyz[1:] - world_xyz[0])
    local_rot = (t0_inverse * world_rot[1:]).as_matrix()
    if local_xyz.shape != (n_future, 3) or not np.isfinite(local_xyz).all():
        raise ValueError("egomotion did not return finite future poses")
    future = {"ego_future_xyz": torch.from_numpy(local_xyz).float().unsqueeze(0),
              "ego_future_rot": torch.from_numpy(local_rot).float().unsqueeze(0)}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.tmp"
        torch.save(future, tmp)
        os.replace(tmp, path)
    return future, "physical_ai_av stream"
