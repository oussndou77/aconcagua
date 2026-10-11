# SPDX-License-Identifier: Apache-2.0
"""
probe_common.py — torch-free helpers shared by the GPU probes of docs/gpu_session_01.md.

Everything here is pure Python so it can be tested cold (tests/test_probes_a2_cold.py):

  * analyse a generated token sequence without presupposing its format (unknown 1, D-008):
    first N tokens, positions of every structural marker, tagged blocks if any, meta-action
    lines whether tagged or not;
  * select the LM parameters the memory probe trains (q_proj of the last N layers);
  * the OOM fallback loop of the memory probe, which records every attempt so that a
    partial failure is still a measurement;
  * JSON written atomically after every step, so a crash leaves the last state on disk.

Token ids are never hard-coded: the GPU probes resolve them from the release tokenizer and
pass them in. EXPECTED_TOKEN_IDS only documents what the checkpoint config implies
(docs/configs/alpamayo2-super/config.json, traj_ids) and is used to flag a mismatch.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

DEFAULT_CLIP = "0ea6fd88-dcdd-434e-9fa3-56ce0fb35bf2"
DEFAULT_T0_US = 5_100_000
DEFAULT_MODEL_ID = "nvidia/Alpamayo2-Super"

# Structural markers whose position in a generation we report. Keys are the release's
# SPECIAL_TOKENS keys (NVlabs/alpamayo2 src/alpamayo2_super/models/utils.py) plus the two
# chat-template tokens of the Qwen tokenizer.
MARKER_TOKENS = {
    "cot_start": "<|cot_start|>",
    "cot_end": "<|cot_end|>",
    "meta_action_start": "<|meta_action_start|>",
    "meta_action_end": "<|meta_action_end|>",
    "traj_future_start": "<|traj_future_start|>",
    "traj_future_end": "<|traj_future_end|>",
    "answer_start": "<|answer_start|>",
    "answer_end": "<|answer_end|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "endoftext": "<|endoftext|>",
}

# Ids implied by the checkpoint config (traj_ids) and by the order in which
# build_alpamayo2_super_tokenizer adds SPECIAL_TOKENS after <i0>…<i3999> (151669…155668).
# Informational: the probe reports any disagreement with the live tokenizer.
EXPECTED_TOKEN_IDS = {
    "<|cot_start|>": 155677,
    "<|cot_end|>": 155678,
    "<|meta_action_start|>": 155679,
    "<|meta_action_end|>": 155680,
    "<|traj_future_start|>": 155681,
    "<|traj_future_end|>": 155683,
    "<|traj_history|>": 155684,
    "<|traj_future|>": 155685,
    "<|image_pad|>": 151655,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|endoftext|>": 151643,
}
HISTORY_ID0, FUTURE_ID0, TRAJ_VOCAB = 151669, 152669, 4000

META_ACTION_AXES = ("Longitudinal", "Lateral", "Lane")
_AXIS_LINE_RE = re.compile(r"(?P<axis>Longitudinal|Lateral|Lane)\s*:\s*(?P<value>[^\n]*)")


# --------------------------------------------------------------------------- generation analysis

def _positions(ids: Sequence[int], target: Optional[int]) -> list[int]:
    if target is None:
        return []
    return [i for i, t in enumerate(ids) if t == target]


def _between(text: str, start: str, end: str) -> Optional[str]:
    """Text between the first `start` and the next `end`; None if `start` is absent.
    An unterminated block (start without end) returns the rest of the text."""
    i = text.find(start)
    if i < 0:
        return None
    rest = text[i + len(start):]
    j = rest.find(end)
    return (rest if j < 0 else rest[:j]).strip()


def parse_meta_actions(text: Optional[str]) -> list[dict[str, str]]:
    """Every `Longitudinal|Lateral|Lane: value` line in `text`, in order."""
    if not text:
        return []
    return [{"axis": m.group("axis"), "value": m.group("value").strip()}
            for m in _AXIS_LINE_RE.finditer(text)]


def untagged_cot(text: str) -> str:
    """CoC of an untagged generation: text before the first axis head, the first
    <|traj_future_start|> or <|im_end|>, whichever comes first."""
    cut = len(text)
    for marker in (MARKER_TOKENS["traj_future_start"], MARKER_TOKENS["im_end"]):
        k = text.find(marker)
        if k >= 0:
            cut = min(cut, k)
    m = re.search(r"(?:Longitudinal|Lateral|Lane)\s*:", text[:cut])
    if m:
        cut = m.start()
    return text[:cut].strip()


def analyze_generation(
    token_ids: Sequence[int],
    id_to_token: Callable[[int], str],
    decode: Callable[[Sequence[int]], str],
    marker_ids: dict[str, Optional[int]],
    first_n: int = 10,
) -> dict[str, Any]:
    """
    Describe a generated token sequence (prompt already removed) without assuming a format.

    token_ids:   generated ids, as returned by generate() minus the prompt.
    id_to_token: tokenizer.convert_ids_to_tokens for one id.
    decode:      tokenizer.decode(ids, skip_special_tokens=False).
    marker_ids:  {marker key of MARKER_TOKENS: id or None} from the live tokenizer.

    Returns first tokens, marker positions, an explicit verdict on <|cot_start|>, the full
    text, the tagged blocks if present, the meta-action lines found and where they came
    from, and the CoC to feed the training-memory probe.
    """
    ids = [int(t) for t in token_ids]
    first = [{"position": i, "id": t, "token": id_to_token(t), "text": decode([t])}
             for i, t in enumerate(ids[:first_n])]
    markers = {k: _positions(ids, marker_ids.get(k)) for k in MARKER_TOKENS}
    traj_tokens = sum(1 for t in ids if HISTORY_ID0 <= t < HISTORY_ID0 + TRAJ_VOCAB)

    cot_pos = markers["cot_start"]
    if marker_ids.get("cot_start") is None:
        verdict = "unknown_token"        # tokenizer has no <|cot_start|>: cannot answer
    elif not cot_pos:
        verdict = "absent"
    elif cot_pos[0] == 0:
        verdict = "first_token"
    else:
        verdict = "later"

    text = decode(ids)
    tfs = MARKER_TOKENS["traj_future_start"]
    text_before_traj = text.split(tfs, 1)[0]

    cot_tagged = _between(text, MARKER_TOKENS["cot_start"], MARKER_TOKENS["cot_end"])
    ma_tagged = _between(text, MARKER_TOKENS["meta_action_start"], MARKER_TOKENS["meta_action_end"])
    if ma_tagged is not None:
        meta_actions = [dict(m, source="tagged") for m in parse_meta_actions(ma_tagged)]
    else:
        meta_actions = [dict(m, source="untagged") for m in parse_meta_actions(text_before_traj)]

    if cot_tagged is not None:
        cot_for_training, cot_source = cot_tagged, "tagged_block"
    else:
        cot_for_training, cot_source = untagged_cot(text), "untagged_text_before_meta_action_or_traj"

    return {
        "n_generated": len(ids),
        "first_tokens": first,
        "cot_start": {
            "token_id": marker_ids.get("cot_start"),
            "verdict": verdict,
            "first_position": cot_pos[0] if cot_pos else None,
            "statement": _cot_statement(verdict, cot_pos),
        },
        "marker_positions": markers,
        "n_discrete_trajectory_tokens": traj_tokens,
        "text_full": text,
        "text_before_traj_future_start": text_before_traj,
        "cot_tagged_block": cot_tagged,
        "meta_action_tagged_block": ma_tagged,
        "meta_actions": meta_actions,
        "cot_for_training": cot_for_training,
        "cot_source": cot_source,
    }


def _cot_statement(verdict: str, positions: list[int]) -> str:
    return {
        "first_token": "<|cot_start|> is the first generated token (position 0).",
        "later": f"<|cot_start|> is generated, first at position {positions[0] if positions else '?'}, "
                 "not as the first token.",
        "absent": "<|cot_start|> does not appear in the generated tokens.",
        "unknown_token": "the tokenizer has no <|cot_start|> token; the question cannot be answered.",
    }[verdict]


def check_token_ids(resolved: dict[str, Optional[int]]) -> dict[str, Any]:
    """Compare live tokenizer ids with EXPECTED_TOKEN_IDS; report mismatches, never fail."""
    mismatches = {tok: {"expected": exp, "live": resolved.get(tok)}
                  for tok, exp in EXPECTED_TOKEN_IDS.items()
                  if tok in resolved and resolved.get(tok) != exp}
    return {"resolved": resolved, "mismatches": mismatches, "all_match": not mismatches}


# --------------------------------------------------------------------------- memory probe helpers

_LM_LAYER_RE = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.(?P<proj>[a-z_]+)$")


def select_last_layer_modules(module_names: Iterable[str], n_last: int = 4,
                              proj: str = "q_proj") -> list[str]:
    """
    Names of the `proj` linear modules of the last `n_last` decoder layers of the language
    model. Only `language_model.layers.N.self_attn.<proj>` matches: the vision tower
    (`visual.blocks.N.attn.qkv`, `attn.proj`) and the diffusion expert (`expert.expert.layers…`,
    no `language_model` in its path) never do.
    """
    found: dict[int, str] = {}
    for name in module_names:
        m = _LM_LAYER_RE.search(name)
        if m and m.group("proj") == proj:
            found[int(m.group(1))] = name
    if not found:
        raise ValueError(f"no language_model.layers.*.self_attn.{proj} module found")
    layers = sorted(found)[-n_last:]
    return [found[i] for i in layers]


@dataclass(frozen=True)
class InputConfig:
    """Cameras (loader ids, ascending) and frames (indices into the 4 loaded frames) of an attempt."""
    name: str
    camera_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]


# Fewer cameras first, then fewer frames. Frames are always the LAST ones (ending at t0).
FALLBACK_CONFIGS = (
    InputConfig("7cam_4f_recipe_default", (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3)),
    InputConfig("6cam_4f_release_trajectory_profile", (0, 1, 2, 3, 5, 6), (0, 1, 2, 3)),
    InputConfig("4cam_4f_a15_default", (0, 1, 2, 6), (0, 1, 2, 3)),
    InputConfig("2cam_4f_front", (1, 6), (0, 1, 2, 3)),
    InputConfig("1cam_4f_front_wide", (1,), (0, 1, 2, 3)),
    InputConfig("1cam_2f_front_wide", (1,), (2, 3)),
    InputConfig("1cam_1f_recipe_debug", (1,), (3,)),
)


def classify_error(exc: BaseException) -> str:
    """'oom' (retry with a smaller input), 'too_long' (collator max_length, also retried),
    or 'other' (stop: a bug is not a memory measurement)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "OutOfMemoryError" or "out of memory" in msg or "cuda error: out of memory" in msg:
        return "oom"
    if "exceeds max_length" in msg:
        return "too_long"
    return "other"


@dataclass
class Attempt:
    config: dict
    status: str = "pending"            # ok | oom | too_long | error
    failed_step: Optional[str] = None
    error: Optional[str] = None
    steps: dict = field(default_factory=dict)   # step name -> measurements


def run_with_fallback(
    configs: Sequence[InputConfig],
    attempt_fn: Callable[[InputConfig, dict], None],
    on_update: Optional[Callable[[list[dict]], None]] = None,
    cleanup: Optional[Callable[[], None]] = None,
    sweep: bool = False,
) -> dict[str, Any]:
    """
    Try `configs` in order. `attempt_fn(config, steps)` fills `steps` (a dict it mutates
    step by step, so measurements taken before a failure are kept) and raises on failure;
    it should set `steps["_current"]` to the step name it is running.

    OOM or too-long -> record, `cleanup()`, try the next config. Any other error -> record
    and stop. First success stops unless `sweep`. `on_update(attempts)` is called after
    every attempt (write the JSON there).
    """
    attempts: list[dict] = []
    retained: Optional[str] = None
    stopped_on_error = False
    for cfg in configs:
        att = Attempt(config=asdict(cfg))
        steps: dict = {}
        try:
            attempt_fn(cfg, steps)
            att.status = "ok"
            if retained is None:
                retained = cfg.name
        except BaseException as exc:  # noqa: BLE001 — every failure must be recorded
            if isinstance(exc, KeyboardInterrupt):
                raise
            kind = classify_error(exc)
            att.status = kind if kind != "other" else "error"
            att.failed_step = steps.get("_current")
            att.error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if kind == "other":
                stopped_on_error = True
        finally:
            steps.pop("_current", None)
            att.steps = steps
            attempts.append(asdict(att))
            if cleanup is not None:
                cleanup()
            if on_update is not None:
                on_update(attempts)
        if stopped_on_error or (retained is not None and not sweep):
            break
    return {"retained_config": retained, "stopped_on_error": stopped_on_error,
            "attempts": attempts}


def count_span(ids: Sequence[int], start_id: Optional[int], end_id: Optional[int]) -> Optional[int]:
    """Tokens strictly between the first start_id and the next end_id; None if either is absent."""
    if start_id is None or end_id is None:
        return None
    ids = list(ids)
    try:
        i = ids.index(start_id)
        j = ids.index(end_id, i + 1)
    except ValueError:
        return None
    return j - i - 1


# --------------------------------------------------------------------------- output

def write_json(path: str, payload: dict) -> None:
    """Atomic write (tmp file + rename): an interrupted probe never leaves a truncated JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def package_provenance(dist_name: str) -> dict[str, Any]:
    """Version and, for git installs, the commit of an installed distribution (PEP 610)."""
    from importlib import metadata
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return {"installed": False}
    out: dict[str, Any] = {"installed": True, "version": dist.version}
    raw = dist.read_text("direct_url.json")
    if raw:
        try:
            info = json.loads(raw)
            out["url"] = info.get("url")
            out["commit"] = (info.get("vcs_info") or {}).get("commit_id")
            out["editable"] = (info.get("dir_info") or {}).get("editable", False)
        except json.JSONDecodeError:
            pass
    return out
