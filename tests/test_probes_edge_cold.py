#!/usr/bin/env python3
"""
Cold tests for the GPU-session-02 probes (Alpamayo 2 Edge) — NO GPU, NO torch, NO alpamayo2_edge.

Covers what the probes do outside the model: the demo events and their navigation
counterfactuals, the format-agnostic reading of a generated sequence (reasoning length,
zero included), image selection by camera identifier on Edge's flat image list (blackout,
memory fallback), trajectory deviation with its noise floor, the predicted number of visual
tokens, the fallback loop with a too-long finding, every summary the probes write, and that
each command of docs/gpu_session_02.md only uses flags the scripts define.

Constants of the checkpoint are checked against docs/configs/alpamayo2-edge/ (HF revision
f8951d0); the chat-template test renders the checkpoint's own template.
"""

import json
import os
import re
import sys

import numpy as np
import pytest

import runners.probe_blackout_edge as pblack
import runners.probe_common as pc
import runners.probe_edge_common as ec
import runners.probe_nav_edge as pnav
import runners.probe_tokens_edge as ptok
import runners.probe_train_memory_edge as pmem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT, "docs/configs/alpamayo2-edge")

# clip_id, t0_relative and nav_text of the 19 events of NVlabs/alpamayo-recipes @ 622f9d2,
# recipes/alpamayo2edge_sft/assets/nav_demo_samples.json (blocks 1 to 3). The copy in
# tests/data/ is compared with the pinned asset by tests/test_edge_pinned_api_cold.py.
DEMO_EVENTS = [(r["clip_id"], r["t0_relative"], r["nav_text"])
               for r in json.load(open(os.path.join(ROOT, "tests/data/edge_demo_events.json")))]

IDS = ec.EDGE_EXPECTED_IDS
ANCHOR, FUT_END, EOS, PAD = IDS["future_start"], IDS["future_end"], IDS["eos_token_id"], IDS["pad_token_id"]
TRAJ_IDS = {k: v for k, v in IDS.items() if k.startswith(("history", "future"))}


# --------------------------------------------------------------------------- fakes

class FakeTokenizer:
    """Named tokens get the checkpoint's ids; every other word is one token."""

    def __init__(self):
        self.inv = {ANCHOR: "<|traj_future_start|>", FUT_END: "<|traj_future_end|>",
                    EOS: "<|im_end|>", PAD: "<unk>", 12: "<think>", 13: "</think>"}
        self.fwd = {v: k for k, v in self.inv.items()}
        self.words: dict[str, int] = {}

    def ids(self, *pieces):
        out = []
        for p in pieces:
            if isinstance(p, int):
                out.append(p)
            elif p in self.fwd:
                out.append(self.fwd[p])
            else:
                for w in p.split(" "):
                    out.append(self.words.setdefault(w, 1000 + len(self.words)))
        return out

    def token(self, i):
        if i in self.inv:
            return self.inv[i]
        if IDS["future_id0"] <= i < IDS["future_id0"] + ec.FUTURE_VOCAB:
            return f"<i{i - IDS['history_id0']}>"
        return next(w for w, k in self.words.items() if k == i)

    def decode(self, ids):
        out = ""
        for i in ids:
            t = self.token(int(i))
            out += t if t.startswith("<") or not out or out.endswith(">") else " " + t
        return out


def _analyze(tok, ids, **kw):
    return ec.analyze_edge_generation(ids, tok.token, tok.decode, TRAJ_IDS, [EOS], pad_id=PAD, **kw)


def _events_file(tmp_path, rows=DEMO_EVENTS, cot="The light is green, so turn left to follow our route."):
    p = tmp_path / "events.json"
    p.write_text(json.dumps([{"clip_id": c, "t0_relative": t, "nav_text": n, "cot": cot,
                              "nav_maneuver": "X", "distance_m": 1.0} for c, t, n in rows]))
    return str(p)


# --------------------------------------------------------------------------- checkpoint constants

def test_expected_ids_match_the_checkpoint_config():
    cfg = json.load(open(os.path.join(CONFIG_DIR, "config.json")))
    for key, value in cfg["traj_ids"].items():
        assert IDS[key] == value, key
    assert IDS["image_token_id"] == cfg["vlm_config"]["image_token_id"]
    assert cfg["history_vocab_size"] == ec.HISTORY_VOCAB and cfg["future_vocab_size"] == ec.FUTURE_VOCAB
    assert cfg["tokens_per_history_traj"] == 57 and cfg["tokens_per_future_traj"] == 128
    gen = json.load(open(os.path.join(CONFIG_DIR, "generation_config.json")))
    assert IDS["eos_token_id"] == gen["eos_token_id"] and IDS["pad_token_id"] == gen["pad_token_id"]
    # the front cameras use the checkpoint's own pixel budget
    assert ec.EDGE_PIXEL_BOUNDS[1] == ec.EDGE_PIXEL_BOUNDS[6] == (cfg["min_pixels"], cfg["max_pixels"])
    assert sum(ec.EDGE_CAMERA_FRAME_COUNTS.values()) == 22
    assert set(ec.EDGE_CAMERA_NAMES) == set(ec.EDGE_CAMERA_FRAME_COUNTS) == {0, 1, 2, 3, 5, 6}


def test_pins_are_the_ones_of_d016():
    assert ec.EDGE_REPO_COMMIT.startswith("02f55a7") and ec.RECIPES_COMMIT.startswith("622f9d2")
    assert ec.HF_REVISION.startswith("f8951d0") and ec.PAI_REVISION.startswith("33f9bf4")
    decisions = open(os.path.join(ROOT, "docs/DECISIONS.md")).read()
    d016 = decisions[decisions.index("## D-016"):]
    for short in ("622f9d2", "02f55a7", "f8951d0", "33f9bf4", "aaaa32c"):
        assert short in d016, short


def test_checkpoint_template_training_sequence_extends_the_inference_prompt():
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    assert jinja2.__version__
    tpl = env.from_string(open(os.path.join(CONFIG_DIR, "chat_template.jinja")).read())
    text = lambda s: {"type": "text", "text": s}                              # noqa: E731
    base = [{"role": "system", "content": [text("You are a driving assistant.")]},
            {"role": "user", "content": [text("Front camera: "), text("<t=0.0s>"),
                                         {"type": "image", "image": "IMG"}, text("\n"),
                                         text("Navigation input is: Turn left in 25m")]}]
    infer = tpl.render(messages=base, add_generation_prompt=True, add_vision_id=False,
                       enable_thinking=False)
    assert infer.endswith("<|im_start|>assistant\n" + ec.CLOSED_THINKING)
    answer = "Yield before turning left.\n<|traj_future_start|><|traj_future|><|traj_future_end|>"
    train = tpl.render(messages=base + [{"role": "assistant", "content": [text(answer)]}],
                       add_generation_prompt=False, add_vision_id=False, enable_thinking=False)
    assert train.startswith(infer)
    assert train.endswith(ec.CLOSED_THINKING + answer + "<|im_end|>\n")
    # the template enables thinking by default: the release passes enable_thinking=False
    default = tpl.render(messages=base, add_generation_prompt=True, add_vision_id=False)
    assert default.endswith("<|im_start|>assistant\n<think>\n")


# --------------------------------------------------------------------------- events and navigation

def test_load_events_reads_the_recipe_schema(tmp_path):
    events = ec.load_events(_events_file(tmp_path))
    assert len(events) == 19 and [e.index for e in events] == list(range(19))
    assert events[7].nav_text == "Turn right in 30m" and events[7].t0_us == 4_000_000
    assert events[0].key == "1ae17e2a-bc07-4e57-a04b-e9d5a372cd1e_5784920" and len(DEMO_EVENTS) == 19
    assert events[0].cot.startswith("The light is green")
    # every demo event keeps 2 s of camera history and 6.4 s of future inside the clip
    assert all(ec.MIN_T0_US <= e.t0_us <= ec.MAX_T0_US for e in events)
    assert min(e.t0_us for e in events) == ec.MIN_T0_US        # event 15 sits on the lower bound
    assert [e.index for e in ec.load_events(_events_file(tmp_path), "0,3,5-7")] == [0, 3, 5, 6, 7]


@pytest.mark.parametrize("row, match", [
    ({"t0_relative": 1, "nav_text": "Turn left in 5m"}, "clip_id"),
    ({"clip_id": "c", "t0_relative": 1.5, "nav_text": "Turn left in 5m"}, "t0_relative"),
    ({"clip_id": "c", "t0_relative": 1, "nav_text": " "}, "nav_text"),
])
def test_load_events_rejects_incomplete_rows(tmp_path, row, match):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match=match):
        ec.load_events(str(p))


def test_load_events_accepts_list_navigation_and_missing_cot(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps([{"clip_id": "c", "t0_relative": 5_000_000, "nav_text": ["Keep right in 80m"]}]))
    (e,) = ec.load_events(str(p))
    assert e.nav_text == "Keep right in 80m" and e.cot is None
    with pytest.raises(ValueError):
        ec.parse_indices("19", 19)
    with pytest.raises(ValueError):
        ec.parse_indices("", 19)
    assert ec.parse_indices("all", 3) == [0, 1, 2] and ec.parse_indices("2,2,0", 3) == [2, 0]


def test_navigation_counterfactuals_keep_the_release_syntax():
    for _, _, nav in DEMO_EVENTS:
        cfs = {c["label"]: c["nav_text"] for c in ec.nav_counterfactuals(nav)}
        assert set(cfs) == {"mirror", "straight"}
        assert cfs["straight"] == "Continue straight"
        side, other = ("left", "right") if "left" in nav else ("right", "left")
        assert cfs["mirror"] == nav.replace(side, other) and cfs["mirror"] != nav
        # distance and wording untouched: only the side changes
        assert re.fullmatch(r"Turn (left|right) in \d+m", cfs["mirror"])
    assert ec.nav_counterfactuals("Turn left in 11m")[0] == {"label": "mirror", "nav_text": "Turn right in 11m"}
    assert ec.nav_counterfactuals("Continue straight") == []
    assert ec.nav_counterfactuals("Exit roundabout") == [{"label": "straight", "nav_text": "Continue straight"}]
    assert ec.nav_counterfactuals("Keep slight right in 80m")[0]["nav_text"] == "Keep slight left in 80m"


def test_direction_mentions_counts_whole_words():
    m = ec.direction_mentions("Turn LEFT at the leftmost lane, then straight; the right is clear. Left.")
    assert m == {"left": 2, "right": 1, "straight": 1}
    assert ec.direction_mentions(None) == {"left": 0, "right": 0, "straight": 0}


# --------------------------------------------------------------------------- generation analysis

def test_reasoning_then_anchor_is_measured_without_assuming_tags():
    tok = FakeTokenizer()
    ids = tok.ids("Yield to oncoming traffic before turning left.", "<|traj_future_start|>", PAD, PAD)
    r = _analyze(tok, ids)
    assert r["stop"] == "anchor" and r["anchor_position"] == 7 and r["reasoning_tokens"] == 7
    assert r["cot"] == "Yield to oncoming traffic before turning left." and not r["cot_empty"]
    assert r["trailing_pad_tokens_removed"] == 2 and r["n_generated"] == 8
    assert r["think"] == {"open_tag_generated": False, "close_tag_generated": False,
                          "starts_with_closed_block": False, "block_text": None}
    assert [t["token"] for t in r["first_tokens"]][:2] == ["Yield", "to"]
    assert r["first_tokens"][-1]["id"] == ANCHOR and r["after_anchor"]["n_tokens"] == 0
    assert r["n_future_trajectory_tokens"] == 0 and r["meta_actions"] == []
    json.dumps(r)


def test_anchor_as_first_token_is_an_empty_reasoning_not_an_error():
    tok = FakeTokenizer()
    r = _analyze(tok, tok.ids("<|traj_future_start|>"))
    assert r["stop"] == "anchor" and r["anchor_position"] == 0
    assert r["reasoning_tokens"] == 0 and r["cot"] == "" and r["cot_empty"] and r["cot_words"] == 0


def test_generation_without_anchor_is_reported_with_its_text():
    tok = FakeTokenizer()
    r = _analyze(tok, tok.ids("The road ahead is clear so we keep"))
    assert r["stop"] == "no_anchor_no_eos" and r["anchor_position"] is None
    assert r["reasoning_tokens"] == 8 and r["cot"].startswith("The road ahead")
    r = _analyze(tok, tok.ids("Stop here.", "<|im_end|>", "<|traj_future_start|>"))
    assert r["stop"] == "eos_before_anchor" and r["eos_positions"] == [2]
    assert r["reasoning_tokens"] == 2 and r["cot"] == "Stop here." and r["anchor_position"] == 3


def test_think_tags_in_the_generation_are_reported():
    tok = FakeTokenizer()
    r = _analyze(tok, tok.ids("<think>", "the light is red", "</think>", "Stop at the line.",
                              "<|traj_future_start|>"))
    assert r["think"]["open_tag_generated"] and r["think"]["close_tag_generated"]
    assert r["think"]["block_text"] == "the light is red" and r["cot"] == "Stop at the line."
    assert not r["think"]["starts_with_closed_block"]
    r = _analyze(tok, tok.ids("<think>", "</think>", "Keep lane.", "<|traj_future_start|>"))
    assert r["think"]["starts_with_closed_block"] and r["cot"] == "Keep lane."


def test_unsteered_generation_with_trajectory_tokens_and_meta_actions():
    tok = FakeTokenizer()
    traj = [IDS["future_id0"] + k for k in range(128)]
    ids = tok.ids("Slow for the bus. Longitudinal: brake Lateral: straight", "<|traj_future_start|>",
                  *traj, "<|traj_future_end|>", "<|im_end|>")
    r = _analyze(tok, ids, first_n=4)
    assert r["stop"] == "anchor" and r["n_future_trajectory_tokens"] == 128
    assert r["after_anchor"]["n_tokens"] == 130 and len(r["after_anchor"]["first_tokens"]) == 4
    assert r["traj_future_end_positions"] == [r["anchor_position"] + 129]
    assert r["eos_positions"] == [r["anchor_position"] + 130]
    assert r["meta_actions"] == [{"axis": "Longitudinal", "value": "brake"},
                                 {"axis": "Lateral", "value": "straight"}]
    assert ec.parse_meta_actions("Lane : keep\nSlow down.") == [{"axis": "Lane", "value": "keep"}]
    assert ec.parse_meta_actions(None) == []
    assert len(r["first_tokens"]) == 4


def test_summarize_reasoning_counts_empty_and_unanchored_candidates():
    tok = FakeTokenizer()
    cands = [_analyze(tok, tok.ids("Keep lane and follow the lead vehicle.", "<|traj_future_start|>")),
             _analyze(tok, tok.ids("<|traj_future_start|>")),
             _analyze(tok, tok.ids("Turn left now"))]
    s = ec.summarize_reasoning(cands)
    assert s["n_candidates"] == 3 and s["n_empty_cot"] == 1
    assert s["stop_counts"] == {"anchor": 2, "no_anchor_no_eos": 1}
    assert s["reasoning_tokens"] == {"min": 0, "median": 3.0, "max": 7, "mean": 3.33}
    assert ec.summarize_reasoning([]) == {"n_candidates": 0}


# --------------------------------------------------------------------------- images by camera id

def _native_layout(time_major=False):
    """camera id and capture time of the 22 native images."""
    front, other = [-2.0, -1.0, -0.5, -0.1, 0.0], [-1.0, -0.1, 0.0]
    items = [(cam, int(5_100_000 + off * 1e6) + cam)        # + cam: cameras are not synchronised
             for cam in (0, 1, 2, 3, 5, 6) for off in (front if cam in (1, 6) else other)]
    if time_major:
        items.sort(key=lambda it: (it[1], it[0]))
    return [c for c, _ in items], [t for _, t in items]


def test_blackout_selects_images_by_camera_identifier():
    cams, _ = _native_layout()
    assert ec.positions_for_cameras(cams, ec.BLACKOUT_CONDITIONS["blackout_all"]) == list(range(22))
    # order of the session: no camera, then both forward cameras, then the front wide alone
    assert list(ec.BLACKOUT_CONDITIONS) == ["blackout_all", "blackout_front_both", "blackout_front"]
    assert "default=list(BLACKOUT_CONDITIONS)" in open(os.path.join(ROOT, "runners/probe_blackout_edge.py")).read()
    front = ec.positions_for_cameras(cams, ec.BLACKOUT_CONDITIONS["blackout_front"])
    assert front == [3, 4, 5, 6, 7] and all(cams[i] == 1 for i in front)
    assert ec.BLACKOUT_CONDITIONS["blackout_front"] == (1,)          # the telephoto (6) stays
    both = ec.positions_for_cameras(cams, ec.BLACKOUT_CONDITIONS["blackout_front_both"])
    assert ec.BLACKOUT_CONDITIONS["blackout_front_both"] == (1, 6)
    assert both == [3, 4, 5, 6, 7, 17, 18, 19, 20, 21] and {cams[i] for i in both} == {1, 6}
    assert {cams[i] for i in range(22) if i not in both} == {0, 2, 3, 5}   # side and rear stay
    assert set(front) < set(both) < set(range(22))
    # identity, not position: the same camera in a time-major list
    cams_t, _ = _native_layout(time_major=True)
    pos = ec.positions_for_cameras(cams_t, [1])
    assert len(pos) == 5 and pos != front and all(cams_t[i] == 1 for i in pos)
    with pytest.raises(ValueError, match=r"\[4\]"):
        ec.positions_for_cameras(cams, [4])                          # rear tele is not an Edge camera


@pytest.mark.parametrize("time_major", [False, True])
def test_fallback_configs_keep_the_latest_images_per_camera(time_major):
    cams, ts = _native_layout(time_major)
    counts = []
    for cfg in ec.EDGE_FALLBACK_CONFIGS:
        pos = ec.select_positions(cams, ts, cfg)
        counts.append(len(pos))
        assert pos == sorted(pos) and len(pos) == cfg.n_images
        for cam, k in zip(cfg.camera_ids, cfg.frames_per_camera):
            own = sorted(ts[i] for i, c in enumerate(cams) if c == cam)
            kept = sorted(ts[i] for i in pos if cams[i] == cam)
            assert kept == own[-k:], (cfg.name, cam)
        assert {cams[i] for i in pos} == set(cfg.camera_ids)
    assert counts == [22, 12, 6, 2, 1]
    assert ec.select_positions(cams, ts, ec.EDGE_FALLBACK_CONFIGS[0]) == list(range(22))
    last = ec.EDGE_FALLBACK_CONFIGS[-1]
    assert last.camera_ids == (1,) and last.frames_per_camera == (1,)   # the recipe's debug input
    assert len({c.name for c in ec.EDGE_FALLBACK_CONFIGS}) == 5
    with pytest.raises(ValueError, match="requested"):
        ec.select_positions(cams[:3], ts[:3], ec.EDGE_FALLBACK_CONFIGS[0])


def test_predicted_visual_tokens_of_the_native_profile():
    assert ec.smart_resize(1080, 1920, 32, *ec.EDGE_PIXEL_BOUNDS[1]) == (320, 576)
    assert ec.smart_resize(1080, 1920, 32, *ec.EDGE_PIXEL_BOUNDS[0]) == (224, 416)
    assert ec.predict_visual_tokens(1080, 1920) == 2892             # docs/edge_review.md §5
    assert [ec.predict_visual_tokens(1080, 1920, c) for c in ec.EDGE_FALLBACK_CONFIGS] == \
        [2892, 1448, 724, 360, 180]
    review = open(os.path.join(ROOT, "docs/edge_review.md")).read()
    assert "2 892" in review and "3 236" in review
    # upscaling branch: a small source is brought up to the minimum budget
    h, w = ec.smart_resize(120, 160, 32, 163840, 196608)
    assert h % 32 == 0 and w % 32 == 0 and 163840 <= h * w


# --------------------------------------------------------------------------- trajectories

def _line(speed, lateral=0.0, n=64):
    t = np.arange(1, n + 1) * 0.1
    return np.stack([speed * t, lateral * t / t[-1], np.zeros(n)], axis=1)


def test_trajectory_summary_and_deviation():
    s = ec.traj_summary(_line(10.0))
    assert s["end_x_m"] == 64.0 and s["end_y_m"] == 0.0 and s["path_length_m"] == 64.0
    assert s["speed_first_second_mps"] == 10.0 and s["speed_last_second_mps"] == 10.0
    stop = _line(10.0)
    stop[32:] = stop[31]                                           # stops after 3.2 s
    assert ec.traj_summary(stop)["speed_last_second_mps"] == 0.0
    assert ec.ade(_line(10.0), _line(10.0)) == 0.0
    base = np.stack([_line(10.0, 0.0), _line(10.0, 0.4), _line(10.0, -0.4)])
    assert ec.within_spread(base) > 0 and ec.within_spread(base[:1]) is None
    same = ec.deviation(base, base)
    assert same["ade_between_means_m"] == 0.0 and same["ade_paired_m"] == 0.0
    assert same["exceeds_baseline_spread"] is False
    braked = np.stack([_line(2.0)] * 3)
    d = ec.deviation(base, braked)
    assert d["exceeds_baseline_spread"] is True and d["ade_between_means_m"] > 20
    assert d["path_length_condition_m"] < d["path_length_baseline_m"]
    assert d["condition_within_spread_m"] == 0.0
    turned = np.stack([_line(10.0, -8.0)] * 2)
    d = ec.deviation(np.stack([_line(10.0, 8.0)] * 2), turned)
    assert d["end_y_baseline_m"] == 8.0 and d["end_y_condition_m"] == -8.0
    assert d["baseline_within_spread_m"] == 0.0
    # different candidate counts: no pairing, the means are still compared
    assert ec.deviation(base, braked[:2])["ade_paired_m"] is None
    assert ec.deviation(base[:1], braked[:1])["exceeds_baseline_spread"] is None
    with pytest.raises(ValueError):
        ec.deviation(base, braked[:, :10])
    json.dumps(d)


def test_baseline_lookup_by_event_key():
    block1 = {"args": {"seed": 43}, "events": [
        {"clip_id": "a", "t0_us": 1, "conditions": {"clean": {"pred_xyz": [[[1, 2, 3]]],
                                                               "candidates": [{"cot": "Go."}]}}},
        {"clip_id": "b", "t0_us": 2, "conditions": {"clean": {"status": "no_anchor",
                                                               "candidates": [{"cot": ""}]}}},
        {"clip_id": "c", "t0_us": 3, "conditions": {}}]}
    assert ec.baseline_trajectories(block1) == {"a_1": [[[1, 2, 3]]]}
    assert ec.baseline_cots(block1) == {"a_1": ["Go."], "b_2": [""]}
    assert ec.baseline_trajectories(None) == {} and ec.read_json(None) is None


# --------------------------------------------------------------------------- fallback and memory summary

def test_gpu_gate_records_an_override():
    assert ec.gpu_gate(79.6, 75.0, False) == {"min_gib": 75.0, "total_gib": 79.6, "meets_requirement": True,
                                              "overridden": False, "proceed": True}
    small = ec.gpu_gate(39.5, 75.0, False)
    assert not small["proceed"] and not small["overridden"]
    forced = ec.gpu_gate(39.5, 75.0, True)
    assert forced["proceed"] and forced["overridden"] and not forced["meets_requirement"]
    assert not ec.gpu_gate(None, 75.0, False)["proceed"]


class OutOfMemoryError(RuntimeError):
    """Same class name as torch.cuda.OutOfMemoryError."""


def _fake_attempt(outcomes, max_length=pmem.MAX_LENGTH):
    def attempt(cfg, steps):
        what = outcomes[cfg.name]
        steps["_current"] = "preprocess"
        steps["images"] = {"n_images": cfg.n_images,
                           "predicted_visual_tokens": ec.predict_visual_tokens(1080, 1920, cfg)}
        real_len = 9000 if what == "too_long" else 478 + steps["images"]["predicted_visual_tokens"]
        steps["sequence_before_padding"] = {"real_len": real_len, "max_length": max_length,
                                            "visual_tokens": steps["images"]["predicted_visual_tokens"]}
        steps["_current"] = "length_check"
        pc.check_sequence_length(real_len, max_length)
        steps["train_steps"] = [{"step": 0, "loss": 1.5, "peak_after_forward_gib": 30.0}]
        steps["_current"] = "step0_backward"
        if what == "oom":
            steps["peak_memory_gib"] = 79.0
            raise OutOfMemoryError("CUDA out of memory")
        if what == "bug":
            raise KeyError("labels_mask")
        steps["train_steps"][0].update(optimizer_seconds=0.2, step_seconds=1.9)
        steps["train_steps"].append({"step": 1, "loss": 1.4, "optimizer_seconds": 0.2, "step_seconds": 1.1})
        steps["peak_memory_gib"] = 52.0
        steps["seconds_per_step"] = 1.1
    return attempt


def test_native_config_that_fits_answers_d004():
    names = [c.name for c in ec.EDGE_FALLBACK_CONFIGS]
    res = pc.run_with_fallback(ec.EDGE_FALLBACK_CONFIGS, _fake_attempt({n: "ok" for n in names}))
    s = pmem.summarize(res, 9.9, 79.6)
    assert s["native_full_finetune_fits"] is True and s["retained_config"] == names[0] == s["native_config"]
    assert not s["retained_reduced"] and s["abandoned"] == [] and s["findings"] == []
    row = s["attempts"][0]
    assert len(s["attempts"]) == 1 and row["n_images"] == 22 and row["real_len"] == 3370
    assert row["visual_tokens"] == row["predicted_visual_tokens"] == 2892
    assert row["peak_memory_gib"] == 52.0 and row["seconds_per_step"] == 1.1
    assert row["steps_completed"] == 2 and row["loss_first_step"] == 1.5
    json.dumps(s)


def test_oom_then_too_long_then_fit_is_recorded_and_marked_reduced():
    names = [c.name for c in ec.EDGE_FALLBACK_CONFIGS]
    outcomes = {n: "ok" for n in names}
    outcomes[names[0]] = "oom"
    outcomes[names[1]] = "too_long"
    res = pc.run_with_fallback(ec.EDGE_FALLBACK_CONFIGS, _fake_attempt(outcomes))
    s = pmem.summarize(res, 9.9, 79.6)
    assert s["native_full_finetune_fits"] is False and not s["stopped_on_error"]
    assert s["retained_config"] == names[2] and s["retained_reduced"] and s["reduced_because"] == ["oom", "too_long"]
    assert [(a["config"], a["reason"]) for a in s["abandoned"]] == [(names[0], "oom"), (names[1], "too_long")]
    assert [r["status"] for r in s["attempts"]] == ["oom", "too_long", "ok"]
    assert s["attempts"][0]["failed_step"] == "step0_backward" and s["attempts"][0]["peak_memory_gib"] == 79.0
    assert s["attempts"][0]["steps_completed"] == 0
    # the too-long sequence is a finding: measured length, configuration, never truncated
    (finding,) = s["findings"]
    assert finding["finding"] == "SequenceTooLong" and finding["config"] == names[1]
    assert finding["real_len"] == 9000 and finding["max_length"] == 8192 and finding["truncated"] is False
    assert finding["camera_ids"] == [0, 1, 2, 3, 5, 6] and finding["frame_indices"] is None
    assert finding["config_detail"]["frames_per_camera"] == (2,) * 6 or \
        list(finding["config_detail"]["frames_per_camera"]) == [2] * 6
    assert s["attempts"][2]["reduced"] and s["attempts"][2]["reduced_because"] == ["oom", "too_long"]


def test_nothing_fits_and_a_bug_stops():
    names = [c.name for c in ec.EDGE_FALLBACK_CONFIGS]
    res = pc.run_with_fallback(ec.EDGE_FALLBACK_CONFIGS, _fake_attempt({n: "oom" for n in names}))
    s = pmem.summarize(res, 9.9, 39.5)
    assert s["retained_config"] is None and s["native_full_finetune_fits"] is False
    assert len(s["attempts"]) == 5 and not s["stopped_on_error"]
    outcomes = {n: "ok" for n in names}
    outcomes[names[0]] = "bug"
    res = pc.run_with_fallback(ec.EDGE_FALLBACK_CONFIGS, _fake_attempt(outcomes))
    s = pmem.summarize(res, 9.9, 79.6)
    assert s["stopped_on_error"] and len(s["attempts"]) == 1 and s["attempts"][0]["status"] == "error"


def test_memory_probe_constants_are_the_recipe_native_settings():
    assert pmem.MAX_LENGTH == 8192
    assert pmem.NATIVE_PREPROCESS["components_order"] == \
        ["image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"]
    assert pmem.NATIVE_PREPROCESS["label_components"] == ["assistant"]
    assert pmem.OPTIMIZER == {"lr": 3.5e-6, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    assert pmem.MIN_GPU_GIB == 75.0


# --------------------------------------------------------------------------- block summaries

def _cand(tok, text, anchor=True):
    return _analyze(tok, tok.ids(*([text] if text else []), *(["<|traj_future_start|>"] if anchor else [])))


def test_tokens_summary_over_events():
    tok = FakeTokenizer()
    payload = {"events": [
        {"index": 0, "nav_text": "Turn left in 11m", "conditions": {"clean": {
            "status": "ok", "prompt": {"prompt_len": 3236, "ends_with_closed_thinking": True},
            "candidates": [_cand(tok, "Turn left on green."), _cand(tok, "")], "within_spread_m": 0.4}}},
        {"index": 1, "nav_text": "Turn left in 13m", "conditions": {"clean": {
            "status": "no_anchor", "prompt": {"prompt_len": 3240, "ends_with_closed_thinking": True},
            "candidates": [_cand(tok, "Proceed and keep going", anchor=False)]}}},
        {"index": 2, "nav_text": "Turn left in 15m", "conditions": {"clean": {"status": "error"}}},
        {"index": 3, "nav_text": "Turn left in 6m", "conditions": {}, "load": {"status": "error"}}]}
    s = ptok.summarize(payload)
    assert s["events_with_output"] == 2 and s["events_failed_or_not_loaded"] == [2, 3]
    assert s["events_without_anchor"] == [1] and s["prompt_len"] == {"min": 3236, "max": 3240}
    assert s["prompt_ends_with_closed_thinking"] == {"true": 2, "false": 0}
    assert s["reasoning"]["n_candidates"] == 3 and s["reasoning"]["n_empty_cot"] == 1
    assert s["per_event"][0]["reasoning_tokens"] == [4, 0]
    assert s["per_event"][0]["cot_first_candidate"] == "Turn left on green."


def test_blackout_summary_compares_with_the_clean_spread():
    tok = FakeTokenizer()
    base = np.stack([_line(10.0, 0.0), _line(10.0, 0.5)])
    dev_big = ec.deviation(base, np.stack([_line(1.0)] * 2))
    dev_none = ec.deviation(base, base)
    payload = {"events": [
        {"index": 0, "conditions": {"blackout_all": {"status": "ok", "deviation_vs_clean": dev_big,
                                                      "cot_clean": ["Turn left on green."],
                                                      "candidates": [_cand(tok, "The road is clear.")]},
                                    "blackout_front": {"status": "ok", "deviation_vs_clean": dev_none,
                                                       "cot_clean": ["Turn left on green."],
                                                       "candidates": [_cand(tok, "Turn left on green.")]}}},
        {"index": 1, "conditions": {"blackout_all": {"status": "error"}}}]}
    payload["events"][0]["conditions"]["blackout_front_both"] = {
        "status": "ok", "deviation_vs_clean": dev_big, "cot_clean": ["Turn left on green."],
        "candidates": [_cand(tok, "")]}
    s = pblack.summarize(payload, list(ec.BLACKOUT_CONDITIONS))
    assert list(s) == ["blackout_all", "blackout_front_both", "blackout_front"]
    assert s["blackout_front_both"]["n_exceeding_clean_spread"] == 1
    assert s["blackout_front_both"]["reasoning"]["n_empty_cot"] == 1
    assert s["blackout_front_both"]["events_missing"] == [1]
    assert ec.forbidden_keys(s) == []
    a, f = s["blackout_all"], s["blackout_front"]
    assert a["events_with_output"] == 1 and a["events_missing"] == [1]
    assert a["n_exceeding_clean_spread"] == 1 and a["n_compared_with_spread"] == 1
    assert a["mean_path_length_condition_m"] < a["mean_path_length_clean_m"]
    assert a["per_event"][0]["cot_condition"] == "The road is clear."
    assert a["per_event"][0]["cot_clean"] == "Turn left on green."
    assert f["n_exceeding_clean_spread"] == 0 and f["events_missing"] == [1]
    assert f["mean_ade_between_means_m"] == 0.0
    json.dumps(s)


def test_navigation_follow_report():
    left = np.stack([_line(8.0, 9.0)] * 2)
    right = np.stack([_line(8.0, -9.0)] * 2)
    dev = ec.deviation(left, right)
    r = pnav.follow_report("Turn left in 11m", "Turn right in 11m", dev,
                           ["Turn left on green to follow our route."],
                           ["Turn right at the intersection to follow our route."], 1.0)
    assert r["new_direction_words"] == ["right"] and r["old_direction_words"] == ["left"]
    assert r["cot_names_new_direction"] is True and r["cot_still_names_old_direction"] is False
    assert r["cot_text_changed"] is True and r["lateral_sign_flipped"] is True
    # the model ignores the instruction: same trajectory, same words
    r = pnav.follow_report("Turn left in 11m", "Turn right in 11m", ec.deviation(left, left),
                           ["Turn left on green."], ["Turn left on green."], 1.0)
    assert r["lateral_sign_flipped"] is False and r["cot_names_new_direction"] is False
    assert r["cot_still_names_old_direction"] is True and r["cot_text_changed"] is False
    # "Continue straight": no lateral motion left to compare signs on
    straight = np.stack([_line(8.0, 0.1)] * 2)
    r = pnav.follow_report("Turn left in 11m", "Continue straight", ec.deviation(left, straight),
                           ["Turn left."], ["Continue straight through the intersection."], 1.0)
    assert r["lateral_sign_flipped"] is None and r["new_direction_words"] == ["straight"]
    assert r["cot_names_new_direction"] is True
    # no trajectory (no anchor): text facts only
    r = pnav.follow_report("Turn left in 11m", "Turn right in 11m", None, ["Turn left."], [], 1.0)
    assert r["lateral_sign_flipped"] is None and r["cot_text_changed"] is None


def test_navigation_summary_counts_per_label():
    tok = FakeTokenizer()
    left, right = np.stack([_line(8.0, 9.0)] * 2), np.stack([_line(8.0, -9.0)] * 2)
    dev = ec.deviation(left, right)
    follow = pnav.follow_report("Turn left in 11m", "Turn right in 11m", dev, ["Turn left."], ["Turn right."], 1.0)
    payload = {"events": [
        {"index": 0, "nav_text": "Turn left in 11m", "conditions": {
            "mirror": {"status": "ok", "nav_text": "Turn right in 11m", "validation": {"validated": True, "valid": True},
                       "deviation_vs_clean": dev, "follow": follow, "cot_clean": ["Turn left."],
                       "candidates": [_cand(tok, "Turn right.")]},
            "straight": {"status": "invalid_navigation", "nav_text": "Continue straight",
                         "validation": {"validated": True, "valid": False}}}},
        {"index": 1, "nav_text": "Turn left in 13m", "conditions": {
            "mirror": {"status": "error", "nav_text": "Turn right in 13m",
                       "validation": {"validated": False, "reason": "validator unavailable"}}}}]}
    s = pnav.summarize(payload)
    m = s["mirror"]
    assert m["events_run"] == 1 and m["events_skipped_or_failed"] == [1]
    assert m["instructions_not_validated"] == 1
    assert m["lateral_sign_flipped"] == {"true": 1, "of": 1}
    assert m["cot_names_new_direction"] == {"true": 1, "of": 1}
    assert m["trajectory_beyond_clean_spread"] == {"true": 1, "of": 1}
    assert m["mean_abs_end_y_clean_m"] == 9.0 and m["mean_abs_end_y_counterfactual_m"] == 9.0
    assert m["per_event"][0]["cot_counterfactual"] == "Turn right."
    assert s["straight"]["events_run"] == 0 and s["straight"]["events_skipped_or_failed"] == [0]
    json.dumps(s)


# --------------------------------------------------------------------------- no ground truth in the JSON

def test_guard_refuses_dataset_ground_truth_keys(tmp_path):
    for key in ("ego_future_xyz", "ego_future_rot", "ego_history_xyz", "future_end_xyz", "future_poses",
                "gt_xy", "ground_truth", "logged_future_xy", "ade_vs_logged_future_m", "min_ade",
                "eval_min_ade", "eval_min_ade_successful", "ade_vs_gt_m", "Ego_Future_XYZ"):
        payload = {"example": {"nested": [{"ok": 1}, {key: [1.0, 2.0, 3.0]}]}}
        assert ec.forbidden_keys(payload) == [f"example.nested[1].{key}"], key
        with pytest.raises(ec.GroundTruthInPayload, match="ground truth"):
            ec.write_probe_json(str(tmp_path / "x.json"), payload)
    assert not (tmp_path / "x.json").exists()
    # what the probes do write is accepted: model outputs, deviations between model outputs,
    # training-sequence bookkeeping
    allowed = {"pred_xyz": [[[1, 2, 3]]], "deviation_vs_clean": {"ade_between_means_m": 1.0, "ade_paired_m": 1.0,
               "end_y_baseline_m": 0.0, "path_length_condition_m": 3.0}, "future_placeholders": 128,
               "n_future_trajectory_tokens": 0, "traj_future_end_positions": [], "future_traj_ce": 1.2,
               "target_trajectory": {"n_waypoints": 64, "values_written": False},
               "mean_ade_between_means_m": 0.5, "within_spread_m": 0.3}
    assert ec.forbidden_keys(allowed) == []
    ec.write_probe_json(str(tmp_path / "ok.json"), allowed)
    assert json.loads((tmp_path / "ok.json").read_text())["future_placeholders"] == 128


def test_edge_probes_only_write_through_the_guard():
    common = open(os.path.join(ROOT, "runners/probe_edge_common.py")).read()
    # the single unguarded write is the one inside write_probe_json itself
    assert common.count("write_json(") == 1 and "    write_json(path, payload)" in common
    for script in EDGE_SCRIPTS:
        src = open(os.path.join(ROOT, "runners", script)).read()
        assert "write_json(" not in src, script
        assert "write_probe_json(" in src, script
        assert "json.dump" not in src and "open(" not in src.split('"""', 2)[2], script


def _parents(tree):
    import ast
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def test_future_poses_only_feed_the_training_sample():
    """Static: in the memory probe the loaded future is used twice and nowhere else — merged
    into the training sample, and asked for its number of waypoints."""
    import ast
    src = open(os.path.join(ROOT, "runners/probe_train_memory_edge.py")).read()
    tree = ast.parse(src)
    parents = _parents(tree)
    uses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "future" and isinstance(node.ctx, ast.Load):
            uses.append(ast.unparse(parents[parents[node]] if isinstance(parents[node], ast.Subscript)
                                    else parents[node]))
    assert sorted(uses) == ['future[\'ego_future_xyz\'].shape', 'sub.update(future)'], uses
    assert "future_end_xyz" not in src and "values_written\": False" in src
    # blocks 1 to 3 never load a future, and no Edge probe reads the pose tensors directly
    for script in ("probe_tokens_edge.py", "probe_blackout_edge.py", "probe_nav_edge.py"):
        other = open(os.path.join(ROOT, "runners", script)).read()
        assert "load_future_poses" not in other and "ego_future" not in other and "ego_history" not in other
    # in the shared module the pose tensors appear only where they are loaded or handed to
    # the release (unsteered run: history tokens fused into the prompt)
    tree = ast.parse(open(os.path.join(ROOT, "runners/probe_edge_common.py")).read())
    owners = set()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith(("ego_future", "ego_history"))
               for n in ast.walk(fn) if not (isinstance(n, ast.Constant) and n is ast.get_docstring(fn))):
            owners.add(fn.name)
    assert owners == {"load_future_poses", "run_unconstrained"}, owners


class _Arr:
    """Tiny stand-in for a tensor: what sample_report and prompt_report touch."""

    def __init__(self, a):
        self.a = np.asarray(a)
        self.shape = self.a.shape

    def tolist(self):
        return self.a.tolist()

    def reshape(self, *shape):
        return _Arr(self.a.reshape(*shape))


def test_sample_and_prompt_reports_carry_no_pose():
    cams, ts = _native_layout()
    sentinel = 7.13579
    sample = {"image_frames": _Arr(np.zeros((22, 1, 3, 4, 8), dtype=np.uint8)),
              "camera_indices": _Arr(cams), "absolute_timestamps": _Arr(ts),
              "ego_history_xyz": _Arr(np.full((1, 20, 3), sentinel)),
              "ego_history_rot": _Arr(np.full((1, 20, 3, 3), sentinel)),
              "ego_future_xyz": _Arr(np.full((1, 64, 3), sentinel)),
              "provenance": {"dataset": "nvidia/PhysicalAI-Autonomous-Vehicles", "clip_id": "c", "t0_us": 5}}
    report = ec.sample_report(sample)
    dumped = json.dumps(report)
    assert "7.13579" not in dumped and ec.forbidden_keys(report) == []
    assert report["images_per_camera"] == {"0": 3, "1": 5, "2": 3, "3": 3, "5": 3, "6": 5}
    assert report["image_frames_shape"] == [22, 1, 3, 4, 8]

    # history tokens that would fall in the reported end of the prompt are withheld
    tok = FakeTokenizer()
    h0 = IDS["history_id0"]

    class Model:
        class config:                                                   # noqa: N801
            traj_ids = TRAJ_IDS
            history_vocab_size = ec.HISTORY_VOCAB

            class vlm_config:                                           # noqa: N801
                image_token_id = IDS["image_token_id"]

        class tokenizer:                                                # noqa: N801
            decode = staticmethod(lambda ids, skip_special_tokens=False: tok.decode(ids))
            convert_ids_to_tokens = staticmethod(lambda ids: [tok.token(i) for i in ids])

    text = tok.ids("output the future trajectory.", "<think>", "</think>")
    prompt = [IDS["image_token_id"]] * 3 + [h0 + 5, h0 + 999] + text
    rep = ec.prompt_report(prompt, Model, tail=len(text) + 2)
    assert rep["history_tokens_withheld_from_tail"] == 2 and rep["tail_ids"] == text
    assert all(not h0 <= t < h0 + ec.HISTORY_VOCAB for t in rep["tail_ids"])
    assert rep["n_image_tokens"] == 3 and rep["prompt_len"] == len(prompt)
    assert rep["ends_with_closed_thinking"] and ec.forbidden_keys(rep) == []
    assert ec.prompt_report(prompt, Model, tail=3)["history_tokens_withheld_from_tail"] == 0


def test_probe_summaries_hold_no_ground_truth_key():
    names = [c.name for c in ec.EDGE_FALLBACK_CONFIGS]
    res = pc.run_with_fallback(ec.EDGE_FALLBACK_CONFIGS, _fake_attempt({n: "ok" for n in names}))
    assert ec.forbidden_keys(pmem.summarize(res, 9.9, 79.6)) == []
    assert ec.forbidden_keys(res) == []
    tok = FakeTokenizer()
    assert ec.forbidden_keys(_cand(tok, "Keep lane.")) == []
    dev = ec.deviation(np.stack([_line(8.0, 9.0)] * 2), np.stack([_line(8.0, -9.0)] * 2))
    assert ec.forbidden_keys(dev) == []
    assert ec.forbidden_keys(pnav.follow_report("Turn left in 11m", "Turn right in 11m", dev,
                                                ["Turn left."], ["Turn right."], 1.0)) == []


# --------------------------------------------------------------------------- contracts

EDGE_SCRIPTS = ("probe_tokens_edge.py", "probe_blackout_edge.py", "probe_nav_edge.py",
                "probe_train_memory_edge.py")


def test_edge_probe_modules_import_without_torch():
    assert "torch" not in sys.modules
    assert "alpamayo2_edge" not in sys.modules and "alpamayo2edge_sft" not in sys.modules


def test_previous_probes_are_kept():
    for name in ("probe_tokens_a2.py", "probe_train_memory_a2.py", "probe_blackout_a15.py"):
        assert os.path.exists(os.path.join(ROOT, "runners", name)), name


def test_session_02_commands_use_existing_flags():
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    calls = re.findall(r"runners/(\w+\.py)((?:[^\n]*\\\n)*[^\n]*)", doc)
    assert {c[0] for c in calls} >= set(EDGE_SCRIPTS)
    common = open(os.path.join(ROOT, "runners/probe_edge_common.py")).read()
    for script, rest in calls:
        src = open(os.path.join(ROOT, "runners", script)).read()
        if "add_common_args(ap)" in src:
            src += common
        for flag in set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", rest)):
            assert f'"{flag}"' in src, f"{script} has no {flag} (used in gpu_session_02.md)"


def test_session_02_names_the_pins_and_the_five_blocks():
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    for needed in (ec.RECIPES_COMMIT, ec.EDGE_REPO_COMMIT, ec.HF_REVISION, ec.PAI_REVISION,
                   "UV_HTTP_TIMEOUT=900", "uv sync --locked", "nav_demo_samples.json",
                   "alpamayo-faithfulness"):
        assert needed in doc, needed
    for block in range(6):
        assert f"## Bloc {block} " in doc, block


def test_session_02_closing_publishes_nothing_and_states_what_is_publishable():
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    closing = doc[doc.index("## Clôture"):doc.index("## Ce qui est publiable")]
    commands = "\n".join(re.findall(r"```bash\n(.*?)```", closing, re.S))
    for forbidden in ("git push", "git add", "git commit", "git checkout"):
        assert forbidden not in commands, forbidden
    assert "tar czf $WS/session02_outputs.tgz -C $OUT --exclude=samples ." in commands
    assert "runpodctl send $WS/session02_outputs.tgz" in commands
    assert closing.index("runpodctl send") < closing.index("nvidia-smi")     # archive out before the pod stops
    assert "avant** d'arrêter le pod" in closing
    # no command of the whole session publishes
    all_commands = "\n".join(re.findall(r"```bash\n(.*?)```", doc, re.S))
    assert "git push" not in all_commands and "git commit" not in all_commands
    section = doc[doc.index("## Ce qui est publiable"):doc.index("## Ce que la séance ne fait pas")]
    for needed in ("D-012", "minADE du bloc 5", "eval_base", "vérité terrain", "pred_xyz",
                   "write_probe_json", "samples/"):
        assert needed in section, needed
    publishable, private = [], []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 2 and not set(cells[0]) <= set("-"):
            publishable.append(cells[0])
            private.append(cells[1])
    assert any("CoC" in c for c in publishable) and any("trajectoires prédites" in c for c in publishable)
    assert any("mémoire" in c for c in publishable) and any("écarts" in c for c in publishable)
    assert any("minADE" in c for c in private) and any("poses" in c for c in private)
    assert not any("minADE" in c for c in publishable[1:])


def test_session_02_runs_the_api_check_on_the_installed_sources():
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    block0 = doc[doc.index("## Bloc 0 "):doc.index("## Bloc 1 ")]
    assert block0.index("uv sync --active --locked") < block0.index("tests/test_edge_pinned_api_cold.py")
    (command,) = re.findall(r"(EDGE_INSTALLED=1(?:[^\n]*\\\n)*[^\n]*)", block0)
    for needed in ("EDGE_INSTALLED=1", "RECIPES_REPO=$REPO_DIR", "tests/test_edge_pinned_api_cold.py",
                   "--fail-on-skip", "false; }"):
        assert needed in command, needed
    assert "| tee" not in command                    # a pipe would hide the exit status
    conftest = open(os.path.join(ROOT, "tests/conftest.py")).read()
    assert '"--fail-on-skip"' in conftest and "EDGE_INSTALLED" in conftest


def test_session_02_recommends_the_h100_and_lists_three_blackout_conditions():
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    (gpu_row,) = [line for line in doc.splitlines() if line.startswith("| GPU |")]
    assert "H100 80 Go, recommandé" in gpu_row and "A100 80 Go en repli" in gpu_row
    rows = re.findall(r"^\| `(blackout_\w+)` \|", doc, re.M)
    assert rows == list(ec.BLACKOUT_CONDITIONS)
    tpl = open(os.path.join(ROOT, "docs/observations/TEMPLATE-session02.md")).read()
    assert "| Mesure | `blackout_all` | `blackout_front_both` | `blackout_front` |" in tpl
    assert tpl.count("(non publiable)") >= 3 and "D-012" in tpl


def test_session_02_template_has_one_section_per_block_with_the_four_fields():
    tpl = open(os.path.join(ROOT, "docs/observations/TEMPLATE-session02.md")).read()
    sections = re.split(r"\n## ", tpl)[1:]
    titles = [s.splitlines()[0] for s in sections]
    for needed in ("Bloc 1 ", "Bloc 2 ", "Bloc 3 ", "Bloc 4 ", "Bloc 5 ", "Décisions"):
        assert any(t.startswith(needed) for t in titles), needed
    for s in sections:
        for field in ("**Question.**", "**Valeur observée.**", "**Commande et JSON source.**", "**Conclusion.**"):
            assert field in s, (s.splitlines()[0], field)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
