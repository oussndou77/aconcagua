#!/usr/bin/env python3
"""
Cold tests for the GPU-session-01 probes — NO GPU, NO torch, NO alpamayo2_super.

Covers what the probes do outside the model: reading a generated token sequence without
presupposing its format (first tokens, <|cot_start|> verdict, tagged and untagged meta-actions,
CoC handed to block 2), the generate() recorder, the selection of the trained parameters, the
OOM fallback loop and its JSON, and that every command in docs/gpu_session_01.md only uses
flags the scripts define.

Fake outputs follow two shapes: the untagged one NVlabs/alpamayo2 #18 uses in its own tests
(tests/test_text_extraction.py:170-181) and the tagged one build_conversation produces in
training mode (D-008).
"""

import json
import os
import re
import sys

import pytest

import runners.probe_common as pc
import runners.probe_tokens_a2 as ptok
import runners.probe_train_memory_a2 as pmem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- fake tokenizer

class FakeTokenizer:
    """Special tokens get the ids of the release checkpoint; any other character is one token."""

    pad_token_id = 151643
    unk_token_id = None

    def __init__(self, specials=None):
        self.specials = dict(pc.EXPECTED_TOKEN_IDS if specials is None else specials)
        self.inv = {v: k for k, v in self.specials.items()}

    def encode(self, text):
        ids, i = [], 0
        names = sorted(self.specials, key=len, reverse=True)
        while i < len(text):
            for name in names:
                if text.startswith(name, i):
                    ids.append(self.specials[name])
                    i += len(name)
                    break
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(i) for i in ids]
        return self.inv.get(ids, chr(ids - 1000) if 1000 <= ids < 1000 + 0x110000 else f"<id{ids}>")

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(self.convert_ids_to_tokens(int(i)) for i in ids)


def marker_ids(tok):
    return {k: tok.convert_tokens_to_ids(v) for k, v in pc.MARKER_TOKENS.items()}


def analyze(tok, text, **kw):
    return pc.analyze_generation(tok.encode(text), tok.convert_ids_to_tokens,
                                 lambda ids: tok.decode(list(ids)), marker_ids(tok), **kw)


UNTAGGED = ("A red light is ahead.\nLongitudinal: brake\nLateral: straight\nLane: keep"
            "<|traj_future_start|>")
TAGGED = ("<|cot_start|>Slow down for the pedestrian crossing ahead.<|cot_end|>"
          "<|meta_action_start|>Longitudinal: decelerate\nLateral: straight\nLane: keep"
          "<|meta_action_end|><|traj_future_start|>")


# --------------------------------------------------------------------------- generation analysis

def test_untagged_generation_reports_absent_and_untagged_meta_actions():
    tok = FakeTokenizer()
    r = analyze(tok, UNTAGGED)
    assert r["cot_start"]["verdict"] == "absent"
    assert r["cot_start"]["first_position"] is None
    assert "does not appear" in r["cot_start"]["statement"]
    assert r["cot_tagged_block"] is None and r["meta_action_tagged_block"] is None
    assert r["meta_actions"] == [
        {"axis": "Longitudinal", "value": "brake", "source": "untagged"},
        {"axis": "Lateral", "value": "straight", "source": "untagged"},
        {"axis": "Lane", "value": "keep", "source": "untagged"},
    ]
    assert r["cot_for_training"] == "A red light is ahead."
    assert r["cot_source"].startswith("untagged")
    assert r["marker_positions"]["traj_future_start"] == [len(UNTAGGED) - len("<|traj_future_start|>")]
    assert r["text_before_traj_future_start"].endswith("Lane: keep")


def test_first_ten_tokens_with_ids_tokens_and_text():
    tok = FakeTokenizer()
    r = analyze(tok, UNTAGGED)
    assert len(r["first_tokens"]) == 10
    assert [t["position"] for t in r["first_tokens"]] == list(range(10))
    assert "".join(t["text"] for t in r["first_tokens"]) == UNTAGGED[:10]
    assert r["first_tokens"][0] == {"position": 0, "id": 1000 + ord("A"), "token": "A", "text": "A"}
    assert len(analyze(tok, UNTAGGED, first_n=3)["first_tokens"]) == 3


def test_tagged_generation_first_token_and_tagged_blocks():
    tok = FakeTokenizer()
    r = analyze(tok, TAGGED)
    assert r["cot_start"]["verdict"] == "first_token"
    assert r["cot_start"]["first_position"] == 0
    assert r["first_tokens"][0] == {"position": 0, "id": 155677, "token": "<|cot_start|>",
                                    "text": "<|cot_start|>"}
    assert r["cot_tagged_block"] == "Slow down for the pedestrian crossing ahead."
    assert r["cot_for_training"] == r["cot_tagged_block"] and r["cot_source"] == "tagged_block"
    assert r["meta_action_tagged_block"].startswith("Longitudinal: decelerate")
    assert [m["source"] for m in r["meta_actions"]] == ["tagged"] * 3
    assert r["meta_actions"][0]["value"] == "decelerate"
    pos = r["marker_positions"]
    assert pos["cot_start"] == [0] and len(pos["cot_end"]) == 1
    assert pos["cot_end"][0] < pos["meta_action_start"][0] < pos["meta_action_end"][0] \
        < pos["traj_future_start"][0]


def test_cot_start_generated_later_is_reported_with_its_position():
    tok = FakeTokenizer()
    r = analyze(tok, "Ok.<|cot_start|>Keep lane.<|cot_end|><|traj_future_start|>")
    assert r["cot_start"]["verdict"] == "later"
    assert r["cot_start"]["first_position"] == 3
    assert "position 3" in r["cot_start"]["statement"]


def test_tokenizer_without_cot_start_cannot_answer():
    specials = {k: v for k, v in pc.EXPECTED_TOKEN_IDS.items() if k != "<|cot_start|>"}
    tok = FakeTokenizer(specials)
    r = analyze(tok, UNTAGGED)
    assert r["cot_start"]["verdict"] == "unknown_token"
    assert r["cot_start"]["token_id"] is None


def test_unterminated_tagged_block_and_im_end_cut():
    tok = FakeTokenizer()
    r = analyze(tok, "<|cot_start|>Yield to the cyclist")
    assert r["cot_tagged_block"] == "Yield to the cyclist"
    r2 = analyze(tok, "Clear road ahead.<|im_end|>")
    assert r2["cot_for_training"] == "Clear road ahead."
    assert r2["meta_actions"] == [] and r2["marker_positions"]["im_end"] == [17]


def test_discrete_trajectory_tokens_are_counted_not_decoded():
    tok = FakeTokenizer()
    ids = tok.encode(UNTAGGED) + [152669 + 5, 151669]
    r = pc.analyze_generation(ids, tok.convert_ids_to_tokens, lambda x: tok.decode(list(x)), marker_ids(tok))
    assert r["n_discrete_trajectory_tokens"] == 2


def test_parse_meta_actions_and_untagged_cot_helpers():
    assert pc.parse_meta_actions(None) == []
    assert pc.parse_meta_actions("Longitudinal:  accelerate \nLane:keep") == [
        {"axis": "Longitudinal", "value": "accelerate"}, {"axis": "Lane", "value": "keep"}]
    assert pc.untagged_cot("Lane: keep") == ""
    assert pc.untagged_cot("Go.<|traj_future_start|>Lane: x") == "Go."


def test_check_token_ids_reports_mismatch_without_failing():
    ok = pc.check_token_ids(dict(pc.EXPECTED_TOKEN_IDS))
    assert ok["all_match"] and ok["mismatches"] == {}
    bad = pc.check_token_ids({**pc.EXPECTED_TOKEN_IDS, "<|cot_start|>": 1})
    assert not bad["all_match"]
    assert bad["mismatches"] == {"<|cot_start|>": {"expected": 155677, "live": 1}}


def test_expected_ids_match_the_checkpoint_config():
    with open(os.path.join(ROOT, "docs/configs/alpamayo2-super/config.json")) as fh:
        traj = json.load(fh)["traj_ids"]
    assert pc.EXPECTED_TOKEN_IDS["<|traj_future_start|>"] == traj["future_start"]
    assert pc.EXPECTED_TOKEN_IDS["<|traj_future_end|>"] == traj["future_end"]
    assert pc.EXPECTED_TOKEN_IDS["<|traj_history|>"] == traj["history_pad"]
    assert pc.EXPECTED_TOKEN_IDS["<|traj_future|>"] == traj["future_pad"]
    assert pc.HISTORY_ID0 == traj["history_id0"] and pc.FUTURE_ID0 == traj["future_id0"]
    # history_start 155674 + 3 positions (pre_tkn, end, then cot_start) in SPECIAL_TOKENS order
    assert pc.EXPECTED_TOKEN_IDS["<|cot_start|>"] == traj["history_start"] + 3


# --------------------------------------------------------------------------- token probe glue

class FakeTensor:
    """The few tensor methods GenerateRecorder and _analyze_call touch."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __getitem__(self, i):
        return FakeTensor([self.rows[i]]) if isinstance(i, int) else NotImplemented

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.rows)

    def cpu(self):
        return self

    def tolist(self):
        return self.rows[0] if len(self.rows) == 1 else self.rows


def test_recorder_keeps_raw_sequences_and_analysis_strips_padding():
    tok = FakeTokenizer()
    prompt = tok.encode("<|im_start|>assistant\n")
    gen = tok.encode(UNTAGGED)
    seq = FakeTensor([prompt + gen + [tok.pad_token_id] * 3])

    class Out:
        sequences = seq

    rec = ptok.GenerateRecorder(lambda **kw: Out())
    out = rec(input_ids=FakeTensor([prompt]))
    assert out.sequences is seq
    call = rec.calls[0]
    assert call["prompt_len"] == len(prompt) and call["prompt_ids"] == prompt
    seq.rows[0][len(prompt)] = 0                 # the release overwrites in place afterwards
    assert call["sequences"].rows[0][len(prompt)] == gen[0], "recorder must keep a copy"

    rep = ptok._analyze_call(call, tok, marker_ids(tok), first_n=10)
    assert rep["trailing_pad_tokens_removed"] == 3
    assert rep["generated_ids"] == gen
    assert rep["cot_start"]["verdict"] == "absent"
    pr = ptok._prompt_report(prompt, tok)
    assert pr["tail_tokens"][-1] == "\n" and pr["contains"]["cot_start"] is False


# --------------------------------------------------------------------------- memory probe helpers

LM = "vlm.model.language_model.layers.{i}.self_attn.{p}"
NAMES = (
    [LM.format(i=i, p=p) for i in range(64) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    + [f"vlm.model.language_model.layers.{i}.mlp.gate_proj" for i in range(64)]
    + [f"vlm.model.visual.blocks.{i}.attn.qkv" for i in range(27)]
    + [f"vlm.model.visual.blocks.{i}.attn.proj" for i in range(27)]
    + [f"expert.expert.layers.{i}.self_attn.q_proj" for i in range(64)]
)


def test_select_last_layer_q_proj_of_the_language_model_only():
    sel = pc.select_last_layer_modules(NAMES, n_last=4)
    assert sel == [LM.format(i=i, p="q_proj") for i in (60, 61, 62, 63)]
    assert not any("expert" in n or "visual" in n for n in sel)
    # numeric, not lexicographic, layer order
    assert pc.select_last_layer_modules([LM.format(i=i, p="q_proj") for i in (9, 10, 2)], 2) == \
        [LM.format(i=9, p="q_proj"), LM.format(i=10, p="q_proj")]
    with pytest.raises(ValueError):
        pc.select_last_layer_modules(["vlm.model.visual.blocks.0.attn.qkv"])


def test_fallback_configs_shrink_cameras_then_frames():
    cfgs = pc.FALLBACK_CONFIGS
    assert cfgs[0].camera_ids == tuple(range(7)) and cfgs[0].frame_indices == (0, 1, 2, 3)
    assert cfgs[1].camera_ids == (0, 1, 2, 3, 5, 6)          # release trajectory profile
    assert cfgs[-1].camera_ids == (1,) and cfgs[-1].frame_indices == (3,)   # recipe debug config
    n_cam = [len(c.camera_ids) for c in cfgs]
    n_f = [len(c.frame_indices) for c in cfgs]
    assert n_cam == sorted(n_cam, reverse=True)
    first_frame_cut = next(i for i, n in enumerate(n_f) if n < 4)
    assert all(n == 1 for n in n_cam[first_frame_cut:]), "frames shrink only once cameras are at 1"
    assert n_f == sorted(n_f, reverse=True)
    for c in cfgs:
        assert list(c.camera_ids) == sorted(c.camera_ids) and set(c.camera_ids) <= set(range(7))
        assert c.frame_indices[-1] == 3, "kept frames always end at t0"
    assert len({c.name for c in cfgs}) == len(cfgs)


class OutOfMemoryError(RuntimeError):
    """Same class name as torch.cuda.OutOfMemoryError."""


def test_classify_error():
    assert pc.classify_error(OutOfMemoryError("CUDA out of memory. Tried to allocate")) == "oom"
    assert pc.classify_error(RuntimeError("CUDA error: out of memory")) == "oom"
    assert pc.classify_error(ValueError("Tokenized sample length 7001 exceeds max_length=6144")) == "too_long"
    assert pc.classify_error(KeyError("pixel_values")) == "other"


def _fake_attempt(outcomes):
    """attempt_fn whose behaviour per config name is 'ok', 'oom_forward', 'oom_backward',
    'too_long' or 'bug'; it records measurements step by step like the real probe."""
    def attempt(cfg, steps):
        what = outcomes[cfg.name]
        steps["_current"] = "collate"
        if what == "too_long":
            raise ValueError("Tokenized sample length 9000 exceeds max_length=6144")
        steps["sequence"] = {"real_len": 100 * len(cfg.camera_ids) * len(cfg.frame_indices)}
        steps["_current"] = "forward"
        if what == "oom_forward":
            raise OutOfMemoryError("CUDA out of memory")
        if what == "bug":
            raise KeyError("labels_mask")
        steps["forward"] = {"max_memory_allocated_gib": 70.0 + len(cfg.camera_ids)}
        steps["_current"] = "backward"
        if what == "oom_backward":
            raise OutOfMemoryError("CUDA out of memory")
        steps["backward"] = {"max_memory_allocated_gib": 75.0}
    return attempt


def test_fallback_keeps_partial_measurements_and_stops_at_first_success():
    names = [c.name for c in pc.FALLBACK_CONFIGS]
    outcomes = {n: "ok" for n in names}
    outcomes[names[0]] = "too_long"
    outcomes[names[1]] = "oom_forward"
    outcomes[names[2]] = "oom_backward"
    updates, cleanups = [], []
    res = pc.run_with_fallback(pc.FALLBACK_CONFIGS, _fake_attempt(outcomes),
                               on_update=lambda a: updates.append(len(a)),
                               cleanup=lambda: cleanups.append(1))
    assert res["retained_config"] == names[3] and not res["stopped_on_error"]
    att = res["attempts"]
    assert [a["status"] for a in att] == ["too_long", "oom", "oom", "ok"]
    assert [a["failed_step"] for a in att] == ["collate", "forward", "backward", None]
    assert att[1]["steps"]["sequence"]["real_len"] == 2400          # measured before the OOM
    assert att[2]["steps"]["forward"]["max_memory_allocated_gib"] == 74.0   # partial = measurement
    assert "_current" not in att[3]["steps"]
    assert updates == [1, 2, 3, 4] and len(cleanups) == 4
    json.dumps(res)


def test_fallback_sweep_and_stop_on_non_oom_error():
    names = [c.name for c in pc.FALLBACK_CONFIGS]
    res = pc.run_with_fallback(pc.FALLBACK_CONFIGS, _fake_attempt({n: "ok" for n in names}), sweep=True)
    assert res["retained_config"] == names[0] and len(res["attempts"]) == len(names)
    outcomes = {n: "ok" for n in names}
    outcomes[names[0]] = "oom_forward"
    outcomes[names[1]] = "bug"
    res = pc.run_with_fallback(pc.FALLBACK_CONFIGS, _fake_attempt(outcomes))
    assert res["stopped_on_error"] and res["retained_config"] is None
    assert [a["status"] for a in res["attempts"]] == ["oom", "error"]
    assert res["attempts"][1]["error"].startswith("KeyError")


def test_fallback_every_config_oom_is_still_a_result():
    names = [c.name for c in pc.FALLBACK_CONFIGS]
    res = pc.run_with_fallback(pc.FALLBACK_CONFIGS, _fake_attempt({n: "oom_forward" for n in names}))
    assert res["retained_config"] is None and not res["stopped_on_error"]
    assert len(res["attempts"]) == len(names)
    s = pmem.summarize(res, {"memory_allocated_gib": 68.1})
    assert s["weights_gib"] == 68.1 and s["retained_config"] is None
    assert all(r["status"] == "oom" and r["real_len"] for r in s["attempts"])


def test_summarize_rows():
    names = [c.name for c in pc.FALLBACK_CONFIGS]
    outcomes = {n: "ok" for n in names}
    outcomes[names[0]] = "oom_backward"
    res = pc.run_with_fallback(pc.FALLBACK_CONFIGS, _fake_attempt(outcomes))
    s = pmem.summarize(res, {"memory_allocated_gib": 68.0})
    assert s["retained_config"] == names[1]
    assert s["attempts"][0]["peak_forward_gib"] == 77.0 and s["attempts"][0]["peak_backward_gib"] is None
    assert s["attempts"][1]["peak_backward_gib"] == 75.0


def test_count_span():
    assert pc.count_span([1, 5, 9, 9, 6, 2], 5, 6) == 2
    assert pc.count_span([1, 2], 5, 6) is None
    assert pc.count_span([5, 1], 5, 6) is None
    assert pc.count_span([5, 6], None, 6) is None


def test_read_cot_from_block1_json(tmp_path):
    p = tmp_path / "tokens.json"
    p.write_text(json.dumps({"summary": {"cot_for_training": "A red light is ahead.",
                                         "cot_source": "untagged_text_before_meta_action_or_traj"}}))
    cot, src = pmem.read_cot(str(p), None)
    assert cot == "A red light is ahead." and "untagged" in src
    assert pmem.read_cot(str(p), "override")[0] == "override"
    assert pmem.read_cot(None, None) == (None, "none")


def test_write_json_is_atomic(tmp_path):
    out = tmp_path / "sub" / "x.json"
    pc.write_json(str(out), {"a": 1, "obj": object()})
    assert json.loads(out.read_text())["a"] == 1
    assert not (tmp_path / "sub" / "x.json.tmp").exists()


def test_package_provenance_of_missing_distribution():
    assert pc.package_provenance("surely-not-installed-xyz") == {"installed": False}


# --------------------------------------------------------------------------- contracts

def test_probe_modules_import_without_torch():
    assert "torch" not in sys.modules
    assert "alpamayo2_super" not in sys.modules and "alpamayo2_sft" not in sys.modules
    assert pmem.COMPONENTS_RECIPE == ["image", "traj_history", "prompt", "traj_future"]
    assert pmem.COMPONENTS_WITH_COT[3] == "cot" and pmem.RECIPE_MAX_LENGTH == 6144


def test_session_doc_commands_use_existing_flags():
    doc = open(os.path.join(ROOT, "docs/gpu_session_01.md")).read()
    calls = re.findall(r"runners/(\w+\.py)((?:[^\n]*\\\n)*[^\n]*)", doc)
    assert {c[0] for c in calls} >= {"probe_tokens_a2.py", "probe_train_memory_a2.py",
                                     "probe_blackout_a15.py"}
    for script, rest in calls:
        src = open(os.path.join(ROOT, "runners", script)).read()
        for flag in set(re.findall(r"(--[a-z][a-z0-9-]+)", rest)):
            assert f'"{flag}"' in src, f"{script} has no {flag} (used in gpu_session_01.md)"


def test_template_has_one_section_per_unknown_with_the_four_fields():
    tpl = open(os.path.join(ROOT, "docs/observations/TEMPLATE-session.md")).read()
    sections = re.split(r"\n## ", tpl)[1:]
    titles = [s.splitlines()[0] for s in sections]
    for needed in ("Inconnue 1 ", "Inconnue 4 ", "Inconnue 5 ", "Inconnue 6 ", "Inconnue 8 ", "Go / no-go W3"):
        assert any(t.startswith(needed) for t in titles), needed
    for s in sections:
        for field in ("**Question.**", "**Valeur observée.**", "**Commande et JSON source.**", "**Conclusion.**"):
            assert field in s, (s.splitlines()[0], field)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
