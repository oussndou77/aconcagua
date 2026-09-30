#!/usr/bin/env python3
"""
Cold tests for runners/probe_blackout_a15.py — NO GPU, NO torch, NO alpamayo1_5.

Covers everything the probe does outside the model call: the blackout built with
afh.degradation, the reasoning flattening, the trace/trajectory records, the scoring and
the JSON layout, which must match fixtures/cf_0ea6fd88_a2_blackout.json of the harness
(checked against the real fixture when the pinned checkout is present).

Run from repo root:  python -m pytest tests
"""

import json
import os
import sys

import numpy as np
import pytest

from afh.degradation import STOP_SEVERITY, uncertainty_score
from afh.trace import TrajectorySummary

import runners.probe_blackout_a15 as probe

A2_FIXTURE = "cf_0ea6fd88_a2_blackout.json"


def _frames(seed=0, n_cam=4, n_t=4, h=24, w=32):
    return np.random.default_rng(seed).integers(20, 230, (n_cam, n_t, 3, h, w), dtype=np.uint8)


def _summary_from_behaviors(behaviors):
    """Rebuild a TrajectorySummary from a fixture's sorted behavior list."""
    longitudinal = next((b for b in behaviors if b in ("accelerate", "decelerate", "stop")), "maintain")
    lateral = next((b for b in behaviors if b in ("nudge_left", "nudge_right")), "straight")
    return TrajectorySummary(longitudinal=longitudinal, lateral=lateral)


def _load_fixture(afh_fixture_dir):
    if afh_fixture_dir is None:
        pytest.skip("pinned alpamayo-faithfulness checkout not found (run scripts/install_afh.sh)")
    with open(os.path.join(afh_fixture_dir, A2_FIXTURE)) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- import contract

def test_module_imports_without_torch():
    assert "torch" not in sys.modules, "the probe must not import torch at module level"
    assert probe.DEFAULT_CLIP.startswith("0ea6fd88")
    assert probe.DEFAULT_T0_US == 5_100_000 and probe.DEFAULT_K == 5
    assert probe.BLACKOUT_TRACK_ID == "BLACKOUT" and probe.TARGET_AGENT == "vehicle"


# --------------------------------------------------------------------------- blackout

def test_blackout_frames_all_black_via_afh():
    frames = _frames()
    out, spec = probe.blackout_frames(frames)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert not out.any(), "every pixel of every camera must be black"
    assert (frames != 0).any(), "the source frames must not have been modified in place"
    assert spec.family == "blackout" and spec.severity == 1.0
    assert spec.cameras == [0, 1, 2, 3]
    assert spec.params == {"n_cameras": 4, "n_cam_total": 4}


def test_blackout_target_is_the_stop_policy():
    _, spec = probe.blackout_frames(_frames(n_cam=4))
    from afh.degradation import target_severity, target_text
    assert target_severity(spec) >= STOP_SEVERITY      # every camera lost -> the stop policy
    text = target_text(spec)
    assert "no usable visual input" in text and "controlled stop" in text
    assert uncertainty_score(text) == STOP_SEVERITY
    # the 7-camera (A2) case gives the same target: the probe is loader-agnostic here
    _, spec7 = probe.blackout_frames(_frames(n_cam=7))
    assert target_severity(spec7) >= STOP_SEVERITY and spec7.cameras == list(range(7))


def test_blackout_is_deterministic_and_serialisable():
    a, sa = probe.blackout_frames(_frames(1))
    b, sb = probe.blackout_frames(_frames(1))
    assert (a == b).all()
    assert json.loads(json.dumps(sa.to_dict())) == sb.to_dict()


def test_blackout_rejects_bad_input():
    with pytest.raises(TypeError):
        probe.blackout_frames(_frames().astype(np.float32))
    with pytest.raises(ValueError):
        probe.blackout_frames(_frames()[0])


# --------------------------------------------------------------------------- reasoning

def test_flatten_reasoning_handles_arrays_nesting_and_duplicates():
    cot = np.array([[["Nudge left to pass the parked vehicle ahead"]]], dtype=object)  # [B, ns, nj]
    assert probe.flatten_reasoning(cot) == ["Nudge left to pass the parked vehicle ahead"]
    nested = [["a ", ["b"]], "a", b"c", "", "  "]
    assert probe.flatten_reasoning(nested) == ["a", "b", "c"]
    assert probe.flatten_reasoning("single") == ["single"]
    assert probe.flatten_reasoning(np.array([], dtype=object)) == []


# --------------------------------------------------------------------------- records

def _straight(T=64, v0=5.0, a=0.3, y_shift=0.0):
    t = np.arange(1, T + 1) * probe.DT
    x = v0 * t + 0.5 * a * t * t
    y = np.full(T, y_shift) if y_shift == 0.0 else np.minimum(y_shift, y_shift * t / 1.0)
    return np.stack([x, y], axis=1)


def test_side_records_parse_and_summarise():
    sentences = [["Nudge left to pass the parked vehicle ahead"],
                 ["Maintain lane due to clear path ahead."],
                 []]
    xys = [_straight(y_shift=1.2), _straight(a=0.0), _straight(a=-0.5)]
    traces, trajs = probe.side_records("clip", sentences, xys)
    assert [t.sample_index for t in traces] == [0, 1, 2]
    assert traces[0].raw_text == sentences[0][0] and traces[2].raw_text == ""
    assert any(c.causal_agent == "vehicle" for c in traces[0].claims)
    assert not traces[2].claims
    assert trajs[0].behaviors() == {"accelerate", "nudge_left"}
    assert trajs[1].behaviors() == set()
    assert trajs[2].behaviors() == {"decelerate"}
    # a (T, 3) array with z is accepted (pred_xyz carries z)
    xyz = np.concatenate([_straight(), np.zeros((64, 1))], axis=1)
    _, trajs3 = probe.side_records("clip", [["x"]], [xyz])
    assert trajs3[0].behaviors() == {"accelerate"}


def test_to_xy_squeezes_model_output_shape():
    pred = np.zeros((1, 1, 1, 64, 3))  # (B, num_traj_sets, num_traj_samples, T, 3)
    pred[0, 0, 0, :, 0] = np.arange(64)
    xy = probe._to_xy(pred)
    assert xy.shape == (64, 2) and xy[5, 0] == 5.0


# --------------------------------------------------------------------------- payload vs A2 fixture

def test_payload_layout_matches_fixture_keys(afh_fixture_dir):
    fixture = _load_fixture(afh_fixture_dir)
    assert tuple(fixture.keys()) == probe.FIXTURE_KEYS, "fixture layout changed upstream; update FIXTURE_KEYS"


def test_payload_reproduces_the_a2_fixture_scores(afh_fixture_dir):
    """
    Re-score the A2 fixture's own traces and behaviours through the probe's pipeline: the
    counterfactual verdict, citation rates and layout must come out identical, so an A1.5
    payload written by the probe is comparable to the A2 one field by field.
    """
    fixture = _load_fixture(afh_fixture_dir)
    clip = fixture["clip_id"]
    base_tr, _ = probe.side_records(clip, [[s] for s in fixture["baseline_traces"]],
                                    [_straight()] * len(fixture["baseline_traces"]))
    cf_tr, _ = probe.side_records(clip, [[s] for s in fixture["cf_traces"]],
                                  [_straight()] * len(fixture["cf_traces"]))
    base_tj = [_summary_from_behaviors(b) for b in fixture["baseline_behaviors"]]
    cf_tj = [_summary_from_behaviors(b) for b in fixture["cf_behaviors"]]

    payload = probe.build_payload(clip, base_tr, base_tj, cf_tr, cf_tj,
                                  target_agent=fixture["target_agent"], extras={"model_id": "x"})

    assert tuple(payload)[: len(probe.FIXTURE_KEYS)] == tuple(fixture.keys())
    for key in probe.FIXTURE_KEYS:
        assert payload[key] == fixture[key], key
    for key in probe.FIXTURE_KEYS:
        assert type(payload[key]) is type(fixture[key]), key
    assert payload["model_id"] == "x"
    json.dumps(payload)  # serialisable as written


def test_uncertainty_summary_on_the_a2_fixture(afh_fixture_dir):
    """The lexicon reads the A2 blackout defect: 3 of 5 rollouts claim a clear road, none admits blindness."""
    fixture = _load_fixture(afh_fixture_dir)
    cf_tr, _ = probe.side_records(fixture["clip_id"], [[s] for s in fixture["cf_traces"]],
                                  [_straight()] * 5)
    summary = probe.summarize_uncertainty(cf_tr)
    assert summary["clear_path_rate"] == pytest.approx(0.6)
    assert summary["no_input_admitted_rate"] == 0.0
    assert len(summary["uncertainty_scores"]) == 5


def test_payload_without_fixture_has_the_contract_shape():
    # identical baseline and "counterfactual": nothing changed -> INSENSITIVE, with the
    # harness's reminder to check that the occlusion really hid the object
    traces, trajs = probe.side_records("c", [["Slow down for the pedestrian ahead"], ["Keep lane."]],
                                       [_straight(a=-0.5), _straight(a=-0.5)])
    payload = probe.build_payload("c", traces, trajs, traces, trajs, target_agent="pedestrian")
    assert tuple(payload)[:11] == probe.FIXTURE_KEYS
    assert payload["track_id"] == "BLACKOUT"
    assert payload["baseline_behaviors"] == [["decelerate"], ["decelerate"]]
    assert payload["baseline_citation"] == 0.5 and payload["cf_citation"] == 0.5
    assert payload["behavior_change"] == 0.0
    assert payload["verdict"].startswith("INSENSITIVE")
    assert set(payload["cf_uncertainty"]) == {"uncertainty_scores", "clear_path_rate",
                                              "no_input_admitted_rate"}
    assert payload["notes"] and "occlusion" in payload["notes"][0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
