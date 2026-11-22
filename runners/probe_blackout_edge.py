#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_blackout_edge.py — Alpamayo 2 Edge with black cameras (gpu_session_02, block 2)

Same events, same seed and same release path as block 1 (probe_tokens_edge.py), with images
replaced by black frames before preprocessing:

    blackout_all          every image of every camera is black;
    blackout_front_both   both forward cameras are black: front wide (id 1) and front
                          telephoto (id 6); the four side and rear cameras stay;
    blackout_front        only the front wide camera (id 1) is black; the telephoto stays.

They run in that order, from no camera at all to a single camera missing.

Images are selected by camera identifier on Edge's flat list of 22 images (one identifier
per image). `afh.degradation` is not used: it assumes a camera x time tensor (D-010 update).

For each event and condition it records the same outputs as block 1 (raw tokens, reasoning
and its length, trajectories) plus the deviation from the clean run of block 1: distance
between mean trajectories, paired distance candidate by candidate (same seed), and the
spread among clean candidates as the floor to compare with. The reasoning texts are kept
verbatim, clean and blacked-out side by side; this probe does not score them.

The clean run comes from --baseline-json (block 1 output). Without it the probe runs the
clean condition itself first.

Usage:
    python runners/probe_blackout_edge.py --events $EVENTS \\
        --baseline-json outputs/session02/tokens_edge.json --out outputs/session02/blackout_edge.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_edge_common import (  # noqa: E402
    BLACKOUT_CONDITIONS,
    add_common_args,
    baseline_cots,
    baseline_trajectories,
    deviation,
    load_event_sample,
    open_session,
    read_json,
    run_condition,
    summarize_reasoning,
    with_blackout,
    write_probe_json,
)


def summarize(payload: dict, conditions: list[str]) -> dict:
    """Per condition: how many events deviate beyond the clean spread, mean deviation,
    reasoning lengths, path lengths (pure; tested cold)."""
    out: dict = {}
    for name in conditions:
        devs, cands, rows, missing = [], [], [], []
        for ev in payload.get("events", []):
            res = (ev.get("conditions") or {}).get(name)
            if not res or res.get("status") == "error":
                missing.append(ev.get("index"))
                continue
            cands += res.get("candidates") or []
            dev = res.get("deviation_vs_clean")
            if dev:
                devs.append(dev)
            rows.append({"index": ev.get("index"), "status": res.get("status"),
                         "deviation_vs_clean": dev,
                         "cot_clean": (res.get("cot_clean") or [None])[0],
                         "cot_condition": ((res.get("candidates") or [{}])[0]).get("cot")})
        flags = [d["exceeds_baseline_spread"] for d in devs if d["exceeds_baseline_spread"] is not None]
        out[name] = {
            "events_with_output": len(rows), "events_missing": missing,
            "events_with_deviation": len(devs),
            "n_exceeding_clean_spread": sum(flags), "n_compared_with_spread": len(flags),
            "mean_ade_between_means_m": round(float(np.mean([d["ade_between_means_m"] for d in devs])), 3) if devs else None,
            "mean_clean_within_spread_m": (round(float(np.mean([d["baseline_within_spread_m"] for d in devs
                                                                 if d["baseline_within_spread_m"] is not None])), 3)
                                           if flags else None),
            "mean_path_length_clean_m": round(float(np.mean([d["path_length_baseline_m"] for d in devs])), 3) if devs else None,
            "mean_path_length_condition_m": round(float(np.mean([d["path_length_condition_m"] for d in devs])), 3) if devs else None,
            "reasoning": summarize_reasoning(cands),
            "per_event": rows,
        }
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--conditions", nargs="+", default=list(BLACKOUT_CONDITIONS),
                    choices=list(BLACKOUT_CONDITIONS))
    ap.add_argument("--baseline-json", default=None,
                    help="block-1 output; without it the clean condition is run here first")
    ap.add_argument("--out", default="outputs/session02/blackout_edge.json")
    args = ap.parse_args(argv)

    baseline = read_json(args.baseline_json)
    base_xyz, base_cot = baseline_trajectories(baseline), baseline_cots(baseline)
    payload, model, recorder, avdi, events = open_session(args, "probe_blackout_edge")
    payload["baseline"] = {"json": args.baseline_json, "events_with_clean_trajectory": len(base_xyz),
                           "seed_matches": None if baseline is None
                           else (baseline.get("args") or {}).get("seed") == args.seed}
    out = args.out
    for event in events:
        sample, entry = load_event_sample(event, avdi, args.sample_cache)
        payload["events"].append(entry)
        if sample is None:
            write_probe_json(out, payload)
            continue
        clean_xyz, clean_cot = base_xyz.get(event.key), base_cot.get(event.key)
        if clean_xyz is None:
            clean = run_condition(model, sample, recorder, args)
            entry["conditions"]["clean"] = clean
            clean_xyz = clean.get("pred_xyz")
            clean_cot = [c["cot"] for c in clean.get("candidates") or []]
        for name in args.conditions:
            black, positions = with_blackout(sample, BLACKOUT_CONDITIONS[name])
            res = run_condition(model, black, recorder, args)
            res["blacked_out"] = {"camera_ids": list(BLACKOUT_CONDITIONS[name]),
                                  "image_positions": positions, "n_images": len(positions)}
            res["cot_clean"] = clean_cot
            if clean_xyz and res.get("pred_xyz"):
                res["deviation_vs_clean"] = deviation(clean_xyz, res["pred_xyz"])
            entry["conditions"][name] = res
            write_probe_json(out, payload)
            rows = res.get("candidates") or []
            print(f"[event {event.index}] {name}: status={res.get('status')} "
                  f"deviation={res.get('deviation_vs_clean')}")
            if rows:
                print(f"    cot[0]: {rows[0]['cot'][:160]!r}")

    payload["summary"] = summarize(payload, args.conditions)
    write_probe_json(out, payload)
    for name, s in payload["summary"].items():
        print(f">>> {name}: {s['n_exceeding_clean_spread']}/{s['n_compared_with_spread']} events beyond "
              f"the clean spread; mean deviation {s['mean_ade_between_means_m']} m")
    print(f"Saved -> {out}")
    return payload


if __name__ == "__main__":
    main()
