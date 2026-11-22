#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_nav_edge.py — does Alpamayo 2 Edge follow the navigation instruction? (gpu_session_02, block 3)

Same events, same images, same seed and same release path as block 1; only the navigation
instruction changes. Two counterfactual instructions per event, in the release's syntax:

    mirror     left and right swapped, everything else kept ("Turn left in 11m" ->
               "Turn right in 11m");
    straight   "Continue straight".

Each instruction is checked with the recipe's own validator before use
(src/alpamayo/data/pai_metadata.py, validate_navigation); an instruction it rejects is not
run, and an unavailable validator is recorded.

For each event and instruction it records the same outputs as block 1 plus:
  * the deviation of the trajectory from the clean run (distance between mean trajectories,
    paired distance, clean spread as floor), and the lateral end point of both;
  * for `mirror`, whether the lateral end point changed sign (both beyond --min-lateral-m);
  * which direction words (left / right / straight) the reasoning uses, clean and
    counterfactual, and the texts verbatim.

The navigation instruction is a mandatory input of Edge's native task and public PAI does
not provide it (docs/edge_review.md §7): how much the output depends on it decides how
carefully instructions must be produced for our own events (ROADMAP W3).

Usage:
    python runners/probe_nav_edge.py --events $EVENTS \\
        --baseline-json outputs/session02/tokens_edge.json --out outputs/session02/nav_edge.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_edge_common import (  # noqa: E402
    add_common_args,
    baseline_cots,
    baseline_trajectories,
    deviation,
    direction_mentions,
    load_event_sample,
    nav_counterfactuals,
    open_session,
    read_json,
    run_condition,
    validate_navigation_live,
    with_navigation,
    write_probe_json,
)

LABELS = ("mirror", "straight")


def _sum_mentions(texts) -> dict[str, int]:
    total = {"left": 0, "right": 0, "straight": 0}
    for t in texts or []:
        for k, v in direction_mentions(t).items():
            total[k] += v
    return total


def follow_report(nav_original: str, nav_new: str, dev: dict | None,
                  cots_clean, cots_new, min_lateral_m: float) -> dict:
    """Facts about one counterfactual instruction (pure; tested cold).

    new_words / old_words: direction words the new instruction has and the original lacks,
    and the reverse. lateral_sign_flipped is only computed when both lateral end points are
    beyond min_lateral_m; otherwise it is None (no lateral motion to compare)."""
    o, n = direction_mentions(nav_original), direction_mentions(nav_new)
    new_words = [w for w in n if n[w] and not o[w]]
    old_words = [w for w in o if o[w] and not n[w]]
    mc, mn = _sum_mentions(cots_clean), _sum_mentions(cots_new)
    flipped = None
    if dev is not None:
        yb, yc = dev["end_y_baseline_m"], dev["end_y_condition_m"]
        if abs(yb) >= min_lateral_m and abs(yc) >= min_lateral_m:
            flipped = bool(np.sign(yb) != np.sign(yc))
    return {
        "new_direction_words": new_words, "old_direction_words": old_words,
        "cot_mentions_clean": mc, "cot_mentions_counterfactual": mn,
        "cot_names_new_direction": bool(any(mn[w] for w in new_words)) if new_words else None,
        "cot_still_names_old_direction": bool(any(mn[w] for w in old_words)) if old_words else None,
        "cot_text_changed": (None if not cots_clean or not cots_new
                             else list(cots_clean) != list(cots_new)),
        "lateral_sign_flipped": flipped, "min_lateral_m": min_lateral_m,
    }


def summarize(payload: dict) -> dict:
    """Counts per counterfactual label (pure; tested cold)."""
    out: dict = {}
    for label in LABELS:
        rows = []
        for ev in payload.get("events", []):
            res = (ev.get("conditions") or {}).get(label)
            if res:
                rows.append((ev, res))
        ran = [(ev, r) for ev, r in rows if r.get("status") in ("ok", "no_anchor")]
        devs = [r["deviation_vs_clean"] for _, r in ran if r.get("deviation_vs_clean")]
        fol = [r["follow"] for _, r in ran if r.get("follow")]

        def count(key):
            vals = [f[key] for f in fol if f[key] is not None]
            return {"true": sum(vals), "of": len(vals)}

        spread = [d["exceeds_baseline_spread"] for d in devs if d["exceeds_baseline_spread"] is not None]
        out[label] = {
            "events_run": len(ran),
            "events_skipped_or_failed": [ev.get("index") for ev, r in rows
                                         if r.get("status") not in ("ok", "no_anchor")],
            "instructions_not_validated": sum(1 for _, r in rows
                                              if not (r.get("validation") or {}).get("validated")),
            "trajectory_beyond_clean_spread": {"true": sum(spread), "of": len(spread)},
            "mean_ade_between_means_m": round(float(np.mean([d["ade_between_means_m"] for d in devs])), 3) if devs else None,
            "mean_abs_end_y_clean_m": round(float(np.mean([abs(d["end_y_baseline_m"]) for d in devs])), 3) if devs else None,
            "mean_abs_end_y_counterfactual_m": round(float(np.mean([abs(d["end_y_condition_m"]) for d in devs])), 3) if devs else None,
            "lateral_sign_flipped": count("lateral_sign_flipped"),
            "cot_names_new_direction": count("cot_names_new_direction"),
            "cot_still_names_old_direction": count("cot_still_names_old_direction"),
            "cot_text_changed": count("cot_text_changed"),
            "per_event": [{"index": ev.get("index"), "nav_original": ev.get("nav_text"),
                           "nav_counterfactual": r.get("nav_text"), "status": r.get("status"),
                           "deviation_vs_clean": r.get("deviation_vs_clean"), "follow": r.get("follow"),
                           "cot_clean": (r.get("cot_clean") or [None])[0],
                           "cot_counterfactual": ((r.get("candidates") or [{}])[0]).get("cot")}
                          for ev, r in rows],
        }
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--labels", nargs="+", default=list(LABELS), choices=list(LABELS))
    ap.add_argument("--min-lateral-m", type=float, default=1.0,
                    help="lateral end offset below which a sign flip is not assessed")
    ap.add_argument("--baseline-json", default=None,
                    help="block-1 output; without it the clean condition is run here first")
    ap.add_argument("--out", default="outputs/session02/nav_edge.json")
    args = ap.parse_args(argv)

    baseline = read_json(args.baseline_json)
    base_xyz, base_cot = baseline_trajectories(baseline), baseline_cots(baseline)
    payload, model, recorder, avdi, events = open_session(args, "probe_nav_edge")
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
        for cf in nav_counterfactuals(event.nav_text):
            label, nav = cf["label"], cf["nav_text"]
            if label not in args.labels:
                continue
            validation = validate_navigation_live(nav)
            if validation.get("validated") and not validation.get("valid"):
                entry["conditions"][label] = {"status": "invalid_navigation", "nav_text": nav,
                                              "validation": validation}
                write_probe_json(out, payload)
                continue
            res = run_condition(model, with_navigation(sample, nav), recorder, args)
            res.update(nav_text=nav, nav_text_original=event.nav_text, validation=validation,
                       cot_clean=clean_cot)
            dev = None
            if clean_xyz and res.get("pred_xyz"):
                dev = res["deviation_vs_clean"] = deviation(clean_xyz, res["pred_xyz"])
            res["follow"] = follow_report(event.nav_text, nav, dev, clean_cot,
                                          [c["cot"] for c in res.get("candidates") or []],
                                          args.min_lateral_m)
            entry["conditions"][label] = res
            write_probe_json(out, payload)
            rows = res.get("candidates") or []
            print(f"[event {event.index}] {event.nav_text!r} -> {nav!r}: status={res.get('status')} "
                  f"end_y {dev and dev['end_y_baseline_m']} -> {dev and dev['end_y_condition_m']}")
            if rows:
                print(f"    cot[0]: {rows[0]['cot'][:160]!r}")

    payload["summary"] = summarize(payload)
    write_probe_json(out, payload)
    for label, s in payload["summary"].items():
        print(f">>> {label}: trajectory beyond clean spread {s['trajectory_beyond_clean_spread']}, "
              f"lateral sign flipped {s['lateral_sign_flipped']}, "
              f"reasoning names the new direction {s['cot_names_new_direction']}")
    print(f"Saved -> {out}")
    return payload


if __name__ == "__main__":
    main()
