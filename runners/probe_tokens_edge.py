#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
probe_tokens_edge.py — what does Alpamayo 2 Edge generate? (gpu_session_02, block 1)

Runs the release inference path unchanged on the annotated events of the recipe's demo
asset (19 events: clip, time, navigation instruction):

    load_physical_aiavdataset -> helper.prepare_model_inputs
    -> Alpamayo2Edge.sample_trajectories_from_data(return_extra=True)

and keeps the raw output. The only addition is a recorder around `model.vlm.generate` that
copies the returned token ids before the release looks for the trajectory anchor. Nothing
in the prompt, the sampling or the model is touched, and the seed is fixed.

It presupposes nothing about the format. For every event and every candidate it reports:
  * the end of the prompt (does it end with the closed <think></think> block?);
  * the first generated tokens (ids, token strings, decoded text);
  * where <|traj_future_start|> is, hence the length of the reasoning before it — zero
    included ("adaptive reasoning" has no switch in the code: docs/edge_review.md §10);
  * anything else that shows up: <think> tags, EOS, discrete trajectory tokens,
    meta-action lines;
  * the release's own text extraction, for comparison;
  * the trajectory of each candidate, the spread among candidates, peak memory, time.

Two optional checks:
  --unconstrained-events N   on the first N events, one direct vlm.generate WITHOUT the
                             release's logits processors: what the model emits unsteered;
  --repro-check              re-run the first event with the same seed, compare token ids.

Environment: the recipe venv (uv sync --locked in recipes/alpamayo2edge_sft). afh is not
needed. See docs/gpu_session_02.md.

Usage:
    python runners/probe_tokens_edge.py --events $EVENTS --out outputs/session02/tokens_edge.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.probe_edge_common import (  # noqa: E402
    add_common_args,
    load_event_sample,
    open_session,
    run_condition,
    run_unconstrained,
    summarize_reasoning,
    write_probe_json,
)


def summarize(payload: dict) -> dict:
    """Block-1 summary over every candidate of every event (pure; tested cold)."""
    cands, per_event, no_anchor_events, failed = [], [], [], []
    prompt_lens, closed = [], []
    for ev in payload.get("events", []):
        clean = (ev.get("conditions") or {}).get("clean")
        if not clean or clean.get("status") == "error":
            failed.append(ev.get("index"))
            continue
        if clean.get("status") == "no_anchor":
            no_anchor_events.append(ev.get("index"))
        rows = clean.get("candidates") or []
        cands += rows
        if clean.get("prompt"):
            prompt_lens.append(clean["prompt"]["prompt_len"])
            closed.append(clean["prompt"]["ends_with_closed_thinking"])
        per_event.append({"index": ev.get("index"), "nav_text": ev.get("nav_text"),
                          "status": clean.get("status"),
                          "reasoning_tokens": [r["reasoning_tokens"] for r in rows],
                          "cot_first_candidate": rows[0]["cot"] if rows else None,
                          "within_spread_m": clean.get("within_spread_m")})
    return {
        "events_with_output": len(per_event),
        "events_failed_or_not_loaded": failed,
        "events_without_anchor": no_anchor_events,
        "prompt_len": {"min": min(prompt_lens), "max": max(prompt_lens)} if prompt_lens else None,
        "prompt_ends_with_closed_thinking": {"true": sum(closed), "false": len(closed) - sum(closed)},
        "reasoning": summarize_reasoning(cands),
        "per_event": per_event,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--unconstrained-events", type=int, default=1,
                    help="first N events also get one unsteered vlm.generate (0 to skip)")
    ap.add_argument("--unconstrained-max-new-tokens", type=int, default=448,
                    help="room for a reasoning, the anchor, 128 trajectory tokens and the end")
    ap.add_argument("--repro-check", action="store_true",
                    help="re-run the first loaded event with the same seed and compare ids")
    ap.add_argument("--out", default="outputs/session02/tokens_edge.json")
    args = ap.parse_args(argv)

    payload, model, recorder, avdi, events = open_session(args, "probe_tokens_edge")
    out = args.out
    first_sample = None
    for n, event in enumerate(events):
        sample, entry = load_event_sample(event, avdi, args.sample_cache)
        payload["events"].append(entry)
        if sample is None:
            write_probe_json(out, payload)
            continue
        if first_sample is None:
            first_sample = (len(payload["events"]) - 1, sample)
        clean = run_condition(model, sample, recorder, args)
        entry["conditions"]["clean"] = clean
        if n < args.unconstrained_events and clean.get("status") != "error":
            try:
                entry["conditions"]["unconstrained"] = run_unconstrained(
                    model, sample, recorder, seed=args.seed,
                    max_new_tokens=args.unconstrained_max_new_tokens, first_n=args.first_n)
            except Exception as exc:  # noqa: BLE001 — optional check; block 1's answer is saved
                entry["conditions"]["unconstrained"] = {
                    "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        write_probe_json(out, payload)
        rows = clean.get("candidates") or []
        print(f"[event {event.index}] {event.nav_text!r} status={clean.get('status')} "
              f"reasoning tokens={[r['reasoning_tokens'] for r in rows]}")
        if rows:
            print(f"    first tokens: {[t['token'] for t in rows[0]['first_tokens']]}")
            print(f"    cot[0]: {rows[0]['cot'][:160]!r}")

    if args.repro_check and first_sample is not None:
        idx, sample = first_sample
        again = run_condition(model, sample, recorder, args)
        a = [c["generated_ids"] for c in payload["events"][idx]["conditions"]["clean"].get("candidates", [])]
        b = [c["generated_ids"] for c in again.get("candidates", [])]
        xa = payload["events"][idx]["conditions"]["clean"].get("pred_xyz")
        payload["reproducibility"] = {"event_index": payload["events"][idx]["index"], "seed": args.seed,
                                      "identical_generated_ids": a == b,
                                      "identical_trajectories": xa == again.get("pred_xyz")}

    payload["summary"] = summarize(payload)
    write_probe_json(out, payload)
    r = payload["summary"]["reasoning"]
    print(f"\n>>> {r.get('n_candidates', 0)} candidates; stop: {r.get('stop_counts')}; "
          f"empty reasoning: {r.get('n_empty_cot')}; reasoning tokens: {r.get('reasoning_tokens')}")
    print(f"Saved -> {out}")
    return payload


if __name__ == "__main__":
    main()
