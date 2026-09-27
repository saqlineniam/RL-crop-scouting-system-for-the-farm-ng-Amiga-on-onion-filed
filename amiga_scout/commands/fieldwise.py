"""`fieldwise`: the score field by field - every flight date of the test field, every field type - with what the
score is made of (the four accuracy parts, hours) and what each strategy did (plants, care, shots, moves).

Every strategy runs the same missions (same map, route and stress), so the differences are paired.
All missions are also saved to a CSV for a closer look.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..config import FLIGHT_DATES
from ..evaluation import build_cfg, mean_ci, run_parallel
from ..fields import field_pool
from ..paths import MODELS

DETAIL = [("score", "SCORE", "{:6.1f}"), ("accuracy", "accur", "{:6.1f}"), ("hours", "hours", "{:6.1f}"),
          ("acc_blocks", "blocks", "{:6.1f}"), ("acc_stress", "stress", "{:6.1f}"), ("acc_parts", "parts", "{:6.1f}"),
          ("acc_protocol", "protoc", "{:6.1f}"), ("plants_per_block", "pl/blk", "{:6.2f}"),
          ("careful_share", "carefl", "{:6.0%}"), ("shots_per_plant", "shots", "{:6.2f}"),
          ("nudges_per_plant", "moves", "{:6.2f}"), ("recall", "recall", "{:6.0%}"), ("false_alarms", "f.alrm", "{:6.0f}"),
          ("swaps", "swaps", "{:6.1f}")]


def _label(field):
    """'f2_03' -> 'Mar 7', 'f2_03@10' -> 'Mar 7 @10ac'."""
    base, _, piece = field.partition("@")
    date = FLIGHT_DATES.get(base.split("_")[-1], base)
    return f"{date} @{piece}ac" if piece else date


def cmd_fieldwise(args):
    names = list(args.rules or []) + list(args.models or [])
    if not names:
        raise SystemExit("give --models and/or --rules")
    maps = field_pool(args.field)
    scens = args.scenarios
    seeds = list(range(args.seed, args.seed + args.missions))
    jobs = [((m, s, n), build_cfg(args, s, m), n, seeds) for m in maps for s in scens for n in names]
    print(f"{len(maps)} fields x {len(scens)} field types x {len(seeds)} missions x {len(names)} strategies = "
          f"{len(jobs) * len(seeds)} missions on {args.workers} processes ...", flush=True)
    res = run_parallel(jobs, args.workers)
    label = {n: res[(maps[0], scens[0], n)][1] for n in names}
    for rows, _ in res.values():
        for r in rows:
            r["careful_share"] = r["plants_careful"] / max(r["plants_measured"], 1)

    def rows(m=None, s=None, n=None):
        return [r for (mm, ss, nn), (rr, _) in res.items() if (m is None or mm == m) and (s is None or ss == s)
                and nn == n for r in rr]

    def mean(m, s, n, key="score"):
        return float(np.mean([r[key] for r in rows(m, s, n)]))

    short = {n: label[n][:14] for n in names}
    rules = list(args.rules or [])
    lam = build_cfg(args, scens[0], maps[0]).hour_value_points

    brief = {"minimum": "minimum", "two_per_block": "2/blk careful", "all_plants": "all careful",
             "all_plants_quick": "all quick", "adaptive": "adaptive", "uncertainty": "uncertainty",
             "battery_fill": "battery fill"}
    nick = {n: brief.get(n, "RL " + Path(n).name[:12]) for n in names}
    print(f"\n1) FIELD BY FIELD: each cell = SCORE = accuracy - {lam:g} x hours (mean of {len(seeds)} missions)")
    for m in maps:
        r0 = rows(m, None, names[0])
        print(f"\n  {_label(m)} ({m}): {np.mean([r['acres'] for r in r0]):.1f} acres, "
              f"{np.mean([r['blocks'] for r in r0]):,.0f} ten-ft blocks")
        print(f"  {'type':8s} {'winner (lead)':>16s}" + "".join(f" {nick[n]:>21s}" for n in names)
              + ("   what the RL did" if args.models else ""))
        for sc in scens:
            win = max(names, key=lambda n: mean(m, sc, n))
            runner = max((n for n in names if n != win), key=lambda n: mean(m, sc, n), default=win)
            line = f"  {sc:8s} {nick[win][:11]:>11s} +{mean(m, sc, win) - mean(m, sc, runner):3.1f}"
            line += "".join(f" {mean(m, sc, n):5.1f} ={mean(m, sc, n, 'accuracy'):5.1f} -{mean(m, sc, n, 'hours'):5.1f}h"
                            for n in names)
            for mdl in args.models or []:
                line += (f"   {mean(m, sc, mdl, 'plants_per_block'):.1f} plants/blk, "
                         f"{mean(m, sc, mdl, 'careful_share'):.0%} careful, "
                         f"{mean(m, sc, mdl, 'nudges_per_plant'):.1f} moves/plant, "
                         f"stress found {mean(m, sc, mdl, 'recall'):.0%}")
            print(line)
    wins = {n: 0 for n in names}
    for m in maps:
        for sc in scens:
            wins[max(names, key=lambda n: mean(m, sc, n))] += 1
    print(f"\n  wins over {len(maps) * len(scens)} field x type cases: "
          + ", ".join(f"{nick[n]} {w}" for n, w in wins.items()))

    print(f"\n2) SCORE = accuracy - {lam:g} x hours, per field and field type ({len(seeds)} missions each, "
          f"same missions for every strategy)")
    for s in scens + ["ALL TYPES"]:
        ss = None if s == "ALL TYPES" else s
        print(f"\n  {s}")
        head = f"  {'field':14s}" + "".join(f" {short[n]:>14s}" for n in names)
        if args.models and rules:
            head += "   model - best rule"
        print(head)
        for m in maps + ["ALL"]:
            mm = None if m == "ALL" else m
            line = f"  {('all fields' if m == 'ALL' else _label(m)):14s}" + "".join(
                f" {mean(mm, ss, n):14.1f}" for n in names)
            if args.models and rules:
                best = max(rules, key=lambda b: mean(mm, ss, b))
                for mdl in args.models:
                    d = [a["score"] - b["score"] for a, b in zip(rows(mm, ss, mdl), rows(mm, ss, best))]
                    mu, h = mean_ci(d)
                    line += f"   {mu:+5.1f} +- {h:3.1f} vs {label[best][:24]}"
            print(line)

    print("\n3) WHAT THE SCORE IS MADE OF, per field (all field types together)")
    print("   accuracy = 25% blocks (10-ft NDVI + height right) + 30% stress (stressed plants found) + 25% parts "
          "+ 20% protocol (blocks sampled); each part 0-100")
    for m in maps + ["ALL"]:
        mm = None if m == "ALL" else m
        print(f"\n  {('all fields' if m == 'ALL' else _label(m) + ' (' + m + ')')}")
        print(f"  {'strategy':30s}" + "".join(f" {h:>6s}" for _, h, _ in DETAIL))
        for n in names:
            rr = rows(mm, None, n)
            print(f"  {label[n][:30]:30s}" + "".join(
                " " + fmt.format(float(np.mean([r[k] for r in rr]))) for k, _, fmt in DETAIL))

    out = Path(args.csv) if args.csv else MODELS / "fieldwise.csv"
    allrows = [dict(strategy=label[n], field_type=s, **r) for (m, s, n), (rr, _) in res.items() for r in rr]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(f"\nsaved every mission: {out}")
