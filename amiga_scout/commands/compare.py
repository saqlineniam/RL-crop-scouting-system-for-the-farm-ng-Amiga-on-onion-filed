"""`compare`: strategies side by side on the same missions."""
from __future__ import annotations

import numpy as np

from ..config import SCORE_W_BLOCKS, SCORE_W_DETECTION, SCORE_W_PARTS, SCORE_W_PROTOCOL
from ..evaluation import HDR, _mean_rows, build_cfg, fmt, run_parallel
from ..policies.rules import BASELINES


def cmd_compare(args):
    names = (args.policies if args.policies is not None else list(BASELINES)) + list(args.models or [])
    seeds = range(args.seed, args.seed + args.episodes)
    jobs = [((scen, n), build_cfg(args, scen), n, seeds) for scen in args.scenarios for n in names]
    res = run_parallel(jobs, args.workers)
    scores = {}
    for scen in args.scenarios:
        if not args.brief:
            print(f"\n=== {scen}: {args.episodes} fields")
            print(HDR)
            print("-" * len(HDR))
        for n in names:
            rows, label = res[(scen, n)]
            r = _mean_rows(rows)
            scores.setdefault(label, {})[scen] = r["score"]
            if not args.brief:
                print(fmt(label, r))
    lam = build_cfg(args).hour_value_points
    print(f"\n=== OVERALL SCORE = accuracy (0-100) - {lam:g} x hours   (accuracy: blocks {SCORE_W_BLOCKS:.0%}, stress "
          f"found {SCORE_W_DETECTION:.0%}, parts {SCORE_W_PARTS:.0%}, every block sampled {SCORE_W_PROTOCOL:.0%})")
    hdr = f"{'rank':4s} {'strategy':38s}" + "".join(f" {s[:13]:>13s}" for s in args.scenarios) + f" {'OVERALL':>8s}"
    print(hdr)
    print("-" * len(hdr))
    ranked = sorted(scores.items(), key=lambda kv: -np.mean(list(kv[1].values())))
    for k, (label, per) in enumerate(ranked, 1):
        print(f"{k:<4d} {label[:38]:38s}" + "".join(f" {per[s]:13.1f}" for s in args.scenarios)
              + f" {np.mean(list(per.values())):8.1f}")
