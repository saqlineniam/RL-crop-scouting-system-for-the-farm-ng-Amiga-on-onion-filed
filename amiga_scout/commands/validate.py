"""`validate`: is a trained model good enough for the field? (the held-out test regions of both fields, paired
with the hand-written rules, READY / NOT READY)"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import DEFAULT_SCENARIO, PERTURBATIONS, SCENARIOS
from ..evaluation import build_cfg, mean_ci, run_parallel
from ..fields import field_pool, overlaps
from ..policies.rules import BASELINES


def cmd_validate(args):
    """Is the trained model good enough to take to the field? See the module docstring."""
    rules = [n for n in BASELINES if n != "random"]
    models = list(args.models or [])
    names = rules + models
    seeds = list(range(args.seed, args.seed + args.fields))            # missions never used in training
    stress_seeds = seeds[:args.stress_fields]
    test = args.test_field
    base = build_cfg(args, DEFAULT_SCENARIO, test)
    groups = {}                                                        # group label -> list of keys
    jobs = []
    for scen in SCENARIOS:                                             # the held-out field, every field type
        groups.setdefault("field type", []).append(scen)
        jobs += [((scen, n), build_cfg(args, scen, test), n, seeds) for n in names]
    train = field_pool(args.train_field)

    def seen(spec):
        return any(overlaps(a, b) for a in field_pool(spec) for b in train)
    others = ([(f"{test} pieces of {a:g} acres (held out)", " ".join(f"{t}@{a:g}" for t in test.split()))
               for a in args.sizes]
              + [(f"{spec} ({'partly seen in training: new routes + new stress' if seen(spec) else 'never trained on'})",
                  spec) for spec in args.other_fields])
    for key, spec in others:
        groups.setdefault("other field", []).append(key)
        jobs += [((key, n), build_cfg(args, DEFAULT_SCENARIO, spec), n, stress_seeds) for n in names]
    for label, fn in PERTURBATIONS.items():
        groups.setdefault("stress test", []).append(label)
        jobs += [((label, n), replace(base, **fn(base)), n, stress_seeds) for n in names]
    print(f"test field: {test} ({', '.join(field_pool(test))}), never used in training or checkpoint selection")
    print(f"running {sum(len(j[3]) for j in jobs)} missions on {args.workers} processes ...", flush=True)
    res = run_parallel(jobs, args.workers)
    label_of = {n: res[(DEFAULT_SCENARIO, n)][1] for n in names}
    failures, notes = [], []

    def scores(key, n, what="score"):
        return [r[what] for r in res[(key, n)][0]]

    lam = base.hour_value_points
    print(f"\n1) SCORE = accuracy - {lam:g} x hours, on the held-out field {test} ({args.fields} missions per field "
          f"type: its flight dates x different routes and stress; mean +- 95% CI)")
    print(f"{'strategy':44s}" + "".join(f" {s:>17s}" for s in SCENARIOS))
    for n in names:
        print(f"{label_of[n][:44]:44s}" + "".join(" {:9.1f} +- {:4.1f}".format(*mean_ci(scores(s, n))) for s in SCENARIOS))
    print("\n   TIME vs PRECISION: accuracy (0-100) / hours, mean over the same fields")
    print(f"{'strategy':44s}" + "".join(f" {s:>17s}" for s in SCENARIOS))
    for n in names:
        print(f"{label_of[n][:44]:44s}" + "".join(
            f" {np.mean(scores(s, n, 'accuracy')):9.1f} / {np.mean(scores(s, n, 'hours')):4.1f}h" for s in SCENARIOS))

    print("\n2) SAFETY AND PROTOCOL (every mission above and below)")
    for n in names:
        allrows = [r for key, (rows, _) in res.items() if key[1] == n for r in rows]
        stranded = int(sum(r["stranded"] for r in allrows))
        illegal = int(sum(r["illegal"] for r in allrows))
        normal = [r for key, (rows, _) in res.items() if key[1] == n and key[0] in SCENARIOS for r in rows]
        gaps = sum(r["blocks_sampled"] < 1.0 for r in normal)
        gaps_other = sum(r["blocks_sampled"] < 1.0 for r in allrows) - gaps
        print(f"  {label_of[n][:42]:42s} stranded {stranded}  illegal actions {illegal}  "
              f"missions with an unsampled block: {gaps} normal, {gaps_other} other")
        if n in models:
            if stranded or illegal:
                failures.append(f"{label_of[n]}: safety rule broken ({stranded} stranded, {illegal} illegal)")
            if gaps:
                failures.append(f"{label_of[n]}: left 10-ft blocks unsampled in {gaps} normal missions (protocol)")
            if gaps_other:
                notes.append(f"{label_of[n]}: in {gaps_other} stress-test/size missions the batteries ran out "
                             f"before every block was sampled - compare with the rules in the table below")

    if models:
        print("\n3) WHERE IS THE MODEL STRONGER OR WEAKER? model minus the best hand-written rule on the same "
              "fields (paired; + = model better)")
        for n in models:
            for group, keys in groups.items():
                for key in keys:
                    best = max(rules, key=lambda b: np.mean(scores(key, b)))
                    d = [a - b for a, b in zip(scores(key, n), scores(key, best))]
                    mu, h = mean_ci(d)
                    wins = float(np.mean([x > 0 for x in d]))
                    d_acc = np.mean(scores(key, n, "accuracy")) - np.mean(scores(key, best, "accuracy"))
                    d_hrs = np.mean(scores(key, n, "hours")) - np.mean(scores(key, best, "hours"))
                    if len(d) < 3:                     # too few fields for a confidence interval
                        verdict = "too few fields to tell"
                    elif mu - h > 0:
                        verdict = "better"             # confidently above the rule
                    elif mu + h < 0 and mu < -1.0:
                        verdict = "WORSE"              # confidently below, by more than 1 point
                    elif mu - h > -1.0:
                        verdict = "not worse"          # at most 1 point below, with confidence
                    else:
                        verdict = "inconclusive"       # could go either way - run more fields
                    print(f"  {label_of[n][:22]:22s} {group:11s} {key[:34]:34s} vs {label_of[best][:26]:26s} "
                          f"score {mu:+5.1f} +- {h:3.1f} (accuracy {d_acc:+5.1f}, time {d_hrs:+5.2f} h)  "
                          f"wins {wins:4.0%} -> {verdict}")
                    if verdict == "inconclusive":
                        notes.append(f"{label_of[n]}: '{key}' too close to call ({mu:+.1f} +- {h:.1f}) - use more fields")
                    if verdict == "WORSE":
                        failures.append(f"{label_of[n]}: worse than '{label_of[best]}' on '{key}' ({mu:+.1f} +- {h:.1f})")

    print("\n4) MEAN SCORE PER CONDITION (all strategies)")
    print(f"{'condition':48s}" + "".join(f" {label_of[n][:16]:>16s}" for n in names))
    for group in ("other field", "stress test"):
        for key in groups[group]:
            print(f"{key[:48]:48s}" + "".join(f" {np.mean(scores(key, n)):16.1f}" for n in names))
    print("\n   TIME vs PRECISION: accuracy (0-100) / hours, mean over the same fields")
    print(f"{'condition':48s}" + "".join(f" {label_of[n][:16]:>16s}" for n in names))
    for group in ("other field", "stress test"):
        for key in groups[group]:
            print(f"{key[:48]:48s}" + "".join(
                f" {np.mean(scores(key, n, 'accuracy')):8.1f} /{np.mean(scores(key, n, 'hours')):5.1f}h" for n in names))

    print("\n5) VERDICT")
    if not models:
        print("  (no --models given: only the hand-written rules were checked)")
    elif failures:
        print("  NOT READY for the field:")
        for f in failures:
            print(f"   - {f}")
        print("  -> use the best hand-written rule in the field, or fix the above and retrain")
    else:
        print("  READY for a supervised field trial (shadow mode first): safe, follows the protocol, and not worse")
        print("  than the best hand-written rule on any field type, other field or stress test.")
    for nte in notes:
        print(f"  note: {nte}")
