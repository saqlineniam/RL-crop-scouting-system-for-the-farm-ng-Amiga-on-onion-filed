"""`curve`: learning curve over the saved checkpoints, on selection fields (not the test field)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DEFAULT_SCENARIO
from ..evaluation import build_cfg, run_parallel


def group_label(spec):
    """A short name for a selection field group: 'whole regions', 'pieces 5-10 ac', 'big fields ... 30-35 ac'."""
    toks = spec.split()
    sizes = sorted({float(t.partition("@")[2]) for t in toks if "@" in t})
    size = f" {sizes[0]:g}-{sizes[-1]:g} ac" if len(sizes) > 1 else (f" {sizes[0]:g} ac" if sizes else "")
    if all("*" in t for t in toks):
        return "big fields of real drone data (tiled)" + size
    if all("@" in t for t in toks):
        return "pieces" + size
    if all(t.startswith("sq") for t in toks):
        return "synthetic squares"
    return "whole regions"


def cmd_curve(args):
    """Learning curve from the saved checkpoints: every checkpoint runs on the same SELECTION fields (seeds
    separate from `validate`'s test fields, so picking the best checkpoint here does not flatter the final
    test), with the hand-written rules as reference lines. Writes a PNG and a CSV next to the model."""
    mp = Path(args.model_path)
    mp = mp.with_suffix("") if mp.suffix in (".pt", ".zip") else mp
    name, ckdir = mp.name, mp.parent / "checkpoints"
    args.model_path = str(mp)
    rounds = sorted((int(p.stem.rsplit("_it", 1)[1]), str(p)) for p in ckdir.glob(f"{name}_it*.pt")
                    if p.stem.rsplit("_it", 1)[1].isdigit())
    if rounds:                                   # rollout training: one learned policy per round (0 = positioning only)
        unit = "round"
        pos = ckdir / f"{name}_positioning.pt"
        ckpts = ([(0, str(pos))] if pos.exists() else []) + rounds
    else:                                        # PPO: checkpoints every 250k steps
        unit = "steps"
        ckpts = sorted((int(p.stem[len(name) + 1:-len("_steps")]), str(p.with_suffix("")))
                       for p in ckdir.glob(f"{name}_*_steps.zip") if p.stem[len(name) + 1:-len("_steps")].isdigit())
        ckpts = ckpts[args.every - 1::args.every]
        if Path(args.model_path + ".zip").exists() and (not ckpts or ckpts[-1][0] < args.final_steps):
            ckpts.append((args.final_steps, args.model_path))
    if not ckpts:
        raise SystemExit(f"no rounds or checkpoints found for {name} in {ckdir}")
    seeds = list(range(args.seed, args.seed + args.fields))
    conds = [(spec, spec) for spec in args.select_fields]
    rules = ["minimum", "two_per_block", "all_plants_quick", "adaptive", "uncertainty"]
    jobs = [((st, key), build_cfg(args, DEFAULT_SCENARIO, spec), path, seeds) for st, path in ckpts for key, spec in conds]
    jobs += [((r, key), build_cfg(args, DEFAULT_SCENARIO, spec), r, seeds) for r in rules for key, spec in conds]
    print(f"{len(ckpts)} checkpoints x {len(conds)} selection field sets x {len(seeds)} missions "
          f"(+ {len(rules)} rules): {sum(len(j[3]) for j in jobs)} missions on {args.workers} processes ...", flush=True)
    res = run_parallel(jobs, args.workers)
    mean = lambda key: float(np.mean([r["score"] for r in res[key][0]]))  # noqa: E731
    steps = [st for st, _ in ckpts]
    per = {key: [mean((st, key)) for st in steps] for key, _ in conds}
    overall = [float(np.mean([per[key][i] for key, _ in conds])) for i in range(len(steps))]
    rule_all = {r: float(np.mean([mean((r, key)) for key, _ in conds])) for r in rules}
    labels = {r: res[(r, conds[0][0])][1] for r in rules}
    best_i = int(np.argmax(overall))
    csv = Path(args.model_path + "_curve.csv")
    with open(csv, "w") as f:
        f.write(f"{unit},overall," + ",".join(k for k, _ in conds) + "\n")
        for i, st in enumerate(steps):
            f.write(f"{st},{overall[i]:.3f}," + ",".join(f"{per[k][i]:.3f}" for k, _ in conds) + "\n")
    print("\nselection field groups (each: new routes and stress, never the test regions):")
    for g, (k, _) in enumerate(conds, 1):
        print(f"  G{g} = {group_label(k)}:  {k}")
    print(f"\n{unit:>12s} {'overall':>8s}" + "".join(f" {'G' + str(g):>9s}" for g in range(1, len(conds) + 1)))
    for i, st in enumerate(steps):
        print(f"{st:12,d} {overall[i]:8.2f}" + "".join(f" {per[k][i]:9.2f}" for k, _ in conds) + ("   <- best" if i == best_i else "")
              + ("   (learned positioning, the base rule's block choices)" if unit == "round" and st == 0 else ""))
    print("\nrules on the same fields:")
    for r in rules:
        print(f"  {labels[r][:40]:40s} {rule_all[r]:8.2f}" + "".join(f" {mean((r, k)):9.2f}" for k, _ in conds))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.array(steps) / (1.0 if unit == "round" else 1e6)
    ax[0].plot(x, overall, "k-o", ms=3, label="RL model (average of the selection fields)")
    ax[0].axvline(x[best_i], color="g", ls=":", label=f"best: {unit} {x[best_i]:g}" + ("M" if unit == "steps" else ""))
    for r, col in zip(rules, ["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:blue"]):
        ax[0].axhline(rule_all[r], color=col, ls="--", lw=1.2, label=labels[r])
    xl = "training round" if unit == "round" else "training steps (millions)"
    lam = build_cfg(args, DEFAULT_SCENARIO, conds[0][1]).hour_value_points
    ax[0].set(title="Score on selection fields vs training", xlabel=xl, ylabel=f"score = accuracy - {lam:g} x hours")
    ax[0].legend(fontsize=7)
    for key, _ in conds:
        ax[1].plot(x, per[key], "-o", ms=3, label=f"RL, {key}")
    ax[1].set(title="By selection field", xlabel=xl, ylabel="score")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = Path(args.model_path + "_curve.png")
    fig.savefig(png, dpi=130)
    print(f"\nbest on the selection fields: {unit} {steps[best_i]:,} ({ckpts[best_i][1]})")
    print(f"saved: {png} and {csv}")
    sets = "".join(f" --set {x}" for x in (args.set or []))
    print("next, on the test regions (never used to choose it) - copy these lines as they are:")
    print(f"  python -m amiga_scout validate --models {ckpts[best_i][1]}{sets}")
    print(f"  python -m amiga_scout fieldwise --models {ckpts[best_i][1]}{sets}")
