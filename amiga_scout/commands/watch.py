"""`watch`: one mission, pass by pass, with text maps of what was measured and found."""
from __future__ import annotations

from ..actions import ACTION_BRANCH, BRANCHES
from ..env import AmigaMissionEnv
from ..evaluation import build_cfg
from ..policies import make_policy


def cmd_watch(args):
    cfg = build_cfg(args, args.scenario)
    env = AmigaMissionEnv(cfg)
    env.reset(seed=args.seed)
    print(f"field {env.cfg.field} ({env.cfg.field_acres:.1f} acres), field type '{args.scenario}': {env.R} passes, "
          f"{env.NBv} blocks, {env.n_stressed} stressed plants")
    pol = make_policy(args.policy)
    env.positioner = getattr(pol, "positioner", None)
    done, total = False, 0.0
    counts = {n: 0 for n in BRANCHES}
    cur, n_p = env.row, env.n_scanned
    while not done:
        a = pol.act(env)
        counts[BRANCHES[ACTION_BRANCH[a]]] += 1
        _, r, te, tr, info = env.step(a)
        total += r
        done = te or tr
        if info.get("event") == "swap":
            print(f"  [battery swap #{env.swaps}: drove straight on out of pass {env.row}, swapped, came back "
                  f"to the same plant; t={env.time_s / 3600:.2f} h]")
        if env.row != cur or done:
            nb = (int(env.c1[cur]) - int(env.c0[cur]) + 1) // env.K
            print(f"pass {cur:2d}: {env.n_scanned - n_p:4d} plants measured in {nb:3d} blocks  t={env.time_s / 3600:5.2f} h  "
                  f"batt={env.batt_wh / env.capacity_wh:4.0%}  reward so far {total:7.1f}")
            cur, n_p = env.row, env.n_scanned
    sm = env.summary()
    print("\ndecisions:", {k: v for k, v in counts.items() if v},
          f"| plants measured: {env.quick_plants} quick, {env.careful_plants} careful"
          f"\npositioning: shot from {sm['mean_shot_shift_cm']:.0f} cm from where it stopped on average; "
          f"{sm['nudges_per_plant']:.1f} moves of 0.1 m per plant; a second shot on {sm['second_shot_share']:.0%} of plants")
    counts_map, status_map = env.block_maps()
    print(f"\nplants measured per 10-ft block (blank = none):\n{counts_map}")
    print(f"\nstress per 10-ft block:\n{status_map}")
    print("legend: '.' no stress   '#' stress found   '!' stress missed   '?' false alarm")
    print("SUMMARY", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in env.summary().items()})
