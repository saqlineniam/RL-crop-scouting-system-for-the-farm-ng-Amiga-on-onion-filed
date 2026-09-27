"""`explain`: extract a readable decision tree from a trained model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..actions import ACTIONS, ACTION_BRANCH, BRANCHES
from ..env import AmigaMissionEnv
from ..evaluation import HDR, _mean_rows, build_cfg, fmt
from ..paths import MODELS
from ..policies import make_policy
from ..policies.rules import BASELINES
from ..policies.tree import TreePolicy, human_features, tree_to_text


def cmd_explain(args):
    from sklearn.tree import DecisionTreeClassifier
    import joblib

    cfg = build_cfg(args, args.scenario)
    env = AmigaMissionEnv(cfg)
    scale, names = human_features(env.obs_names)
    teacher = None if args.policy in BASELINES else make_policy(args.policy)
    X, y, ep_id = [], [], []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        pol = teacher or make_policy(args.policy)
        env.positioner = getattr(pol, "positioner", None)     # a learned positioner runs inside the mission
        done = False
        while not done:
            obs, m = env._obs(), env.action_masks()
            a = pol.act(env)
            if m.sum() > 1:                     # forced steps (only one legal action) need no explaining
                X.append(obs * scale); y.append(a); ep_id.append(ep)
            _, _, te, tr, _ = env.step(a)
            done = te or tr
        print(f"  collected episode {ep + 1}/{args.episodes} ({len(X)} decisions)", flush=True)
    X, y, ep_id = np.array(X), np.array(y), np.array(ep_id)
    test = ep_id == ep_id.max() if args.episodes > 1 else np.zeros(len(y), bool)
    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=max(20, len(y) // 400), random_state=0)
    tree.fit(X[~test], y[~test])
    fidelity = float((tree.predict(X[test]) == y[test]).mean()) if test.any() else float("nan")
    labels = [f"{BRANCHES[ACTION_BRANCH[c]]} -> {ACTIONS[c]}" for c in tree.classes_]
    text = tree_to_text(tree, names, labels)
    src = Path(args.policy).name
    out = Path(args.out or MODELS / f"{src}_tree")
    header = (f"Decision tree extracted from {src} (scenario {args.scenario}), depth {args.depth}\n"
              f"agrees with the original policy on {fidelity:.1%} of decisions in a held-out mission\n"
              f"decisions at blocks (QUICK/CAREFUL = the plan) and at plants (SHOOT = where to shoot from); units: ndvi values are NDVI, "
              f"*_pct are percent, image qualities 0-1, surprise in standard deviations\n\n")
    out.with_suffix(".txt").write_text(header + text)
    joblib.dump(dict(tree=tree, scale=scale, source=src), out.with_suffix(".joblib"))
    print("\n" + header + text)
    imp = sorted(zip(tree.feature_importances_, names), reverse=True)[:8]
    print("most used inputs:", ", ".join(f"{n} ({v:.0%})" for v, n in imp if v > 0))
    print(f"saved: {out.with_suffix('.txt')} and {out.with_suffix('.joblib')}")
    if args.evaluate:
        print(f"\nhow much performance does the readable tree keep? ({args.evaluate} missions)")
        print(HDR)
        env2 = AmigaMissionEnv(cfg)
        for label, pol_fn in ((src, lambda: teacher or make_policy(args.policy)),
                              (f"tree from {src}", lambda: TreePolicy(tree, scale, f"tree from {src}",
                                                        getattr(teacher or make_policy(args.policy), "positioner", None)))):
            rows = []
            for ep in range(args.evaluate):
                env2.reset(seed=args.seed + 500 + ep)
                pol, done, total = pol_fn(), False, 0.0
                env2.positioner = getattr(pol, "positioner", None)
                while not done:
                    _, r, te, tr, _ = env2.step(pol.act(env2))
                    total += r
                    done = te or tr
                s = env2.summary()
                rows.append(dict(reward=total, **s))
            print(fmt(label, _mean_rows(rows)), flush=True)
