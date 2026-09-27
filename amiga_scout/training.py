"""Training. The default method is simulation-based policy iteration with rollouts (amiga_scout.rollout); this
module holds the earlier method, kept for comparison: MaskablePPO with the hierarchical policy (HAM-PPO),
optionally warm-started by copying a hand-written rule (behaviour cloning) first."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import os

import numpy as np

from .actions import ACTION_BRANCH
from .config import Cfg, DEFAULT_SCENARIO, SCENARIOS
from .env import AmigaMissionEnv
from .evaluation import build_cfg
from .fields import field_pool, overlaps
from .paths import MODELS
from .policies import make_policy


# ---- warm start: copy a hand-written rule first, then let RL improve on it -------------------
def _demo_task(args):
    """Run the teacher rule on one training-style field; return what it saw and did."""
    cfg_dict, mix, randomize, teacher, seed, gamma = args
    env = AmigaMissionEnv(Cfg(**cfg_dict), scenario_mix=mix, randomize=randomize)
    env.reset(seed=seed)
    pol = make_policy(teacher)
    obs, acts, masks, rews = [], [], [], []
    done = False
    while not done:
        m = env.action_masks()
        o = env._obs()
        a = pol.act(env)
        _, r, te, tr, _ = env.step(a)
        obs.append(o); acts.append(a); masks.append(m); rews.append(r)
        done = te or tr
    ret, g = np.zeros(len(rews), dtype=np.float32), 0.0       # discounted return from each step, for the critic
    for k in range(len(rews) - 1, -1, -1):
        g = rews[k] + gamma * g
        ret[k] = g
    return np.array(obs, np.float32), np.array(acts), np.array(masks), ret


def warm_start(model, cfg, mix, randomize, teacher, episodes, workers, gamma, epochs=25, batch=1024):
    """Behaviour cloning: the policy learns to copy `teacher` (and the critic learns how good each
    situation is under it), so RL starts from the best hand-written rule instead of from scratch."""
    import torch as th
    tasks = [(asdict(cfg), mix, randomize, teacher, 50000 + k, gamma) for k in range(episodes)]
    print(f"warm start: recording the '{teacher}' rule on {episodes} training fields ...", flush=True)
    if workers <= 1:
        demos = [_demo_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            demos = list(ex.map(_demo_task, tasks, chunksize=1))
    X = np.concatenate([d[0] for d in demos]); Y = np.concatenate([d[1] for d in demos])
    M = np.concatenate([d[2] for d in demos]); G = np.concatenate([d[3] for d in demos])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_test = max(1, len(X) // 20)
    test, train = idx[:n_test], idx[n_test:]
    pol = model.policy
    pol.set_training_mode(True)
    with th.no_grad():                              # start the critic at the average return; it learns the rest
        pol.value_net.bias.fill_(float(G.mean()))
    opt = th.optim.Adam(pol.parameters(), lr=1e-3)
    for ep in range(epochs):
        rng.shuffle(train)
        tot = vtot = 0.0
        for s in range(0, len(train), batch):
            b = train[s:s + batch]
            values, logp, _ = pol.evaluate_actions(th.as_tensor(X[b]), th.as_tensor(Y[b]), action_masks=M[b])
            copy_loss = -logp.mean()
            value_loss = th.nn.functional.mse_loss(values.flatten(), th.as_tensor(G[b])) / max(float(G.var()), 1.0)
            loss = copy_loss + 0.5 * value_loss
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()
            tot += float(copy_loss) * len(b)
            vtot += float(value_loss) * len(b)
        with th.no_grad():
            dist = pol.get_distribution(th.as_tensor(X[test]), action_masks=M[test])
            agree = float((dist.get_actions(deterministic=True).numpy() == Y[test]).mean())
        print(f"  copy epoch {ep + 1}/{epochs}: copy loss {tot / len(train):.3f}, critic error {vtot / len(train):.2f} (1 = no better than the average), agrees with the rule on {agree:.1%} "
              f"of held-out decisions", flush=True)
    pol.set_training_mode(False)


def cmd_train(args):
    if args.method == "rollout":
        from .rollout import cmd_train_rollout
        args.model_path = args.model_path or str(MODELS / "onion_rl")
        return cmd_train_rollout(args)
    args.model_path = args.model_path or str(MODELS / "onion_ham")
    return cmd_train_ppo(args)


def cmd_train_ppo(args):
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from .policies.ham import HierarchicalMaskablePolicy

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 4)))   # the network update; envs use 1 each
    mix = list(SCENARIOS) if args.scenario == "mixed" else None
    cfg = build_cfg(args, DEFAULT_SCENARIO if mix else args.scenario)
    randomize = args.randomize
    if any(overlaps(a, b) for a in field_pool(cfg.field) for b in field_pool(args.test_field)):
        raise SystemExit(f"the training fields ({cfg.field}) include the test field ({args.test_field}) - "
                         f"`validate` would no longer be a fair test")
    print(f"training fields: {', '.join(field_pool(cfg.field))}  (held out for validate: {args.test_field})")

    def make_env():
        from stable_baselines3.common.monitor import Monitor      # logs ep_rew_mean (= mission score gain)
        return ActionMasker(Monitor(AmigaMissionEnv(cfg, scenario_mix=mix, randomize=randomize)),
                            lambda e: e.unwrapped.action_masks())

    venv = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    net = dict(net_arch=[256, 256])
    if args.arch == "ham":
        policy, kwargs = HierarchicalMaskablePolicy, dict(action_groups=ACTION_BRANCH, **net)
    else:
        policy, kwargs = "MlpPolicy", net
    ckpt_dir = Path(args.model_path).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # one decision per 10-ft block: ~200 (5-acre square) to ~1,900 (Field 1) decisions per mission. Settings follow HAM-PPO
    # (Khosravi et al. 2025: lr 3e-4, 10 epochs, GAE 0.95, clip 0.2, entropy 0.02) with a longer horizon for the
    # longer missions; after a warm start, learn more gently so RL refines the copied rule instead of erasing it
    gamma = 0.995
    lr, ent = (1e-4, 0.005) if args.warm_start else (3e-4, args.ent_coef)
    lr_end = args.lr_end
    # the learning rate shrinks linearly to lr_end x its start (progress goes 1 -> 0): earlier runs peaked at ~1M
    # steps and then drifted with a constant rate
    schedule = (lambda progress: lr * (lr_end + (1.0 - lr_end) * progress)) if lr_end < 1.0 else lr
    model = MaskablePPO(policy, venv, device="cpu", verbose=1, seed=args.seed, n_steps=1024, batch_size=512,
                        n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=ent, gamma=gamma,
                        learning_rate=schedule, policy_kwargs=kwargs)
    if args.warm_start:
        warm_start(model, cfg, mix, randomize, args.warm_start, args.demo_episodes, args.workers, gamma)
        model.save(str(Path(args.model_path)) + "_warmstart")
    model.learn(total_timesteps=args.steps, callback=CheckpointCallback(
        save_freq=max(1, args.checkpoint_steps // args.n_envs), save_path=str(ckpt_dir),
        name_prefix=Path(args.model_path).name))
    model.save(args.model_path)
    venv.close()
    print(f"\nsaved: {args.model_path}.zip")
