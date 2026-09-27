"""Training by simulation-based policy iteration with common random numbers ("rollout" training).

Why: one block decision changes a mission's score by only ~0.001-0.02 points, while the measurement noise that
PPO sees when it compares two decisions is 0.1-0.5 points (measured on this simulator). So PPO needs a very large
number of missions to tell a good choice from a bad one, and its value estimate could not predict the returns.
Here the simulator itself compares the choices: at a moment of a mission, every legal choice is tried in a copy
of the mission (fork) with the SAME measurement noise (common random numbers: PEGASUS, Ng & Jordan 2000; the
"vine" of TRPO, Schulman et al. 2015), which made the comparison 25-650 times less noisy. A network learns, from
what the robot knows at that moment, how much each choice adds to ACCURACY and how many hours it costs; the new
policy takes the best one. That is one step of policy iteration with rollouts (Bertsekas 2020; Lagoudakis & Parr
2003): the improved policy is at least as good as the one it rolled out from, up to the network's error - so
starting from a hand-written rule it improves on that rule. Then it is repeated from the new policy.

Two levels, like the conditional action tree:
    positioning  at a plant: where to shoot from, or DONE - valued until the robot leaves the plant
    blocks       at a block: quick / careful x how many plants - valued over the next `horizon` blocks (the policy
                 being improved does the rest of them), plus the protocol plan for the rest of the route, so the
                 battery swaps a choice makes necessary count too
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import math
import os
import shutil
import time

import numpy as np

from .actions import A, ACTIONS, DONE, FIRST_SHOOT, N_BLOCK_ACTIONS, SHOOT_HERE, decode
from .config import Cfg, DEFAULT_SCENARIO, SCENARIOS
from .env import AmigaMissionEnv
from .evaluation import build_cfg
from .fields import field_pool, overlaps
from .policies import make_policy
from .policies.qnet import BlockChooser, ChoiceModel, RolloutPolicy, save_policy
from .policies.rules import Policy, _largest_legal, position_rule

BLOCK_LEVEL = list(range(N_BLOCK_ACTIONS))                 # QUICK_1..6, CAREFUL_1..6


def block_design():
    """The value of a block choice as a smooth function of how many plants (k) and whether careful, relative to
    QUICK_1: value = b (k-1) + c (k-1)^2 + careful x (d + e (k-1)) - so the network learns 4 numbers per situation
    (for accuracy, and 4 for hours) instead of 12 separate ones, and every valued choice informs all of them. The
    small differences between neighbouring choices (5 vs 6 plants, quick vs careful) are no longer each learned
    from their own noisy examples."""
    rows = []
    for a in BLOCK_LEVEL:
        mode, k = decode(a)
        c = 1.0 if mode == "careful" else 0.0
        rows.append([k - 1.0, (k - 1.0) ** 2, c, c * (k - 1.0)])
    return np.array(rows)
PLANT_LEVEL = list(range(FIRST_SHOOT, len(ACTIONS)))       # SHOOT_* and DONE
REF_BLOCK = A["QUICK_1"]
PLANT_BEHAVIOUR = ("minimum", "two_per_block", "all_plants", "all_plants_quick", "adaptive", "uncertainty")


# ---- valuing every choice at one moment, with common random numbers -------------------------------------------
def _mark(e):
    """What the value of a choice is measured from: (accuracy, time, zapper stops, field work)."""
    return e._accuracy(e._projected_unsampled()), e.time_s, e.zaps, e.work_s


def _gain(e, mark):
    """(accuracy gain, hours) since `mark`. The bug zapper is counted as its average share of the field work
    (zap_share), not as the 15-min stop that happens to fall inside the simulated stretch - the same expected time,
    without a 0.25-h jump in one choice's value that has nothing to do with the choice."""
    acc0, t0, z0, w0 = mark
    secs = e.time_s - t0 - e.zap_s * (e.zaps - z0) + e.zap_share * (e.work_s - w0)
    return e._accuracy(e._projected_unsampled()) - acc0, secs / 3600.0


def _plan_hours(e, averages):
    """Hours the protocol plan needs for the rest of the route from here (incl. the battery swaps it takes), with
    the robot's planning averages frozen at `averages` - so the value of a choice holds its battery effect, not the
    change it made to the robot's own estimates."""
    keep = e.avg_moves, e.avg_shots, e.avg_to_read
    e.avg_moves, e.avg_shots, e.avg_to_read = (dict(a) for a in averages)
    try:
        return e.protocol_plan()[0] / 3600.0
    finally:
        e.avg_moves, e.avg_shots, e.avg_to_read = keep


def plant_values(env, positioner, salts):
    """At a plant decision: for every legal plant action, (accuracy gain, hours) until the robot leaves this plant
    (the positioner decides any later shot at it), once per salt (a salt = one draw of the future readings, the
    same for every action - so the difference between two actions is not drowned by the reading noise). Returns
    {action: array (salts, 2)} relative to the reference (DONE if legal, else SHOOT_HERE) in the same salt."""
    m = env.action_masks()
    ref = DONE if m[DONE] else SHOOT_HERE
    i = env.here
    out = {}
    for a in (a for a in PLANT_LEVEL if m[a]):
        vals = []
        for s in salts:
            e = env.fork(s)
            mark = _mark(e)
            e._plant_step(a)
            while not e.ended and e.phase == "plant" and e.here == i:
                pm = e.action_masks()
                pa = int(positioner(e))
                e._plant_step(pa if pm[pa] else e._legal_fallback(pa, pm))
            vals.append(_gain(e, mark))
        out[a] = np.array(vals)
    return {a: v - out[ref] for a, v in out.items()}


def block_values(env, policy, horizon, salts):
    """At a block decision: for every legal block action, (accuracy gain, hours) over this block and the next
    horizon - 1 blocks (decided by `policy`, positioned by the mission's positioner), plus the hours the protocol
    plan still needs for the rest of the route (so a battery swap a choice makes necessary counts), once per salt.
    Returns {action: array (salts, 2)} relative to QUICK_1 (the protocol minimum, always legal) in the same salt."""
    m = env.action_masks()
    averages = (env.avg_moves, env.avg_shots, env.avg_to_read)
    out = {}
    for a in (a for a in BLOCK_LEVEL if m[a]):
        vals = []
        for s in salts:
            e = env.fork(s)
            mark = _mark(e)
            e.step(a)
            n = 1
            while not e.ended and n < horizon:
                e.step(policy.block_act(e))
                n += 1
            acc, hrs = _gain(e, mark)
            vals.append((acc, hrs + (0.0 if e.ended else _plan_hours(e, averages))))
        out[a] = np.array(vals)
    return {a: v - out[REF_BLOCK] for a, v in out.items()}


class Repeat(Policy):
    """A continuation for valuing: measure the next blocks the way the policy chose for the block being valued (the
    same mode and number of plants, as far as the masks allow). It has the policy's own density, and it is the same
    in every copy of the mission - so values stay clean (common random numbers) and are not valued under a future
    the policy will not have."""
    name = "repeat"

    def __init__(self, a):
        self.mode, self.k = decode(a)

    def block_act(self, env):
        return _largest_legal(env.action_masks(), self.mode, self.k)


# ---- collecting valued moments from training missions -----------------------------------------------------------
def _collect_task(args):
    """One training mission: run it, and at randomly picked decisions value every choice (see above)."""
    (cfg_dict, mix, randomize, seed, level, model_path, base, n_salts, horizon, p_sample, explore, continuation,
     per_mission) = args
    env = AmigaMissionEnv(Cfg(**cfg_dict), scenario_mix=mix, randomize=randomize)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7919)
    salts = list(range(1, n_salts + 1))
    current = RolloutPolicy(model_path) if model_path else None
    level_actions = PLANT_LEVEL if level == "plant" else BLOCK_LEVEL
    X, ACC, HRS, CHOICE = [], [], [], []

    def record(obs, vals, chosen):
        acc = np.full((len(level_actions), n_salts), np.nan)
        hrs = np.full((len(level_actions), n_salts), np.nan)
        for a, v in vals.items():
            acc[level_actions.index(a)], hrs[level_actions.index(a)] = v[:, 0], v[:, 1]
        X.append(obs); ACC.append(acc); HRS.append(hrs); CHOICE.append(level_actions.index(chosen))

    if level == "plant":
        block_pol = make_policy(PLANT_BEHAVIOUR[seed % len(PLANT_BEHAVIOUR)])   # quick and careful plants alike
        base_pos = current.positioner if current else position_rule

        def positioner(e):
            m = e.action_masks()
            choice = int(base_pos(e))
            if rng.random() < p_sample:
                vals = plant_values(e, base_pos, salts)
                record(e._obs(), vals, choice if m[choice] else e._legal_fallback(choice, m))
            if rng.random() < explore:
                return int(rng.choice(np.flatnonzero(m)))
            return choice
        env.positioner = positioner
        done = False
        while not done:
            _, _, te, tr, _ = env.step(block_pol.block_act(env))
            done = te or tr
    else:
        pol = current if current else make_policy(base)
        env.positioner = pol.positioner
        base_pol = make_policy(base)
        # the same number of valued decisions from every mission (per_mission), so small and big fields count alike
        p = min(1.0, per_mission / max(env.NBv, 1)) if per_mission else p_sample
        done = False
        while not done:
            m = env.action_masks()
            choice = int(pol.block_act(env))
            if rng.random() < p:
                # who does the next blocks while a choice is valued: the base rule; the policy measuring them the way it
                # chose for this block ("repeat": its own density, identical in every copy); or the policy itself
                cont = (pol if continuation == "current" else Repeat(choice) if continuation == "repeat" else base_pol)
                record(env._obs(), block_values(env, cont, horizon, salts), choice)
            a = int(rng.choice(np.flatnonzero(m))) if rng.random() < explore else choice
            _, _, te, tr, _ = env.step(a)
            done = te or tr
    n = len(X)
    if not n:
        return (np.zeros((0, len(env._obs())), np.float32), np.zeros((0, len(level_actions), n_salts)),
                np.zeros((0, len(level_actions), n_salts)), np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    return (np.array(X, np.float32), np.array(ACC), np.array(HRS), np.array(CHOICE), np.full(n, seed),
            np.full(n, float(env.NBv)))


def collect(cfg, mix, randomize, level, target, model_path, base, salts, horizon, p_sample, explore, workers,
            seed0, continuation="base", per_mission=0, log=print):
    """Run training missions (in parallel) until `target` valued decisions are collected."""
    parts, n, seed, t = [], 0, seed0, time.time()
    batch = max(1, workers) * 2
    while n < target:
        tasks = [(asdict(cfg), mix, randomize, seed + k, level, model_path, base, salts, horizon, p_sample, explore,
                  continuation, per_mission) for k in range(batch)]
        seed += batch
        if workers <= 1:
            res = [_collect_task(t_) for t_ in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_collect_task, tasks, chunksize=1))
        parts += res
        n = sum(len(r[0]) for r in parts)
        log(f"    {level}: {n:,} valued decisions from {seed - seed0} missions ({time.time() - t:.0f} s)", flush=True)
    return tuple(np.concatenate([r[k] for r in parts]) for k in range(6))


# ---- learning the values from what the robot knows ------------------------------------------------------------
def fit_level(X, ACC3, HRS3, CHOICE, MISSION, actions, ref, hour_value, hidden=256, epochs=80, seed=0, log=print,
              report=None, members=5, caution=0.0, design=None, blocks=None, size_weight=0.0):
    """Regress (accuracy gain, hours) of every action on the observation (masked: only the actions valued at each
    moment; target = mean over the noise draws). An ensemble of `members` networks, each fitted on a bootstrap
    resample of the training missions (bagging), all stopped early on the same held-out missions. Returns a
    ChoiceModel whose info says how good its choices are on the held-out missions."""
    import torch as th
    rng = np.random.default_rng(seed)
    ACC, HRS = np.nanmean(ACC3, 2), np.nanmean(HRS3, 2)
    per_block = blocks is not None
    if per_block:                    # accuracy in units of one block's share of the field: about size-free
        ACC = ACC * (np.asarray(blocks, np.float64)[:, None] / 1000.0)
    X = X.astype(np.float64)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    ok = np.isfinite(ACC)
    acc_scale = float(np.nanstd(ACC)) or 1.0
    hrs_scale = float(np.nanstd(HRS)) or 1.0
    Y = np.concatenate([np.nan_to_num(ACC) / acc_scale, np.nan_to_num(HRS) / hrs_scale], 1)
    W = np.concatenate([ok, ok], 1).astype(np.float64)
    if per_block and size_weight:    # per-block errors count more on small fields, where they are bigger in points
        w = (1000.0 / np.asarray(blocks, np.float64)) ** size_weight
        W = W * (w / w.mean())[:, None]
    missions = np.unique(MISSION)
    held = set(rng.choice(missions, size=max(1, len(missions) // 10), replace=False).tolist())
    te = np.array([m in held for m in MISSION])
    train_missions = np.array([m for m in missions if m not in held])
    by_mission = {m: np.flatnonzero(MISSION == m) for m in train_missions}
    tX, tY, tW = (th.tensor(a, dtype=th.float32) for a in ((X - mean) / std, Y, W))
    n_out = len(actions)
    q = n_out if design is None else design.shape[1]
    tD = None if design is None else th.tensor(np.asarray(design), dtype=th.float32)
    idx_te = np.flatnonzero(te if te.any() else ~te)
    nets = []
    for k in range(members):
        th.manual_seed(seed + 101 * k)
        boot = rng.choice(train_missions, size=len(train_missions), replace=True) if members > 1 else train_missions
        idx_tr = np.concatenate([by_mission[m] for m in boot])
        net = th.nn.Sequential(th.nn.Linear(X.shape[1], hidden), th.nn.Tanh(), th.nn.Linear(hidden, hidden),
                               th.nn.Tanh(), th.nn.Linear(hidden, 2 * q))
        opt = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

        def values(x):
            out = net(x)
            if tD is None:
                return out
            return th.cat([out[:, :q] @ tD.T, out[:, q:] @ tD.T], 1)

        def loss_on(idx):
            with th.no_grad():
                return float(((values(tX[idx]) - tY[idx]) ** 2 * tW[idx]).sum() / tW[idx].sum().clamp(min=1.0))

        best, best_state, bad = math.inf, None, 0
        for _ in range(epochs):
            rng.shuffle(idx_tr)
            for s0 in range(0, len(idx_tr), 512):
                b = idx_tr[s0:s0 + 512]
                loss = ((values(tX[b]) - tY[b]) ** 2 * tW[b]).sum() / tW[b].sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = loss_on(idx_te)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = [p_.detach().clone() for p_ in net.parameters()]
            else:
                bad += 1
                if bad >= 10:
                    break
        with th.no_grad():
            for p_, b_ in zip(net.parameters(), best_state):
                p_.copy_(b_)
        layers = [m for m in net if isinstance(m, th.nn.Linear)]
        nets.append([(l_.weight.detach().numpy().T.copy(), l_.bias.detach().numpy().copy()) for l_ in layers])
    model = ChoiceModel(actions, ref, mean, std, acc_scale, hrs_scale, nets, caution=caution, design=design,
                        per_block=per_block)
    if report is not None and (te & report).any():   # judge on held-out missions of the latest round only
        idx_te = np.flatnonzero(te & report)
    info = choice_report(model, X[idx_te], ACC3[idx_te], HRS3[idx_te], CHOICE[idx_te], actions, ref, hour_value,
                         None if blocks is None else np.asarray(blocks)[idx_te])
    info.update(samples=int(len(X)), missions=int(len(missions)), members=members)
    model.info = info
    log(f"    fit {members} networks on {len(X):,} valued decisions ({len(missions)} missions). On held-out missions, per "
        f"decision: the choice gains {info['gain_vs_policy']:+.4f} +- {info['gain_se']:.4f} points over the policy it "
        f"improves on ({info['gain_vs_ref']:+.4f} over always {ACTIONS[ref]}); choosing with the simulator itself would "
        f"gain {info['gain_oracle']:+.4f}. Noise check: two independent draws of a choice's value agree r = "
        f"{info['draw_agreement']:.2f}; R2 of the predicted values {info['r2']:.2f}", flush=True)
    return model


def choice_report(model, X, ACC3, HRS3, CHOICE, actions, ref, hour_value, blocks=None):
    """How good are a model's choices on valued decisions (unbiased: every choice at a moment is valued with the same
    simulations, and the model was not fitted on these missions). gain_vs_policy / gain_vs_ref: points per decision
    over the collecting policy's own choice / over the reference action; gain_oracle: picking with one draw of the
    simulations and scoring with the other (what choosing by simulation would gain - the ceiling for the network);
    draw_agreement: correlation of two independent draws of the same choice's value (signal vs reading noise)."""
    V3 = ACC3 - hour_value * HRS3                         # (n, actions, draws)
    V = np.nanmean(V3, 2)
    ok = np.isfinite(V)
    pa, ph = model.predict(X, blocks)
    pred = np.where(ok, model.scores(X, hour_value, blocks), -np.inf)
    rows = np.arange(len(X))
    pick = pred.argmax(1)
    ref_i = actions.index(ref)
    g = V[rows, pick] - V[rows, CHOICE]
    gain_pol, gain_se = float(np.mean(g)), float(np.std(g) / np.sqrt(max(len(g), 1)))
    gain_ref = float(np.nanmean(np.where(ok[:, ref_i], V[rows, pick] - V[:, ref_i], np.nan)))
    oracle, agree = float("nan"), float("nan")
    if V3.shape[2] >= 2:
        v0, v1 = V3[..., 0], V3[..., 1]
        p0 = np.where(ok, v0, -np.inf).argmax(1)
        p1 = np.where(ok, v1, -np.inf).argmax(1)
        oracle = float(np.mean(0.5 * (v1[rows, p0] - v1[rows, CHOICE]) + 0.5 * (v0[rows, p1] - v0[rows, CHOICE])))
        nonref = ok.copy()
        nonref[:, ref_i] = False
        if nonref.sum() > 2:
            agree = float(np.corrcoef(v0[nonref], v1[nonref])[0, 1])
    y, p = V[ok], np.where(ok, pa - hour_value * ph, 0.0)[ok]
    r2 = 1.0 - float(np.mean((y - p) ** 2)) / max(float(np.var(y)), 1e-12)
    return dict(gain_vs_policy=gain_pol, gain_se=gain_se, gain_vs_ref=gain_ref, gain_oracle=oracle,
                draw_agreement=agree, r2=r2, hour_value=hour_value, decisions=int(len(X)))


# ---- the training command -----------------------------------------------------------------------------------------
def cmd_train_rollout(args):
    import torch
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    mix = list(SCENARIOS) if args.scenario == "mixed" else None
    cfg = build_cfg(args, DEFAULT_SCENARIO if mix else args.scenario)
    pool, test = field_pool(cfg.field), field_pool(args.test_field)
    bad = sorted({a for a in pool for b in test if overlaps(a, b)})
    if bad:
        raise SystemExit(f"the training fields include the test region ({args.test_field}): {bad[:4]} - "
                         f"`validate` would no longer be a fair test")
    lam = cfg.hour_value_points
    out = Path(args.model_path)
    out = out if out.suffix == ".pt" else out.with_name(out.name + ".pt")
    ckdir = out.parent / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    name = out.stem
    print(f"training fields: {len(pool)} ({args.field}); held out for validate: {args.test_field}")
    print(f"method: policy iteration with rollouts and common random numbers, from the rule '{args.base}'; "
          f"hour value {lam:g} points; horizon {args.horizon} blocks; {args.salts} noise draws per choice")
    info = dict(fields=args.field, test_field=args.test_field, hour_value=lam, base=args.base, horizon=args.horizon,
                salts=args.salts, scenario=args.scenario, randomize=bool(args.randomize), continuation=args.continuation,
                per_mission=args.per_mission, members=args.members, caution=args.caution, value_shape=args.value_shape,
                values=args.values, blend_blocks=list(args.blend_blocks), careful_margin=args.careful_margin)
    plant_model, current = None, None
    if args.plant_states > 0:
        print(f"\n[1] positioning: valuing every place to shoot from, at {args.plant_states:,} plant decisions ...")
        X, ACC, HRS, CH, MI, NB = collect(cfg, mix, args.randomize, "plant", args.plant_states, None, args.base,
                                          args.salts, args.horizon, args.plant_share, args.explore_plant, args.workers,
                                          seed0=1000)
        model = fit_level(X, ACC, HRS, CH, MI, PLANT_LEVEL, SHOOT_HERE, lam, members=args.members,
                          blocks=NB if args.values == "per-block" else None)
        # policy improvement check: keep the learned positioning only if it beats the rule on held-out missions
        if model.info["gain_vs_policy"] > model.info["gain_se"]:
            plant_model = model
            print("    -> the learned positioning beats the hand-written one on held-out missions: using it")
        else:
            print("    -> the learned positioning does not beat the hand-written one on held-out missions: keeping the "
                  "rule (what decides the best spot is not visible to the robot)")
        current = ckdir / f"{name}_positioning.pt"
        save_policy(current, plant=plant_model, base=args.base,
                    info=dict(info, stage="positioning", positioning_fit=model.info, positioning_used=plant_model is not None))
        print(f"    saved {current}")
    data = None
    for it in range(1, args.iterations + 1):
        src = f"the rule {args.base!r}" if it == 1 else f"round {it - 1}"
        how = {"base": f"the rule {args.base!r} doing the next blocks",
               "repeat": "the next blocks measured the way the policy chose for this one",
               "current": "the policy doing the next blocks"}[args.continuation]
        print(f"\n[{it + 1 if args.plant_states > 0 else it}] blocks, round {it}: valuing every block choice at "
              f"{args.states:,} block decisions of missions run by {src} ({how}) ...")
        new = collect(cfg, mix, args.randomize, "block", args.states, str(current) if current else None, args.base,
                      args.salts, args.horizon, args.block_share, args.explore_block, args.workers, seed0=10000 * it,
                      continuation=args.continuation, per_mission=args.per_mission)
        new_round = np.full(len(new[0]), it)
        # the values keep the same meaning every round (base-rule rollouts), so earlier rounds' data is kept: the
        # network learns the states the newer policies visit without forgetting the others (DAgger-style)
        if data is None or args.continuation != "base":
            data, rounds = new, new_round
        else:
            data = tuple(np.concatenate([a, b]) for a, b in zip(data, new))
            rounds = np.concatenate([rounds, new_round])
        X, ACC, HRS, CH, MI, NB = data
        np.savez_compressed(ckdir / f"{name}_data_it{it}.npz", X=X, ACC=ACC, HRS=HRS, CHOICE=CH, MISSION=MI,
                            BLOCKS=NB, ROUND=rounds)             # to refit later without simulating again
        design = block_design() if args.value_shape == "smooth" else None
        fits = {}
        for kind in ("raw", "per-block") if args.values == "blend" else (args.values,):
            print(f"    fitting the {kind} networks ...", flush=True)
            fits[kind] = fit_level(X, ACC, HRS, CH, MI, BLOCK_LEVEL, REF_BLOCK, lam, report=rounds == it,
                                   members=args.members, caution=args.caution, design=design,
                                   blocks=NB if kind == "per-block" else None)
        block_model = BlockChooser(fits.get("raw"), fits.get("per-block"), blend=tuple(args.blend_blocks),
                                   careful_margin=args.careful_margin)
        current = ckdir / f"{name}_it{it}.pt"
        save_policy(current, block=block_model, plant=plant_model, base=args.base, info=dict(info, round=it))
        print(f"    saved {current}")
    if current is None:
        raise SystemExit("nothing trained: set --iterations and/or --plant-states above 0")
    shutil.copyfile(current, out)
    print(f"\nsaved: {out}  (the last round; every round is in {ckdir})")
    print(f"next: python -m amiga_scout curve --model-path {out.with_suffix('')}   (picks the best round on training "
          f"regions)\n      python -m amiga_scout validate --models <best round>   (the test regions)")
