"""Running missions (in parallel), building settings, and the result tables' helpers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
import json
import math

import numpy as np

from .config import Cfg, DEFAULT_SCENARIO, SCENARIOS
from .env import AmigaMissionEnv
from .fields import field_pool
from .policies import make_policy
from .policies.rules import BASELINES


# =====================================================================================
#  Running, scoring, validation
# =====================================================================================
def parse_sets(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_cfg(args, scenario=DEFAULT_SCENARIO, field=None):
    """Settings for one field type (scenario) on one field / pool of fields (default: --field)."""
    ov = dict(SCENARIOS[scenario])
    field = field or getattr(args, "field", None)
    if field:
        field_pool(field)                                   # fail early if a map is missing
        ov["field"] = field
    ov.update(parse_sets(getattr(args, "set", None)))
    valid = {f.name for f in fields(Cfg)}
    bad = [k for k in ov if k not in valid]
    if bad:
        raise SystemExit(f"Unknown parameter(s): {bad}. Run `describe` to list them.")
    return replace(Cfg(), **ov)


def overall_score(s):
    """SCORE of one mission = ACCURACY - HOUR_VALUE_POINTS x hours (0 if stranded); see the env summary."""
    return s["score"]


def run_missions(cfg, policy_name, seeds):
    """Run one strategy on these fields; per-mission results incl. the safety/protocol checks."""
    env = AmigaMissionEnv(cfg)
    rows, pol = [], None
    for sd in seeds:
        env.reset(seed=sd)
        pol = make_policy(policy_name) if (pol is None or policy_name in BASELINES) else pol
        env.positioner = getattr(pol, "positioner", None)    # its positioning runs inside the mission
        done, total, illegal = False, 0.0, 0
        while not done:
            m = env.action_masks()
            a = pol.act(env)
            illegal += int(not m[a])
            _, r, te, tr, _ = env.step(a)
            total += r
            done = te or tr
        s = env.summary()
        rows.append(dict(reward=total, illegal=illegal, **s))
    return rows, pol.name


def _task(args):
    cfg_dict, policy_name, seeds = args
    return run_missions(Cfg(**cfg_dict), policy_name, seeds)


def run_parallel(jobs, workers):
    """jobs: list of (key, cfg, policy_name, seeds) -> {key: (rows, name)}; spread over processes.
    Jobs are split per field so that slow strategies do not hold up the rest."""
    flat = [(k, j, s) for k, (_, cfg, pol, seeds) in enumerate(jobs) for j, s in enumerate(seeds)]
    tasks = [(asdict(jobs[k][1]), jobs[k][2], [s]) for k, _, s in flat]
    if workers <= 1:
        results = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_task, tasks, chunksize=1))
    out = {}
    for (k, _, _), (rows, name) in zip(flat, results):
        key = jobs[k][0]
        prev = out.get(key, ([], name))[0]
        out[key] = (prev + rows, name)
    return out


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), half


HDR = (f"{'strategy':32s} {'SCORE':>5s} {'accur':>5s} {'hours':>5s} {'blocks':>6s} {'plants':>6s} {'/block':>6s} "
       f"{'shots':>5s} {'nudges':>6s} {'blockNDVI':>9s} {'blockHt':>7s} {'recall':>6s} {'f.alarm':>7s} {'partRMSE':>8s} "
       f"{'partF1':>6s} {'swaps':>5s}")


def fmt(name, r):
    return (f"{name[:32]:32s} {r['score']:5.1f} {r['accuracy']:5.1f} {r['hours']:5.2f} {r['blocks_sampled']:6.0%} "
            f"{r['plants_measured']:6.0f} {r['plants_per_block']:6.2f} {r['shots_per_plant']:5.2f} "
            f"{r['nudges_per_plant']:6.2f} {r['block_rmse_ndvi']:9.3f} {r['block_rmse_height_cm']:7.2f} "
            f"{r['recall']:6.0%} {r['false_alarms']:7.0f} {r['part_rmse']:8.3f} {r['part_f1']:6.2f} {r['swaps']:5.2f}")


def _mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if not isinstance(rows[0][k], str)}
