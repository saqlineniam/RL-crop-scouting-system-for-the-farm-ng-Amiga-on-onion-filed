"""A mission must follow the protocol, never strand, and be scored exactly by its rewards."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.actions import A, DONE, SHOOT_HERE
from amiga_scout.config import SCENARIOS, Cfg
from amiga_scout.env import AmigaMissionEnv
from amiga_scout.evaluation import run_missions
from amiga_scout.policies import make_policy


def run(env, policy, seed):
    """One mission; returns (summary, sum of rewards, accuracy - time already spent before the first decision)."""
    env.reset(seed=seed)
    start = env.acc - env.cfg.hour_value_points * env.time_s / 3600.0      # the drive out, before any decision
    pol = make_policy(policy)
    total, illegal, done = 0.0, 0, False
    while not done:
        a = pol.act(env)
        illegal += int(not env.action_masks()[a])
        _, r, te, tr, info = env.step(a)
        total += r
        done = te or tr
    return info["summary"], total, start, illegal


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("policy", ["minimum", "uncertainty", "random"])
def test_synthetic_mission(scenario, policy):
    env = AmigaMissionEnv(replace(Cfg(), field="sq5", **SCENARIOS[scenario]))
    s, total, start, illegal = run(env, policy, seed=3)
    assert s["blocks_sampled"] == 1.0, "a 10-ft block was left without a sample (protocol)"
    assert not s["stranded"] and s["complete"] and illegal == 0
    assert total + start == pytest.approx(s["score"], abs=1e-6), "rewards must add up to the score"


@pytest.mark.parametrize("scenario", ["patches", "as_is"])
@pytest.mark.parametrize("policy", ["minimum", "battery_fill", "random"])
def test_real_mission(real_map, scenario, policy):
    env = AmigaMissionEnv(replace(Cfg(), field=real_map, **SCENARIOS[scenario]), randomize=(policy == "random"))
    s, total, start, illegal = run(env, policy, seed=5)
    assert s["blocks_sampled"] == 1.0 and not s["stranded"] and s["complete"] and illegal == 0
    assert total + start == pytest.approx(s["score"], abs=1e-6)


def test_protocol_minimum_always_allowed():
    """At a block the minimum (QUICK_1) is always legal; at a plant, shooting from where the robot stopped is
    always legal until the plant is measured, and DONE only after."""
    env = AmigaMissionEnv(replace(Cfg(), field="sq5"))
    env.reset(seed=1)
    pol, done = make_policy("random"), False
    while not done:
        m = env.action_masks()
        if env.phase == "block":
            assert m[A["QUICK_1"]] and not m[DONE] and not m[SHOOT_HERE]
        else:
            assert not m[A["QUICK_1"]]
            assert m[DONE] == env.plant_read
            assert m[SHOOT_HERE] or env.plant_read
        _, _, te, tr, _ = env.step(pol.act(env))
        done = te or tr


@pytest.mark.parametrize("policy", ["random", "all_plants", "uncertainty"])
def test_every_planned_plant_is_measured(real_map, policy):
    """Whatever the policy does with its positioning, every plant it planned in a block ends up with an NDVI
    reading before the robot moves on."""
    env = AmigaMissionEnv(replace(Cfg(), field=real_map), randomize=True)
    env.reset(seed=7)
    pol, done, checked = make_policy(policy), False, 0
    while not done:
        if env.phase == "block":
            b = env.blk
        _, _, te, tr, _ = env.step(pol.act(env))
        done = te or tr
        if env.phase == "block" and not done and env.plan:
            assert env.blk_count[b] >= len(set(env.plan)), "a planned plant was left without a reading"
            checked += 1
    assert checked > 50


def test_shooting_position_moves_the_cameras():
    env = AmigaMissionEnv(replace(Cfg(), field="sq5"))
    env.reset(seed=2)
    env.step(A["CAREFUL_1"])
    x0, t0 = env.x0, env.time_s
    env.step(A["SHOOT_BACK_20CM"])
    assert env.x == pytest.approx(x0 - 0.2) and env.time_s > t0 and env.plant_read and env.phase == "plant"
    env.step(DONE)
    assert env.phase == "block"


def test_same_seed_same_mission():
    cfg = replace(Cfg(), field="sq10")
    assert run_missions(cfg, "adaptive", [4, 5])[0] == run_missions(cfg, "adaptive", [4, 5])[0]


def test_observations_finite_and_in_range(real_map):
    env = AmigaMissionEnv(replace(Cfg(), field=real_map, **SCENARIOS["spots"]))
    env.reset(seed=2)
    pol, done = make_policy("adaptive"), False
    while not done:
        o = env._obs()
        assert np.isfinite(o).all() and (np.abs(o) <= 5).all() and o.shape == env.observation_space.shape
        _, _, te, tr, _ = env.step(pol.act(env))
        done = te or tr


@pytest.mark.parametrize("policy", ["random", "two_per_block"])
def test_protocol_holds_in_wind(real_map, policy):
    """Wind makes the robot stop less precisely, so plants often fall outside the NIR camera's narrow view and
    need re-shots. The energy plan learns that overhead, so every 10-ft block is still sampled."""
    from amiga_scout.config import PERTURBATIONS
    cfg = replace(Cfg(), field=real_map)
    cfg = replace(cfg, **PERTURBATIONS["wind: leaves sway 3x, stops 2.5x less exact"](cfg))
    s, total, start, illegal = run(AmigaMissionEnv(cfg), policy, seed=90000)
    assert s["blocks_sampled"] == 1.0 and not s["stranded"] and illegal == 0
    assert total + start == pytest.approx(s["score"], abs=1e-6)
