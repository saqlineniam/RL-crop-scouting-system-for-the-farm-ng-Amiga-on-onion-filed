"""Rollout training: forks of a mission share the measurement noise (common random numbers), the mission's own
positioner gives the same missions as asking the strategy at every plant, choices are valued against the protocol
choice, and a learned policy saves, loads and runs a mission by the protocol."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.actions import A, DONE, N_BLOCK_ACTIONS, SHOOT_HERE
from amiga_scout.config import SCENARIOS, Cfg
from amiga_scout.env import AmigaMissionEnv
from amiga_scout.env.observation import OBS_NAMES
from amiga_scout.policies import make_policy
from amiga_scout.policies.qnet import ChoiceModel, RolloutPolicy, save_policy
from amiga_scout.policies.rules import position_rule
from amiga_scout.rollout import BLOCK_LEVEL, PLANT_LEVEL, REF_BLOCK, block_values, fit_level, plant_values


def _env(real_map, scenario="spots", seed=3, blocks=5):
    env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}~train@5", **SCENARIOS[scenario]))
    env.reset(seed=seed)
    pol = make_policy("minimum")
    env.positioner = pol.positioner
    for _ in range(blocks):
        env.step(pol.block_act(env))
    return env, pol


def test_forks_share_noise_by_salt(real_map):
    env, _ = _env(real_map)
    before = env.prec_n.copy(), env.block_decisions
    e1, e2, e3 = env.fork(1), env.fork(1), env.fork(2)
    for e in (e1, e2, e3):
        e.step(A["QUICK_6"])
    assert np.array_equal(e1.wsum_n, e2.wsum_n), "the same salt: the same readings"
    assert not np.array_equal(e1.wsum_n, e3.wsum_n), "another salt: other readings"
    assert np.array_equal(env.prec_n, before[0]) and env.block_decisions == before[1], "the original is untouched"


@pytest.mark.parametrize("rule", ["adaptive", "all_plants"])
def test_positioner_inside_the_mission_is_the_same_mission(real_map, rule):
    out = []
    for inside in (False, True):
        env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}~test", **SCENARIOS["patches"]))
        env.reset(seed=7)
        pol = make_policy(rule)
        env.positioner = pol.positioner if inside else None
        done, total = False, 0.0
        while not done:
            _, r, te, tr, _ = env.step(pol.act(env))
            total += r
            done = te or tr
        out.append((total, env.summary()))
    (r0, s0), (r1, s1) = out
    assert r0 == pytest.approx(r1) and s0["score"] == pytest.approx(s1["score"])
    assert s0["plants_measured"] == s1["plants_measured"] and s1["decisions"] == s1["block_decisions"]


def test_choices_are_valued_against_the_protocol_choice(real_map):
    env, pol = _env(real_map, blocks=12)
    vals = block_values(env, pol, 3, [1, 2])
    m = env.action_masks()
    assert set(vals) == {a for a in BLOCK_LEVEL if m[a]} and REF_BLOCK in vals
    assert np.allclose(vals[REF_BLOCK], 0.0) and all(np.isfinite(v).all() for v in vals.values())
    assert (vals[A["QUICK_6"]][:, 1] > 0).all(), "measuring every plant takes longer than one"
    env.positioner = None
    env.step(A["CAREFUL_2"])                              # now at a plant, asked where to shoot
    pv = plant_values(env, position_rule, [1, 2])
    assert set(pv) <= set(PLANT_LEVEL) and np.allclose(pv[DONE if env.action_masks()[DONE] else SHOOT_HERE], 0.0)
    assert (pv[A["SHOOT_BACK_30CM"]][:, 1] > 0).all(), "moving the cameras takes time"


def test_learned_policy_saves_loads_and_follows_the_protocol(real_map, tmp_path):
    rng = np.random.default_rng(0)
    d, h, n = len(OBS_NAMES), 8, N_BLOCK_ACTIONS
    weights = [(rng.normal(0, 0.3, (d, h)), np.zeros(h)), (rng.normal(0, 0.3, (h, h)), np.zeros(h)),
               (rng.normal(0, 0.3, (h, 2 * n)), np.zeros(2 * n))]
    block = ChoiceModel(BLOCK_LEVEL, REF_BLOCK, np.zeros(d), np.ones(d), 0.05, 0.01, weights)
    path = tmp_path / "tiny.pt"
    save_policy(path, block=block, base="minimum")
    pol = make_policy(str(path))
    assert isinstance(pol, RolloutPolicy) and pol.positioner is position_rule
    env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}~test", **SCENARIOS["spots"]))
    env.reset(seed=90000)
    env.positioner = pol.positioner
    done, illegal = False, 0
    while not done:
        a = pol.act(env)
        illegal += int(not env.action_masks()[a])
        _, _, te, tr, _ = env.step(a)
        done = te or tr
    s = env.summary()
    assert illegal == 0 and s["blocks_sampled"] == 1.0 and not s["stranded"]


def test_fit_level_learns_which_choice_pays():
    """Synthetic check of the regression: action 2 pays when the first input is positive, action 0 otherwise."""
    rng = np.random.default_rng(1)
    n, d, acts = 3000, 6, [0, 1, 2]
    X = rng.normal(size=(n, d)).astype(np.float32)
    acc = np.zeros((n, 3, 2))
    acc[:, 2, :] = (0.1 * np.sign(X[:, :1]) + rng.normal(0, 0.02, (n, 2)))
    acc[:, 1, :] = -0.05
    hrs = np.zeros((n, 3, 2))
    model = fit_level(X, acc, hrs, np.zeros(n, int), np.arange(n) // 30, acts, 0, 1.0, hidden=32, epochs=30,
                      log=lambda *a, **k: None)
    pa, ph = model.predict(X[:200])
    pick = (pa - ph).argmax(1)
    assert np.mean(pick == np.where(X[:200, 0] > 0, 2, 0)) > 0.85         # (the boundary at 0 is the hard part)
    assert model.info["gain_vs_policy"] > 0.03                               # (at most 0.05: half the states pay 0.1)


def test_ensemble_averages_its_networks_and_repeat_keeps_the_policy_density(real_map):
    from amiga_scout.rollout import Repeat
    rng = np.random.default_rng(3)
    d, h, n = len(OBS_NAMES), 4, N_BLOCK_ACTIONS
    nets = [[(rng.normal(size=(d, h)), np.zeros(h)), (rng.normal(size=(h, 2 * n)), np.zeros(2 * n))] for _ in range(2)]
    both = ChoiceModel(BLOCK_LEVEL, REF_BLOCK, np.zeros(d), np.ones(d), 1.0, 1.0, nets)
    one = [ChoiceModel(BLOCK_LEVEL, REF_BLOCK, np.zeros(d), np.ones(d), 1.0, 1.0, [net]) for net in nets]
    x = rng.normal(size=d)
    assert np.allclose(both.predict(x)[0], 0.5 * (one[0].predict(x)[0] + one[1].predict(x)[0]))
    assert ChoiceModel.from_dict(both.to_dict()).predict(x)[0] == pytest.approx(both.predict(x)[0])
    env, _ = _env(real_map)
    m = env.action_masks()
    a = A["QUICK_5"] if m[A["QUICK_5"]] else A["QUICK_1"]
    assert Repeat(a).block_act(env) == a


def test_block_chooser_blends_by_size_and_needs_agreement_for_careful(real_map, tmp_path):
    from amiga_scout.actions import decode
    from amiga_scout.policies.qnet import BlockChooser
    rng = np.random.default_rng(5)
    d, h, n = len(OBS_NAMES), 6, N_BLOCK_ACTIONS

    def model(per_block):
        nets = [[(rng.normal(0, 0.3, (d, h)), np.zeros(h)), (rng.normal(0, 0.3, (h, 2 * n)), np.zeros(2 * n))]
                for _ in range(3)]
        return ChoiceModel(BLOCK_LEVEL, REF_BLOCK, np.zeros(d), np.ones(d), 0.05, 0.01, nets, per_block=per_block)
    ch = BlockChooser(model(False), model(True), blend=(300, 900), careful_margin=1.0)
    assert ch.weight(100) == 0.0 and ch.weight(2000) == 1.0 and ch.weight(600) == pytest.approx(0.5)
    path = tmp_path / "chooser.pt"
    save_policy(path, block=ch, base="minimum")
    pol = make_policy(str(path))
    assert isinstance(pol.block, BlockChooser) and pol.block.blend == (300, 900)
    never = BlockChooser(ch.raw, ch.pb, careful_margin=1e9)
    env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}~test", **SCENARIOS["as_is"]))
    env.reset(seed=90001)
    env.positioner = pol.positioner
    done, careful = False, 0
    while not done:
        a = never.best(env, 1.0)
        assert env.action_masks()[a]
        careful += decode(a)[0] == "careful"
        _, _, te, tr, _ = env.step(a)
        done = te or tr
    assert careful == 0 and env.summary()["blocks_sampled"] == 1.0
