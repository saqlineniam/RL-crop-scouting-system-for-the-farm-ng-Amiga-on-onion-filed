"""Random pieces of a real field ("f1_05@10"): the right size, inside the field, a new one every mission, and a
mission on a piece still follows the protocol."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.config import SCENARIOS, Cfg
from amiga_scout.env import AmigaMissionEnv
from amiga_scout.fields import base_map, field_pool
from amiga_scout.policies import make_policy


@pytest.mark.parametrize("acres", [5, 10, 15])
def test_piece_size_and_place(real_map, acres):
    env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}@{acres}"))
    seen = set()
    for seed in range(6):
        env.reset(seed=seed)
        k0, k1, j0, j1 = env.piece
        assert 0 <= k0 < k1 and 0 <= j0 < j1
        assert 0.9 * acres <= env.cfg.field_acres <= 1.05 * acres, "a piece is about the size asked for"
        assert env.cfg.field == f"{real_map}@{acres}" and sorted(env.route_rank[env.vblock]) == list(range(env.NBv))
        seen.add(env.piece)
    assert len(seen) > 1, "a new piece every mission"


def test_piece_mission_follows_protocol(real_map):
    env = AmigaMissionEnv(replace(Cfg(), field=f"{real_map}@5", **SCENARIOS["patches"]), randomize=True)
    env.reset(seed=4)
    start = env.acc - env.cfg.hour_value_points * env.time_s / 3600.0
    pol, total, done = make_policy("random"), 0.0, False
    while not done:
        _, r, te, tr, info = env.step(pol.act(env))
        total += r
        done = te or tr
    s = info["summary"]
    assert s["blocks_sampled"] == 1.0 and not s["stranded"] and s["complete"]
    assert total + start == pytest.approx(s["score"], abs=1e-6)
    assert np.isfinite(env.ndvi[env.vplant]).all()


def test_piece_specs_expand(real_map):
    field = real_map.split("_")[0]
    pool = field_pool(f"{field}@10 {real_map}")
    assert real_map in pool and f"{real_map}@10" in pool and all(base_map(m).startswith(field) for m in pool)
    with pytest.raises(SystemExit):
        field_pool(f"{real_map}@0")
