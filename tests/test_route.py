"""The fixed route on a real field: every crop block is sampled, passes are evenly spaced, the order is complete."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.config import BARE_COVER_FRAC, Cfg
from amiga_scout.env import AmigaMissionEnv
from amiga_scout.fields import load_field


def crop_blocks(f, corner, bed, K=6):
    """Crop blocks (enough mapped spots, enough canopy) of one bed, in the route's orientation."""
    cover, good = f["cover"], f["good"]
    if corner & 1:
        cover, good = cover[::-1], good[::-1]
    if corner & 2:
        cover, good = cover[:, ::-1], good[:, ::-1]
    nb = good.shape[1] // K
    g = good[bed, :nb * K].reshape(nb, K)
    cv = np.where(g, cover[bed, :nb * K].reshape(nb, K), 0).sum(1) / np.maximum(g.sum(1), 1)
    return np.flatnonzero((g.sum(1) >= K - 1) & (cv >= BARE_COVER_FRAC * f["cover_median"]))


@pytest.mark.parametrize("corner", range(4))
@pytest.mark.parametrize("start", [0, 5, 11])
def test_every_crop_block_is_on_the_route(real_map, corner, start):
    env = AmigaMissionEnv(replace(Cfg(), field=real_map, route_start_bed=start, route_corner=corner))
    f = load_field(real_map)
    spans = {k: (b0, b1) for k, b0, b1 in env.map_info["spans"]}
    assert np.all(np.diff(env.map_info["passes"]) == env.cfg.pass_every_n_beds), "a bed was skipped between passes"
    for bed in range(start, f["leaf"].shape[0], env.cfg.pass_every_n_beds):
        crop = crop_blocks(f, corner, bed)
        if len(crop):
            assert bed in spans and spans[bed][0] <= crop[0] and crop[-1] <= spans[bed][1]
    # the route visits every crop block exactly once, and nothing else
    assert sorted(env.route_rank[env.vblock]) == list(range(env.NBv))
    assert (env.route_rank[~env.vblock] == -1).all()
    env.reset(seed=start)
    assert np.isfinite(env.ndvi[env.vplant]).all() and np.isfinite(env.height[env.vplant]).all()


def test_synthetic_route_is_complete():
    env = AmigaMissionEnv(replace(Cfg(), field="sq10"))
    assert env.vblock.all() and sorted(env.route_rank) == list(range(env.NB))


def test_parts_follow_the_lab_rule(real_map):
    env = AmigaMissionEnv(replace(Cfg(), field=real_map))
    target = round(env.cfg.points_per_30_acres * env.cfg.field_acres / 30.0)
    assert abs(env.Z - target) <= max(1, target // 20), "50 parts per 30 acres"
