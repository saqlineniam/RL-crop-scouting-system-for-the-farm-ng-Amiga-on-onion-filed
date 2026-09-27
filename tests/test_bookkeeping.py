"""The running totals behind ACCURACY (updated incrementally for speed) must equal a full recomputation."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.config import BLOCK_TOL_CM, BLOCK_TOL_NDVI, SCENARIOS, Cfg
from amiga_scout.env import AmigaMissionEnv
from amiga_scout.policies import make_policy


def recomputed(env):
    thr = env.cfg.ndvi_stress_threshold
    called = (env.mu_n < thr) & env.vplant
    bn = env.mu_n.reshape(env.NB, env.K).sum(1)
    bh = env.mu_h.reshape(env.NB, env.K).sum(1)
    vb = env.vblock
    return {
        "tp": (int((called & env.stressed).sum()), env.tp),
        "fp": (int((called & ~env.stressed).sum()), env.fp),
        "blk_score_sum": (float((0.5 * np.clip(1 - np.abs(bn / env.K - env.blk_true_n) / BLOCK_TOL_NDVI, 0, 1)
                                 + 0.5 * np.clip(1 - np.abs(bh / env.K - env.blk_true_h) / BLOCK_TOL_CM, 0, 1))[vb].sum()),
                          env.blk_score_sum),
        "blk_mu_n": (bn, env.blk_mu_n),
        "z_mu_sum": (np.bincount(env.zone_of, weights=env.mu_n, minlength=env.Z + 1)[:env.Z], env.z_mu_sum[:env.Z]),
        "z_called": (np.bincount(env.zone_of, weights=(env.mu_n < thr).astype(float), minlength=env.Z + 1)[:env.Z],
                     env.z_called[:env.Z]),
        "blk_count": (np.bincount(env.block_of[env.scan_count > 0], minlength=env.NB), env.blk_count),
    }


@pytest.mark.parametrize("field", ["sq10", "real"])
def test_running_totals(field, request):
    if field == "real":
        field = request.getfixturevalue("real_map")
    env = AmigaMissionEnv(replace(Cfg(), field=field, **SCENARIOS["spots"]), randomize=True)
    env.reset(seed=11)
    pol, n, done = make_policy("random"), 0, False
    while not done:
        _, _, te, tr, _ = env.step(pol.act(env))
        n += 1
        done = te or tr
        if n % 150 == 0 or done:
            for name, (want, got) in recomputed(env).items():
                np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), atol=1e-6, err_msg=name)
