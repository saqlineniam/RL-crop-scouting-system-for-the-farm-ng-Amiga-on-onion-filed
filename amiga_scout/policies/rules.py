"""The hand-written strategies the RL policy has to beat.

Each decides at blocks (block_act: how carefully, how many plants) and positions the cameras at plants with the same
hand-written rule (plant_act, position_rule) - so a comparison with the RL policy, which learns both, is fair.
"""
from __future__ import annotations

import numpy as np

from ..actions import A, DONE, FIRST_SHOOT, SHOOT_HERE, encode
from ..config import PLANTS_PER_BLOCK

POOR_DEPTH_VIEW = 0.4        # rule tuning: a first depth view with less fill than this counts as poor


class Policy:
    """A strategy: block_act at a block, plant_act at a plant. `positioner` (set below: the hand-written
    positioning) is handed to the mission, which then makes the plant decisions with it by itself; None = the
    strategy is asked at every plant too (act)."""
    name = "policy"

    def act(self, env):
        return self.plant_act(env) if env.phase == "plant" else self.block_act(env)

    def block_act(self, env):
        raise NotImplementedError

    def plant_act(self, env):
        return position_rule(env)


def _best_seen(values, m, exclude=None):
    """The legal SHOOT action whose position has the best live reading so far (None if nothing was seen)."""
    best, best_v = None, -1.0
    for k, v in enumerate(values):
        a = FIRST_SHOOT + k
        if np.isfinite(v) and m[a] and k != exclude and v > best_v:
            best, best_v = a, float(v)
    return best


def position_rule(env):
    """The hand-written positioning: a quick plant is shot where the robot stopped; a careful plant from the position
    with the best depth quality seen so far and then - if that shot's NIR image had too few clean leaf pixels - once
    more from the best NIR position seen (the careful procedure of the earlier versions, now driven by the live
    readings the robot gathered while driving up). If a shot gave no NDVI (the plant outside the NIR camera's view),
    the next one goes to the position with the best NIR reading."""
    m = env.action_masks()
    if not env.plant_read:
        if env.plant_attempts > 0:
            return _best_seen(env.seen_qn, m) or SHOOT_HERE
        if env.block_mode == "careful":
            return _best_seen(env.seen_qd, m) or SHOOT_HERE
        return SHOOT_HERE
    if env.block_mode == "careful" and env.plant_shots < 2 and env.last_qn < env.cfg.nir_reshoot_below:
        here = env._offset_index(env.x - env.x0)
        a = _best_seen(env.seen_qn, m, exclude=here)
        if a is not None:
            return a
    return DONE if m[DONE] else SHOOT_HERE


Policy.positioner = staticmethod(position_rule)


def _largest_legal(m, mode, k):
    """The action with the most plants <= k in this mode that the masks allow (QUICK_1 if none)."""
    for kk in range(min(int(k), PLANTS_PER_BLOCK), 0, -1):
        if m[encode(mode, kk)]:
            return encode(mode, kk)
    return A["QUICK_1"]


def _care_for_view(env):
    """Measure the protocol plant carefully only if the cameras' first view of it is poor."""
    return "careful" if (env.pv_qd < POOR_DEPTH_VIEW or env.pv_qn < env.cfg.nir_reshoot_below) else "quick"


class EvenEffort(Policy):
    """The same number of plants in every block, as far as the batteries allow. k=1: the protocol minimum
    (one quick shot per block, fastest); k=PLANTS_PER_BLOCK: every plant (slowest)."""

    def __init__(self, k=1, mode="careful"):
        self.k, self.mode = k, ("quick" if k == 1 else mode)
        how = ", carefully" if mode == "careful" else ", one quick shot each"
        self.name = "1 plant per block, quick (minimum)" if k == 1 else (
            ("every plant" if k >= PLANTS_PER_BLOCK else f"{k} plants per block") + how)

    def block_act(self, env):
        return _largest_legal(env.action_masks(), self.mode, self.k)


class AdaptiveRule(Policy):
    """Hand-written adaptive tree: one plant per block (careful only if the first view of it is poor); 4
    plants, carefully, when the block looks interesting - the map expects stress in it, or the previous
    block's readings were near the stress threshold, uneven or surprising."""
    name = "adaptive rule (hand tree)"

    def block_act(self, env):
        b, K, thr = env.blk, env.K, env.cfg.ndvi_stress_threshold
        prev = env._block_at(env.pos - K)
        reads = env.block_readings(prev) if prev is not None else np.zeros(0)
        interesting = (env.pstress[b * K:b * K + K].max() > 0.2
                       or (reads.size > 0 and (reads.min() < thr + 0.08 or env.blk_maxz[prev] > 2.0))
                       or (reads.size >= 2 and reads.std() > 0.05))
        m = env.action_masks()
        return _largest_legal(m, "careful", 4) if interesting else _largest_legal(m, _care_for_view(env), 1)


class UncertaintyRule(Policy):
    """Map-driven active sampling (the Gaussian-process idea of Kumar et al. 2019 for field phenotyping):
    measure, carefully, as many plants as the block has plants whose stress call the map is still unsure
    about (5% < P(stressed) < 95%); one plant where the map is sure."""
    name = "uncertainty rule (map-driven)"

    def block_act(self, env):
        p = env.pstress[env.blk * env.K:env.blk * env.K + env.K]
        unsure = int(((p > 0.05) & (p < 0.95)).sum())
        m = env.action_masks()
        return _largest_legal(m, "careful", unsure) if unsure else _largest_legal(m, _care_for_view(env), 1)


class BatteryFill(Policy):
    """Use the batteries: the energy left after reserving one plant in every block (spare packs included) is
    spread evenly over the blocks still ahead - as many careful plants per block as that allows. On a small
    field that is every plant; on a big one, as many as the packs (and their swap) allow."""
    name = "battery-filling rule (even)"

    def block_act(self, env):
        return _largest_legal(env.action_masks(), "careful", 1 + max(0.0, env.spare_plants_per_block()))


class RandomValid(Policy):
    """Any legal action, blocks and positions alike (a floor, and a stress test of the masks)."""
    name = "random valid actions"
    positioner = None                 # random at plants too: asked at every decision

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        return int(self.rng.choice(np.flatnonzero(env.action_masks())))


BASELINES = {"minimum": lambda: EvenEffort(1), "two_per_block": lambda: EvenEffort(2),
             "all_plants": lambda: EvenEffort(PLANTS_PER_BLOCK), "all_plants_quick": lambda: EvenEffort(PLANTS_PER_BLOCK, "quick"),
             "adaptive": AdaptiveRule, "uncertainty": UncertaintyRule, "battery_fill": BatteryFill,
             "random": RandomValid}
