"""explain: turn a trained policy into a small decision tree people can read (the VIPER idea, Bastani et
al. 2018, in its simplest form: imitate the network with a tree)."""
from __future__ import annotations

import numpy as np

from ..config import FIELD_ACRES, PLANTS_PER_BLOCK, ZAP_EVERY_MIN
from .rules import Policy


# ---- explain: turn a trained policy into a small decision tree people can read -------
# (the VIPER idea, Bastani et al. 2018, in its simplest form: imitate the network with a tree)
# Inputs are converted back to real units first, so the printed thresholds mean something.
HUMAN = {  # observation name -> (scale back to real units, readable name)
    "block_ndvi_vs_threshold": (0.2, "block_ndvi_estimate_minus_threshold"),
    "prev_block_surprise": (5.0, "previous_block_most_surprising_reading_sigmas"),
    "block_p_stressed": (100.0, "block_chance_stressed_pct"), "block_max_p_stressed": (100.0, "block_max_chance_stressed_pct"),
    "block_uncertain_share": (100.0, "block_plants_unsure_pct"),
    "prev_block_ndvi_vs_threshold": (0.2, "previous_block_ndvi_minus_threshold"),
    "prev_block_plants": (float(PLANTS_PER_BLOCK), "previous_block_plants"), "prev_block_spread": (0.1, "previous_block_ndvi_spread"),
    "next_block_p_stressed": (100.0, "next_block_chance_stressed_pct"),
    "other_pass_p_stressed": (100.0, "block_beside_chance_stressed_pct"),
    "other_pass_plants": (float(PLANTS_PER_BLOCK), "block_beside_plants"),
    "stressed_share_so_far": (100.0, "stressed_readings_so_far_pct"), "battery_left": (100.0, "battery_pct"),
    "spare_plants_per_block": (float(PLANTS_PER_BLOCK - 1), "spare_careful_plants_per_block"),
    "protocol_energy_ratio": (300.0, "protocol_energy_need_pct_of_available"),
    "time_to_next_zap": (ZAP_EVERY_MIN, "minutes_to_next_zap"), "blocks_done_frac": (100.0, "blocks_done_pct"),
    "field_acres": (FIELD_ACRES, "field_acres"),
    "camera_offset_now": (30.0, "camera_cm_from_stop"), "plants_left_in_plan": (float(PLANTS_PER_BLOCK), "plants_left_in_block_plan"),
    "plant_readings": (2.0, "shots_with_ndvi_on_this_plant"), "plant_p_stressed": (100.0, "plant_chance_stressed_pct"),
}


def human_features(names):
    scale = np.array([HUMAN.get(n, (1.0, n))[0] for n in names], dtype=np.float32)
    return scale, [HUMAN.get(n, (1.0, n))[1] for n in names]


class TreePolicy(Policy):
    """Runs the extracted decision tree; if its first choice is illegal, takes its next legal choice."""

    def __init__(self, tree, scale, name, positioner=None):
        self.tree, self.scale, self.name, self.positioner = tree, scale, name, positioner

    def act(self, env):
        m = env.action_masks()
        if m.sum() == 1:
            return int(np.flatnonzero(m)[0])
        proba = self.tree.predict_proba((env._obs() * self.scale)[None])[0]
        for k in np.argsort(-proba):
            a = int(self.tree.classes_[k])
            if m[a]:
                return a
        return int(np.flatnonzero(m)[0])


def tree_to_text(tree, names, labels):
    """if/else text of a fitted sklearn tree; branches whose leaves all give the same action are merged."""
    t = tree.tree_
    total = t.n_node_samples[0]
    lines = []

    def leaf_actions(n):
        if t.children_left[n] == -1:
            return {int(np.argmax(t.value[n]))}
        return leaf_actions(t.children_left[n]) | leaf_actions(t.children_right[n])

    def rec(n, depth):
        pad = "|   " * depth
        acts = leaf_actions(n)
        if len(acts) == 1:
            lines.append(f"{pad}=> {labels[acts.pop()]}   ({t.n_node_samples[n] / total:.0%} of decisions)")
            return
        f, thr = names[t.feature[n]], t.threshold[n]
        lines.append(f"{pad}if {f} <= {thr:.2f}:")
        rec(t.children_left[n], depth + 1)
        lines.append(f"{pad}else ({f} > {thr:.2f}):")
        rec(t.children_right[n], depth + 1)

    rec(0, 0)
    return "\n".join(lines) + "\n"
