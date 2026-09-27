"""A trained MaskablePPO model used as a policy."""
from __future__ import annotations

from pathlib import Path

from ..actions import ACTIONS
from .rules import Policy


class ModelPolicy(Policy):
    positioner = None                 # a PPO model makes its own plant decisions (asked at every decision)

    def __init__(self, path):
        from sb3_contrib import MaskablePPO
        from . import ham  # noqa: F401  (needed to unpickle the hierarchical policy)
        self.model = MaskablePPO.load(path, device="cpu")
        self.name = f"PPO: {Path(path).name}"

    def act(self, env):
        if (self.model.action_space.n != len(ACTIONS)
                or self.model.observation_space.shape != env.observation_space.shape):
            raise SystemExit(f"{self.name} was trained on an older version of the simulator - retrain it with `train`")
        a, _ = self.model.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
        return int(a)
