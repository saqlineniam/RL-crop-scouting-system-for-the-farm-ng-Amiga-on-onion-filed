"""Strategies: hand-written rules (rules.py), policies learned by simulation-based policy iteration (qnet.py),
PPO models (model.py, ham.py) and readable trees (tree.py)."""
from pathlib import Path

from .model import ModelPolicy
from .rules import BASELINES, Policy

__all__ = ["BASELINES", "ModelPolicy", "Policy", "make_policy"]


def make_policy(name):
    """A hand-written rule by name (see BASELINES), a learned policy (path to a .pt file, with or without .pt) or a
    PPO model (path without .zip)."""
    if name in BASELINES:
        return BASELINES[name]()
    p = Path(name)
    if p.suffix == ".pt" or p.with_name(p.name + ".pt").exists():
        from .qnet import RolloutPolicy
        return RolloutPolicy(name)
    return ModelPolicy(name)
