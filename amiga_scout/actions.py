"""The robot's decisions: a conditional action tree (HAM-PPO style, masks at both levels).

    arriving at a 10-ft block    QUICK_k / CAREFUL_k   how carefully, and how many of its plants (k = 1 .. 6):
                                                       quick = one frame per shot, careful = frames averaged and
                                                       a second shot allowed
    at every plant it measures   SHOOT_...             where to shoot from: a little back, here, or a little ahead of
                                                       where it stopped (SHOOT_OFFSETS_M) - the robot's own positioning
                                 DONE                  (careful plants, after the first shot) move on

The masks make only the right kind legal at each moment: block actions at a block, SHOOT at a plant (and DONE once
it has been shot at least once). QUICK_1 + SHOOT_HERE is the protocol minimum.
"""
from __future__ import annotations

from .config import PLANTS_PER_BLOCK, SHOOT_OFFSETS_M


def _position_name(offset_m):
    cm = int(round(offset_m * 100))
    return "HERE" if cm == 0 else (f"BACK_{-cm}CM" if cm < 0 else f"AHEAD_{cm}CM")


BLOCK_ACTIONS = ([f"QUICK_{k}" for k in range(1, PLANTS_PER_BLOCK + 1)]
                 + [f"CAREFUL_{k}" for k in range(1, PLANTS_PER_BLOCK + 1)])
SHOOT_ACTIONS = [f"SHOOT_{_position_name(o)}" for o in SHOOT_OFFSETS_M]
ACTIONS = BLOCK_ACTIONS + SHOOT_ACTIONS + ["DONE"]
A = {n: i for i, n in enumerate(ACTIONS)}
BRANCHES = ["QUICK", "CAREFUL", "SHOOT", "NEXT"]
ACTION_BRANCH = [0] * PLANTS_PER_BLOCK + [1] * PLANTS_PER_BLOCK + [2] * len(SHOOT_OFFSETS_M) + [3]
N_BLOCK_ACTIONS = len(BLOCK_ACTIONS)
FIRST_SHOOT = A[SHOOT_ACTIONS[0]]
SHOOT_HERE = A["SHOOT_HERE"]
DONE = A["DONE"]


def decode(a):
    """Block action index -> (mode, plants in this block): ('quick', k) or ('careful', k)."""
    if int(a) >= N_BLOCK_ACTIONS:
        raise ValueError(f"{ACTIONS[int(a)]} is not a block action")
    mode, k = ACTIONS[int(a)].split("_")
    return mode.lower(), int(k)


def encode(mode, k):
    return A[f"{mode.upper()}_{int(k)}"]


def is_block_action(a):
    return int(a) < N_BLOCK_ACTIONS


def shoot_offset(a):
    """SHOOT action index -> how far (m) from where the robot stopped it shoots from."""
    return SHOOT_OFFSETS_M[int(a) - FIRST_SHOOT]


def shoot_action(offset_m):
    """The SHOOT action closest to this offset."""
    k = min(range(len(SHOOT_OFFSETS_M)), key=lambda j: abs(SHOOT_OFFSETS_M[j] - offset_m))
    return FIRST_SHOOT + k
