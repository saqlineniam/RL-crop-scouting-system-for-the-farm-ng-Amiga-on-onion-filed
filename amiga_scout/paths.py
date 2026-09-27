"""Where things are: the repository folders and the drone survey on disk."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"             # robot-view maps and bed geometry (made by `python -m amiga_scout extract ...`)
MODELS = REPO / "models"         # trained models and their checkpoints
FIGS = REPO / "figures"
DRONE_ROOT = "D:/Crops/Onion"    # the 2024 Mavic 3M survey (orthomosaics, shapefiles, lab spreadsheets)
