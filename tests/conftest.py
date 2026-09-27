"""Shared test setup: import the package from the repository, and find a real map if one has been extracted."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amiga_scout.fields import available_maps  # noqa: E402


@pytest.fixture(scope="session")
def real_map():
    """A robot-view map of a real field (skips the test if none has been extracted yet)."""
    for name in ("f1_05", "f2_05", *available_maps()):
        if name in available_maps():
            return name
    pytest.skip("no real map extracted yet: run `python -m amiga_scout extract-all`")
