"""The drone -> robot pipeline: pixel helpers, and sanity of the bed geometry and maps already extracted."""
import json

import numpy as np
import pytest

from amiga_scout.drone.extract import load_map
from amiga_scout.drone.io import _ndvi, _profile, otsu
from amiga_scout.paths import DATA


@pytest.mark.parametrize("mixed", [0, 3000])
def test_otsu_splits_soil_from_leaves(mixed):
    rng = np.random.default_rng(0)
    soil, leaves = rng.normal(0.10, 0.03, 5000), rng.normal(0.55, 0.06, 5000)
    edge = rng.uniform(0.15, 0.50, mixed)                     # pixels on leaf edges: part soil, part leaf
    thr = otsu(np.r_[soil, leaves, edge])
    assert (soil > thr).mean() < 0.01 and (leaves < thr).mean() < 0.01    # soil and leaves end up on their own side


def test_ndvi_ignores_nodata():
    red = np.array([0.1, -32767.0, 0.2]); nir = np.array([0.5, 0.4, 0.2])
    nd = _ndvi(red, nir)
    assert nd[0] == pytest.approx(0.4 / 0.6) and np.isnan(nd[1]) and nd[2] == pytest.approx(0.0)


def test_profile_shares():
    v = np.array([0.001, 0.005, 0.03, 0.031]); plant = np.array([1.0, 0.0, 1.0, 1.0])
    share, n = _profile(None, v, plant, 0.0, 3)
    assert list(n) == [2, 2, 0] and share[0] == 0.5 and share[1] == 1.0


def test_bed_geometry_is_plausible():
    path = DATA / "geometry.json"
    if not path.exists():
        pytest.skip("run `python -m amiga_scout geometry` first")
    for field, g in json.loads(path.read_text()).items():
        widths = np.diff(g["furrows_m"])
        assert 1.5 < np.median(widths) < 2.3, f"{field}: Vidalia beds are ~6 ft"
        assert (widths > 0).all() and g["halves_agree_r"] > 0.6, f"{field}: beds must be straight along the field"


def test_extracted_maps_are_consistent(real_map):
    f, d = real_map.split("_")
    m = load_map(f, d)
    shape = m["leaf_ndvi"].shape
    assert m["cover"].shape == shape and m["mapped"].shape == shape and len(m["furrows_m"]) == shape[0] + 1
    good = m["mapped"] > 0.6
    assert np.nanmin(m["cover"][good]) >= 0 and np.nanmax(m["cover"][good]) <= 1
    assert np.nanmax(np.abs(m["leaf_ndvi"][good])) <= 1
