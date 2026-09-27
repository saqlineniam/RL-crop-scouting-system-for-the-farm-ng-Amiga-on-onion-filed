"""The spatial 80/20 train / test split of both fields: disjoint regions with a buffer between them, pieces that stay
inside the training region, and pools for training, model selection and testing that never share crop."""
import numpy as np
import pytest

from amiga_scout.config import SELECT_FIELDS, SPLIT_BUFFER_M, TEST_FIELD, TRAIN_FIELDS
from amiga_scout.fields import (draw_piece, field_pool, load_field, overlaps, parse_name, region_bounds, split_regions,
                                window_acres)


def test_names_parse():
    assert parse_name("f1_05~train@10") == ("f1_05", "train", 10.0)
    assert parse_name("f2_03~test") == ("f2_03", "test", None)
    assert parse_name("f1_05") == ("f1_05", None, None)


def test_regions_are_disjoint_with_a_buffer(real_map):
    key = real_map.split("_")[0]
    s = split_regions(key)
    f = load_field(real_map)
    t, r = s["test"], s["train"]
    ax = s["axis"]
    lo_t, hi_t = (t[0], t[1]) if ax == 0 else (t[2], t[3])
    lo_r, hi_r = (r[0], r[1]) if ax == 0 else (r[2], r[3])
    gap = lo_r - hi_t if lo_r >= hi_t else lo_t - hi_r
    spacing = f["bed_spacing_m"] if ax == 0 else f["step_m"]
    assert gap >= 0 and gap * spacing >= SPLIT_BUFFER_M - spacing, "a buffer strip between the regions"
    assert 0.18 <= s["test_share"] <= 0.22 and s["train_share"] >= 0.65
    assert window_acres(f, *t) + window_acres(f, *r) < f["acres"]
    assert region_bounds(f"{real_map}~test") == t and region_bounds(real_map) is None


def test_pieces_stay_in_the_training_region(real_map):
    f = load_field(real_map)
    b = split_regions(real_map.split("_")[0])["train"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        k0, k1, j0, j1, acres = draw_piece(f, 5, rng, b)
        assert b[0] <= k0 < k1 <= b[1] and b[2] <= j0 < j1 <= b[3] and acres >= 4.5


def test_training_selection_and_test_fields_never_share_crop(real_map):
    test = field_pool(TEST_FIELD)
    assert not any(overlaps(a, b) for a in field_pool(TRAIN_FIELDS) for b in test), "the test regions are never trained on"
    for spec in SELECT_FIELDS:
        assert not any(overlaps(a, b) for a in field_pool(spec) for b in test), "nor used to pick a model"
    assert overlaps("f1_05", "f1_03~test") and overlaps("f1_05~test", "f1_02~test@3")
    assert not overlaps("f1_05~train@10", "f1_02~test") and not overlaps("f1~train", "f2_01~test")
    with pytest.raises(SystemExit):
        field_pool("f1~val")


def test_tiled_fields_are_real_data_and_keep_the_split(real_map):
    from amiga_scout.fields import field_map, tile_of
    key = real_map.split("_")[0]
    one = load_field(real_map)
    region = window_acres(one, *split_regions(key)["train"])
    two = field_map(f"{real_map}~train*2")
    assert tile_of(f"{real_map}~train*3x2@30") == (3, 2) and tile_of(real_map) == (1, 1)
    assert two["acres"] == pytest.approx(2 * region, rel=1e-6), "two copies of the region, nothing else"
    assert two["leaf"].shape[0] == 2 * (split_regions(key)["train"][1] - split_regions(key)["train"][0])
    assert not overlaps(f"{real_map}~train*2@30", f"{real_map}~test") and overlaps(f"{real_map}~test*5", real_map)
    rng = np.random.default_rng(1)
    k0, k1, j0, j1, acres = draw_piece(two, 25, rng)
    assert 22.5 <= acres <= 27.5 and k1 <= two["leaf"].shape[0]
