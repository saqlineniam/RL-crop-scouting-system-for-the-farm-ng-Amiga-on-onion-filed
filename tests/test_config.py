"""Settings, field pools and the train/test split."""
from types import SimpleNamespace

import pytest

from amiga_scout.config import SCENARIOS, TEST_FIELD, TRAIN_FIELDS, Cfg
from amiga_scout.evaluation import build_cfg, parse_sets
from amiga_scout.fields import field_pool, overlaps


def test_set_overrides():
    assert parse_sets(["hour_value_points=3", "zapper_enabled=false", "field=sq10"]) == dict(
        hour_value_points=3, zapper_enabled=False, field="sq10")
    cfg = build_cfg(SimpleNamespace(set=["hour_value_points=1.5"], field="sq5"), "spots")
    assert cfg.hour_value_points == 1.5 and cfg.field == "sq5" and cfg.spot_stress_prob == SCENARIOS["spots"]["spot_stress_prob"]


def test_unknown_setting_is_refused():
    with pytest.raises(SystemExit):
        build_cfg(SimpleNamespace(set=["no_such_setting=1"], field=None))


def test_missing_map_is_explained():
    with pytest.raises(SystemExit, match="extract"):
        field_pool("f9_99")


def test_train_and_test_fields_never_overlap(real_map):
    train, test = field_pool(TRAIN_FIELDS), field_pool(TEST_FIELD)
    assert not any(overlaps(a, b) for a in train for b in test), "the test regions must never be trained on"


def test_default_field_type_is_known():
    assert set(SCENARIOS) == {"patches", "spots", "poorer", "as_is"} and Cfg().field
