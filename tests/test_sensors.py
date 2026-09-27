"""The camera model against its datasheets and against physics it must obey."""
from dataclasses import replace

import numpy as np
import pytest

from amiga_scout.config import Cfg
from amiga_scout.sensors import (D455_DEPTH, JAI_AD130GE, defocus_px, depth_focal_px, depth_footprint, depth_gsd,
                                 depth_of_field, depth_quality, depth_rms, lens_fov_deg, min_z, nir_footprint,
                                 nir_quality, pixel_ndvi_noise, shot_statistics, viewpoint_error)


def test_d455_numbers_match_the_datasheet():
    # the stereo formula (focal length x baseline / 126 disparities) reproduces the datasheet's minimum distances
    for res, table in (("1280x720", 0.52), ("848x480", 0.35)):
        assert min_z(D455_DEPTH, res) == table
        assert depth_focal_px(D455_DEPTH, res) * D455_DEPTH.baseline_m / 126 == pytest.approx(table, rel=0.05)
    # the spec: RMS error <= 2% of the distance up to 4 m (HD, textured target)
    assert depth_rms(D455_DEPTH, "1280x720", 4.0, 0.08) <= 0.02 * 4.0


def test_depth_noise_grows_with_distance_squared():
    a, b = depth_rms(D455_DEPTH, "848x480", 0.5, 0.08), depth_rms(D455_DEPTH, "848x480", 1.0, 0.08)
    assert b / a == pytest.approx(4.0)
    assert depth_rms(D455_DEPTH, "1280x720", 1.0, 0.08) < b         # more pixels, less noise


def test_no_depth_closer_than_min_z():
    cfg = Cfg()
    zmin = min_z(D455_DEPTH, cfg.depth_resolution)
    assert depth_quality(cfg, zmin - 0.01, 0.0, 0.0, 0.01) == 0.0
    assert depth_quality(cfg, zmin + 0.2, 0.0, 0.0, 0.01) > 0.3


def test_thin_leaves_far_away_give_little_depth():
    cfg = Cfg()
    assert depth_quality(cfg, 0.9, 0.0, 0.0, 0.004) < depth_quality(cfg, 0.9, 0.0, 0.0, 0.012)
    assert depth_gsd(D455_DEPTH, "848x480", 0.9) > depth_gsd(D455_DEPTH, "848x480", 0.5)


def test_off_centre_views_see_less():
    cfg = Cfg()
    centred = depth_quality(cfg, 0.6, 0.0, 0.0, 0.01)
    assert depth_quality(cfg, 0.6, 0.15, 0.0, 0.01) < centred          # neighbours hide more of the top
    assert depth_quality(cfg, 0.6, 0.0, 0.15, 0.01) < centred          # the plant's row is off to the side
    assert depth_quality(cfg, 0.4, 0.3, 0.0, 0.01) == 0.0              # outside the depth camera's view


def test_camera_views_across_the_bed():
    cfg = Cfg()
    along, across = depth_footprint(cfg, 0.6)
    assert across > along                                              # wide side across the bed
    assert across == pytest.approx(2 * 0.6 * np.tan(np.radians(87 / 2)))
    n_along, n_across = nir_footprint(cfg, 0.6)
    f = cfg.nir_lens_focal_mm
    assert n_across == pytest.approx(0.6 * 4.86 / f) and n_along == pytest.approx(0.6 * 3.63 / f)


def test_nir_optics():
    f = 6e-3
    assert defocus_px(f, 4.0, 0.7, 0.7, JAI_AD130GE.pixel_m) == pytest.approx(0.0, abs=1e-9)   # sharp at focus
    assert defocus_px(f, 4.0, 0.7, 0.4, JAI_AD130GE.pixel_m) > defocus_px(f, 4.0, 0.7, 0.6, JAI_AD130GE.pixel_m)
    near, far = depth_of_field(f, 4.0, 0.7, 2 * JAI_AD130GE.pixel_m)
    assert near < 0.7 < far
    h, v = lens_fov_deg(JAI_AD130GE.sensor_mm, f)
    assert 40 < h < 50 and 30 < v < 40
    assert pixel_ndvi_noise(0.75, 54, 52) < 0.01                    # one pixel's sensor noise is small


def test_thin_leaves_and_glare_lower_nir_quality():
    cfg = Cfg()
    q_wide, mixed_wide, _ = nir_quality(cfg, 0.8, 0.12, 0.0, 0.012, 0.0, 0.05)
    q_thin, mixed_thin, _ = nir_quality(cfg, 0.8, 0.12, 0.0, 0.004, 0.0, 0.05)
    assert mixed_thin > mixed_wide and q_thin < q_wide
    q_glare, _, glare = nir_quality(cfg, 0.8, 0.05, 0.0, 0.012, 1.0, 0.05)
    assert glare == pytest.approx(cfg.glare_max_share) and q_glare < q_wide
    half_along = nir_footprint(cfg, 0.8)[0] / 2
    assert nir_quality(cfg, 0.8, half_along + 0.05, 0.0, 0.012, 0.0, 0.05)[0] == 0.0   # outside the NIR camera's view


def test_viewpoint_errors_repeat_at_the_same_spot_and_decorrelate_away():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((20000, 2))
    here = viewpoint_error(g[:, 0], g[:, 1], 0.0, 0.15)
    same = viewpoint_error(g[:, 0], g[:, 1], 0.0, 0.15)
    far = viewpoint_error(g[:, 0], g[:, 1], 0.15, 0.15)
    assert np.allclose(here, same) and abs(np.corrcoef(here, far)[0, 1]) < 0.05 and np.std(here) == pytest.approx(1, abs=0.03)


@pytest.mark.parametrize("height_cm, leaf_mm", [(20, 5), (50, 10)])
def test_one_shot_errors_are_plausible(height_cm, leaf_mm):
    cfg = replace(Cfg(), height_mean_cm=height_cm, leaf_width_mm=leaf_mm)
    st = shot_statistics(cfg, np.random.default_rng(1), height_cm, leaf_mm, n=600)
    for mode in ("quick", "careful"):
        assert 0.3 < st[mode]["height_sd"] < 4.0 and abs(st[mode]["height_bias"]) < 2.0     # cm
        # bias: thin seedling leaves mix with soil in the wide lens's coarser pixels (~-0.055 at 1.5 m, 2.8 mm)
        assert 0.005 < st[mode]["ndvi_sd"] < 0.08 and abs(st[mode]["ndvi_bias"]) < 0.08
    assert st["careful"]["height_sd"] <= st["quick"]["height_sd"] * 1.05                  # careful is not worse


def test_too_close_cameras_lose_the_tall_plants():
    cfg = replace(Cfg(), camera_height_m=1.1, depth_resolution="1280x720", nir_lens_focal_mm=6.0, nir_lens_fnumber=4.0,
                  nir_focus_m=0.7, height_mean_cm=55, leaf_width_mm=11)
    st = shot_statistics(cfg, np.random.default_rng(2), 55, 11, n=600)
    assert st["quick"]["no_depth"] > 0.15          # 1.1 m up at 1280x720: tall April onions are inside Min-Z
