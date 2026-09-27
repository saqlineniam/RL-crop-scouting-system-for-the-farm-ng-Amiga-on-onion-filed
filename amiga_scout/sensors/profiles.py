"""Datasheet facts about the robot's cameras - each number with the document it comes from (see docs/references.md).

    D455 depth   Intel RealSense D455, active stereo depth (D450 module)
    D455 RGB     Intel RealSense D455 colour camera (used for YOLO)
    NIR camera   JAI AD-130GE, 2-CCD prism camera: colour + NIR through ONE lens (no band-to-band parallax);
                 the camera the BoniRob field robot used top-down with its own lighting (Chebrolu et al. 2017).
                 Chosen as the stand-in until the lab picks its NIR camera: its full specification is public.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoDepthCamera:
    name: str
    baseline_m: float
    fov_hd_deg: tuple            # (horizontal, vertical) for 16:9 depth modes
    fov_vga_deg: tuple           # (horizontal, vertical) for 4:3 depth modes
    min_z_m: dict                # depth resolution -> minimum depth distance
    shutter: str
    range_m: tuple               # the working range the manufacturer states
    accuracy_spec: str
    source: str
    notes: str = ""


@dataclass(frozen=True)
class ColourCamera:
    name: str
    width_px: int
    height_px: int
    fov_deg: tuple
    shutter: str
    source: str


@dataclass(frozen=True)
class PrismCamera:
    name: str
    width_px: int
    height_px: int
    pixel_m: float
    sensor_mm: tuple
    prism_fnumber: float         # the prism is designed for lenses of this aperture or slower
    exposure_s: tuple            # shortest, longest electronic shutter
    frame_rate_hz: float
    snr_db: dict                 # channel -> signal-to-noise ratio
    bands_nm: dict               # channel -> spectral range
    shutter: str
    source: str
    notes: str = ""


D455_DEPTH = StereoDepthCamera(
    name="Intel RealSense D455 - stereo depth",
    baseline_m=0.095,
    fov_hd_deg=(87.0, 58.0),
    fov_vga_deg=(75.0, 62.0),
    min_z_m={"1280x720": 0.52, "848x480": 0.35, "640x480": 0.32, "640x360": 0.26, "480x270": 0.20,
             "424x240": 0.18},
    shutter="global (two OV9282 imagers)",
    range_m=(0.4, 10.0),
    accuracy_spec="Z-accuracy <= 2% and RMS error <= 2% up to 4 m, fill rate >= 99% (80% ROI, HD, lab target)",
    source="Intel RealSense D400 Series Datasheet, doc. 337029-009 (2020): tables 3-42, 4-6, 4-9",
    notes="depth noise model: Keselman et al. 2017 and Intel 'Tuning depth cameras for best performance'",
)

D455_RGB = ColourCamera(
    name="Intel RealSense D455 - RGB (OmniVision OV9782)",
    width_px=1280, height_px=800, fov_deg=(90.0, 65.0), shutter="global",
    source="Intel RealSense D400 Series Datasheet 337029-009 (D455 features, table 3-42); D455 product page",
)

JAI_AD130GE = PrismCamera(
    name="JAI AD-130GE - 2-CCD prism camera, colour + NIR",
    width_px=1296, height_px=966, pixel_m=3.75e-6, sensor_mm=(4.86, 3.63), prism_fnumber=2.0,
    exposure_s=(11.49e-6, 31.761e-3), frame_rate_hz=31.0,
    snr_db={"nir": 54.0, "colour": 52.0},
    bands_nm={"colour (red from the Bayer channel)": (400, 700), "nir": (750, 900)},
    shutter="global (progressive-scan interline CCDs)",
    source="JAI AD-130GE user manual, doc. 1036E-1201: sections 3 and 13",
    notes="colour and NIR share one optical path through a dichroic prism, so the red and NIR images line up "
          "at every distance; used top-down under its own lighting on the BoniRob robot (Chebrolu et al. 2017)",
)


# ---- REAL: the lab's own D455 log --------------------------------------------------------------------
# 57 readings of one pot, camera looking down, 0.42-1.01 m: relative depth-image quality vs camera-to-plant
# DISTANCE. (Earlier versions misused it for moving along the row; it belongs to distance.) What the lab's
# quality number combined should be written down with the next calibration run.
LAB_D455_DIST_M = [0.42, 0.49, 0.533, 0.658, 0.754, 0.860, 1.008]
LAB_D455_QUALITY = (np.array([0.720, 0.758, 0.812, 0.883, 0.917, 0.946, 0.990])
                    * np.array([1.0, 1.0, 1.0, 1.0, 0.909, 0.778, 0.5])
                    * np.array([0.919, 0.919, 0.911, 0.936, 0.929, 0.965, 0.934]))


def _lab_raw(z):
    q = float(np.interp(z, LAB_D455_DIST_M, LAB_D455_QUALITY))
    if z > LAB_D455_DIST_M[-1]:
        q *= math.exp(-(z - LAB_D455_DIST_M[-1]) / 0.5)
    elif z < LAB_D455_DIST_M[0]:
        q *= math.exp(-(LAB_D455_DIST_M[0] - z) / 0.1)
    return q


_LAB_PEAK = max(_lab_raw(z) for z in np.linspace(0.3, 2.0, 341))


def lab_depth_quality(z):
    """The lab log's depth-image quality at camera-to-plant distance z (m), 1 at its best distance."""
    return _lab_raw(z) / _LAB_PEAK


PROFILES = {"depth": D455_DEPTH, "rgb": D455_RGB, "nir": JAI_AD130GE}
