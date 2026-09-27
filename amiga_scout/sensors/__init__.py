"""The robot's cameras as physical instruments, built from their datasheets and from published measurements.

    profiles.py   datasheet facts per camera: Intel RealSense D455 (depth + RGB) and the NIR stand-in, the JAI
                  AD-130GE prism camera (colour + NIR through one lens); plus the lab's own D455 log
    optics.py     the formulas: stereo depth noise (z^2 law), minimum distance, pixel size on the plant,
                  focus blur, depth of field, sensor noise of NDVI
    shot.py       one shot of one plant: the quality the robot can compute live, and the height / NDVI errors

Every number is tagged in config.py: SPEC (datasheet), PAPER (published measurement), REAL (the lab's own
data) or PLACEHOLDER (to measure). Sources: docs/references.md. `python -m amiga_scout cameras` shows the
curves for the current settings.
"""
from .optics import (defocus_px, depth_focal_px, depth_gsd, depth_of_field, depth_rms, lens_fov_deg, lens_gsd,
                     min_z, parse_resolution, pixel_ndvi_noise)
from .profiles import D455_DEPTH, D455_RGB, JAI_AD130GE, PROFILES, lab_depth_quality
from .shot import (averaging, depth_footprint, depth_quality, height_shot, ndvi_shot, nir_footprint, nir_quality,
                   p_detect, resolvable_share, rgb_footprint, shot_statistics, view_angle, viewpoint_error,
                   visible_top_share)

__all__ = ["D455_DEPTH", "D455_RGB", "JAI_AD130GE", "PROFILES", "averaging", "defocus_px", "depth_focal_px",
           "depth_footprint", "depth_gsd", "depth_of_field", "depth_quality", "depth_rms", "height_shot", "lab_depth_quality",
           "lens_fov_deg", "lens_gsd", "min_z", "ndvi_shot", "nir_footprint", "nir_quality", "p_detect", "parse_resolution",
           "pixel_ndvi_noise", "resolvable_share", "rgb_footprint", "shot_statistics", "view_angle", "viewpoint_error", "visible_top_share"]
