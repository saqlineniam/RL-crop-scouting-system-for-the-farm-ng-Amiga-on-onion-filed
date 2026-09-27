"""
amiga_scout.drone - Turn the Vidalia onion drone survey (D:/Crops/Onion) into what the ROBOT would see
========================================================================================================

THE DATA (2024, two commercial Vidalia onion fields near Vidalia, GA, DJI Mavic 3M at 33 m)
    Field 1 ("Onions")  30.0 acres     Field 2 ("Ashley") 23.7 acres
    6 flights each: 01 Jan 31, 02 Feb 14, 03 Mar 7, 04 Mar 18, 05 Apr 4, 06 Apr 17
    Orthomosaics: 4 bands (Green, Red, RedEdge, NIR), ~1.6 cm/px, UTM 17N, reflectance
    DEMs (Metashape, ~0.4-0.7 m/px) for 11 of the 12 flights
    Ground truth: 50 sample points per field (5 x 5 m): bulb count, weight, yield, diameters, size grade

DRONE -> ROBOT (what changes, and why)
    - The robot drives ALONG THE BEDS and measures plants on the bed, not the furrow. So the maps are
      sampled on a bed-aligned grid: one cell per REAL bed and per 0.508 m along it (= one of the 6 plant
      spots of a 10-ft block). The beds are found in the images (`geometry`): they are straight along the
      whole field but not evenly spaced across it (a bed shaper lays a few beds per tractor pass: 1.6-2.2 m
      between neighbours, ~1.92 m typical on Field 1, ~1.82 m on Field 2), so every furrow is located and a
      bed is the strip between two furrows. Each flight is registered to the Apr 4 flight first (the
      orthomosaics of different dates are offset by a few decimetres).
    - Early flights (Jan 31) show only tiny plants at 1.6 cm/px: their leaf NDVI hardly varies (SD 0.008 vs
      0.03 in April), so they carry plant-SIZE patterns rather than NDVI patterns.
    - The robot's camera sees LEAVES (YOLO finds the plant), not soil. So a spot's NDVI is the mean NDVI
      of the plant pixels only (soil/plant split by Otsu's threshold per flight); the drone's all-pixel
      NDVI mixes in the sandy soil and mostly measures how much ground is covered.
    - Plant size: the share of the bed covered by leaves (cover). At the lab's 100 sample points (their
      plant-pixel values, onions_dataset_all.xlsx), canopy area is the most consistent yield predictor
      (r 0.2-0.6 from February on, both fields); plant NDVI predicts well early in Field 2 (r 0.6) but
      weakly later and in Field 1 - so both height (size) and NDVI are worth measuring (see `analyse`).
    - Individual onions are not separable from 33 m (a 0.5 m spot of bed holds ~20 plants), so a cell is
      the local mean; the simulator adds each plant's own deviation on top.
    - Height: the DEMs are too coarse (0.4-0.7 m px) for onion leaves - their "canopy height" is 5-12 cm
      where real onions are 20-55 cm, i.e. mostly raised-bed relief. So height is NOT taken from the DEM:
      the simulator uses a typical height per growth stage (PLACEHOLDER, measure it) scaled by the spot's
      canopy size (height ~ sqrt(cover), plant allometry). `analyse` still reports the DEM check.

    python -m amiga_scout geometry             # bed direction and every furrow, per field (~30 s)
    python -m amiga_scout extract --field f1 --date 05     # one flight (~1.5-2 min)
    python -m amiga_scout extract-all          # all 12 flights (~2 min each)
    python -m amiga_scout analyse [--dem]      # statistics, yield check, DEM check, figures

MODULES: io.py (survey on disk, pixel helpers), beds.py (finding the beds, registering flights),
         extract.py (orthomosaic -> robot-view map), analysis.py (statistics and checks)
"""
