# References for the camera model

Each entry says what it supports in the code. The settings are in `amiga_scout/config.py`, tagged **SPEC** (datasheet), **PAPER** (published measurement), **REAL** (the lab's own data) or **PLACEHOLDER** (still to measure).

## Datasheets and manuals

1. **Intel RealSense D400 Series Product Family Datasheet**, doc. 337029-009, Intel, 2020. [PDF](https://www.realsenseai.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf)
   Supports:
   - D455 baseline 95 mm;
   - depth field of view 87° × 58° (HD) and 75° × 62° (VGA);
   - minimum depth distance per resolution (table 4-6);
   - accuracy and fill-rate spec (table 4-9);
   - global-shutter imagers and RGB sensor (OV9782, 1280 × 800).
2. **JAI AD-130GE User Manual**, doc. 1036E-1201, JAI. [PDF](https://www.1stvision.com/cameras/JAI/dataman/Manual_AD-130GE.pdf) · [product page](https://www.jai.com/products/ad-130-ge/)
   Supports:
   - 2-CCD dichroic prism, so colour and NIR share one optical path;
   - 1296 × 966 pixels of 3.75 µm;
   - F2 prism;
   - exposure 11.49 µs – 31.76 ms, 31 fps;
   - S/N > 54 dB (NIR) and > 52 dB (colour).
3. **Grunnet-Jepsen, Sweetser, Woodfill: "Tuning depth cameras for best performance"**, Intel white paper. [link](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)
   Supports:
   - the depth RMS error formula: distance² × subpixel error ÷ (focal length in pixels × baseline);
   - subpixel error < 0.1 (down to ~0.05) on well-textured targets;
   - better performance in bright light;
   - Min-Z scaling with resolution.

## Depth camera (D455) papers

4. **Keselman, Woodfill, Grunnet-Jepsen, Bhowmik (2017). Intel RealSense Stereoscopic Depth Cameras.** CVPR Workshops (CCD). [arXiv:1705.05548](https://arxiv.org/abs/1705.05548)
   Supports:
   - the stereo error model |Δz| = z²·|Δd| / (f·B);
   - RMS noise of about 0.1 disparity;
   - 7×7 census matching windows, the reason thin leaf tips give no depth (`STEREO_MIN_FEATURE_PX`);
   - minimum distance set by the fixed disparity search.
5. **Servi, Mussi, Profili, Furferi, Volpe, Governi, Buonamici (2021). Metrological Characterization and Comparison of D415, D455, L515 RealSense Devices in the Close Range.** *Sensors* 21(22):7770. [doi:10.3390/s21227770](https://doi.org/10.3390/s21227770)
   Supports:
   - D455 systematic depth error: mean −0.7 mm, SD 4.25 mm over 500–1500 mm (`DEPTH_SYSTEMATIC_MM`);
   - larger errors closer than 500 mm;
   - Min-Z of about 52 cm at 1280×720.
6. **Vit, Shani (2018). Comparing RGB-D Sensors for Close Range Outdoor Agricultural Phenotyping.** *Sensors* 18(12):4413. [doi:10.3390/s18124413](https://doi.org/10.3390/s18124413)
   Supports: Intel RealSense stereo cameras are viable for close-range plant measurement outdoors, at various distances and in various light.
7. **Fan, Sun, Qiu, Zhao. Performance Evaluation and Improvement for RGB-D Cameras on High-Throughput Phenotyping Robots.** *Journal of Field Robotics* (2026). [doi:10.1002/rob.70096](https://doi.org/10.1002/rob.70096). The 2020 preprint is withdrawn: [arXiv:2011.01022](https://arxiv.org/abs/2011.01022).
   Supports:
   - RealSense D435i in-field fill rate of about 90% (`DEPTH_FILL_FIELD`);
   - effective range 0.16–1.4 m;
   - errors that depend on brightness and distance.

## NIR / NDVI imaging papers

8. **Chebrolu, Lottes, Schaefer, Winterhalter, Burgard, Stachniss (2017). Agricultural robot dataset for plant classification, localization and mapping on sugar beet fields.** *International Journal of Robotics Research*. [doi:10.1177/0278364917720510](https://doi.org/10.1177/0278364917720510) · [dataset](https://www.ipb.uni-bonn.de/data/sugarbeets2016/)
   Supports: the JAI AD-130GE (RGB + NIR) used top-down, under its own lighting, on the BoniRob field robot. That precedent is why it was chosen as the stand-in NIR camera.
9. **Stamford, Vialet-Chabrand, Cameron, Lawson (2023). Development of an accurate low cost NDVI imaging system for assessing plant health.** *Plant Methods*. [doi:10.1186/s13007-023-00981-8](https://doi.org/10.1186/s13007-023-00981-8)
   Supports:
   - a dual-camera NDVI system with multi-reference calibration, compared against a spectrometer and a MicaSense RedEdge;
   - calibration and the choice of red band matter (`NDVI_CALIBRATION_SD`).
10. **Zhang, Jin, Wang, Rehman, Gee (2022). Elimination of Leaf Angle Impacts on Plant Reflectance Spectra Using Fusion of Hyperspectral Images and 3D Point Clouds.** *Sensors* 23(1):44. [doi:10.3390/s23010044](https://doi.org/10.3390/s23010044)
    Supports: leaf NDVI changes strongly with leaf angle, and depth data can correct it. This is the viewpoint-dependent NDVI error (`NDVI_VIEW_SD`, `VIEW_CORR_M`).
11. **Huang, Luo, Jin, Wang, Zhang, Liu, Zhang (2018). Improving High-Throughput Phenotyping Using Fusion of Close-Range Hyperspectral Camera and Low-Cost Depth Sensor.** *Sensors* 18(8):2711. [doi:10.3390/s18082711](https://doi.org/10.3390/s18082711)
    Supports: plant geometry and its interaction with illumination severely affect close-range spectral measurements, and fusing a depth sensor helps.
12. **Krafft, Scarboro, Hsieh, Doherty, Balint-Kurti, Kudenov (2024). Mitigating Illumination-, Leaf-, and View-Angle Dependencies in Hyperspectral Imaging Using Polarimetry.** *Plant Phenomics* 6:0157. [doi:10.34133/plantphenomics.0157](https://doi.org/10.34133/plantphenomics.0157)
    Supports: specular glare from leaves at certain illumination and view angles corrupts vegetation indices, and polarisation can remove it (`GLARE_*`).

## Crop and data

13. **Lincoln crop vs weed discrimination dataset** (onion and carrot beds, RGB + NIR cameras 5 cm apart on a cart, about 1 m above the beds). [page with citation](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/crop-vs-weed-discrimination-dataset/)
    Supports: two separate cameras need image alignment (parallax), which is why a prism camera is preferred.
14. **Onion leaf morphology**: leaves are hollow and cylindrical, 4–10 per plant, 4–20 mm wide ([HerbiGuide: onion](https://www.herbiguide.com.au/Descriptions/hg_Onion.htm)).
    Supports: the range of `LEAF_WIDTH_MM`. The per-date widths are PLACEHOLDERs; measure them.

## Decision method (for completeness)

15. **Khosravi et al. (2025)**, hierarchical action masking for PPO (HAM-PPO). [arXiv:2503.17985](https://arxiv.org/abs/2503.17985)
    Supports: the two-level action tree with masks (`amiga_scout/policies/ham.py`).

## Training and validating the policy

| Source | What it supports here |
|---|---|
| Ng & Jordan 2000, *PEGASUS: a policy search method for large MDPs and POMDPs*, UAI ([arXiv:1301.3878](https://arxiv.org/abs/1301.3878)) | Comparing policies on fixed random numbers (the simulator made deterministic): the common random numbers of `fork` / `MeasureMixin._rng` |
| Schulman et al. 2015, *Trust Region Policy Optimization*, ICML ([arXiv:1502.05477](https://arxiv.org/abs/1502.05477)) | The "vine": several actions tried from the same state with common random numbers, to compare them with low noise |
| Bertsekas 2020, *Rollout, Policy Iteration, and Distributed Reinforcement Learning*, Athena Scientific ([book](http://www.athenasc.com/rolloutbook_athena.html)) | Rollout: improving on a base policy by simulating it; the improved policy is at least as good as the base |
| Lagoudakis & Parr 2003, *Reinforcement Learning as Classification: Leveraging Modern Classifiers*, ICML ([pdf](https://users.cs.duke.edu/~parr/icml03.pdf)) | Approximate policy iteration with rollouts, a learned model generalising the simulated choices |
| Ross, Gordon & Bagnell 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS | Keeping the data of earlier rounds and adding the states the newer policy visits |
| Andrychowicz et al. 2021, *What Matters in On-Policy RL? A Large-Scale Empirical Study*, ICLR ([arXiv:2006.05990](https://arxiv.org/abs/2006.05990)) | PPO settings (discount, GAE, batch); why PPO needs a usable value estimate |
| Khosravi et al. 2025, HAM-PPO ([arXiv:2503.17985](https://arxiv.org/abs/2503.17985)) | The earlier method (`train --method ppo`); their episodes are 230-290 decisions with immediate economic rewards, ours 1,000-5,000 with a map-accuracy reward |
| Ng, Harada & Russell 1999, *Policy invariance under reward transformations*, ICML | Why spreading the zapper's time over the work that causes it keeps the optimum (a potential-based change of the reward) |
| Roberts et al. 2017, *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 40:913-929 | The spatial train / test split with a buffer between the regions |
| Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice*, NeurIPS ([arXiv:2108.13264](https://arxiv.org/abs/2108.13264)) | Reporting results over many conditions with confidence intervals (for the paper) |
| Rückin, Jin & Popović 2022, *Adaptive Informative Path Planning Using Deep RL for UAV-based Active Sensing*, ICRA | Related work: RL for active sensing of fields, with simulation-based planning |
