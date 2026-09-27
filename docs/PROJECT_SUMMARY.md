# Amiga onion-scouting robot: system design and results (summary for discussion)

Status as of 27 Sep 2026. Everything below runs in simulation. Nothing has been tested on the real robot yet. The code is the Python package `amiga_scout` in the repository `SIM/onion_robot`.

---

## 1. Goal

A farm-ng **Amiga** robot scouts Vidalia onion fields. It measures plant **NDVI** (health) and **height** along a fixed route. A learned decision policy (the "RL brain") chooses **how much to measure where**, trading map accuracy against time:

**SCORE = ACCURACY (0–100) − λ × mission hours.** λ (`HOUR_VALUE_POINTS`) is the lab's value of one robot hour, and we use λ = 1.

## 2. The machine (as simulated)

### Platform and operation
| Item | Value | Source |
|---|---|---|
| Robot | farm-ng Amiga | – |
| Speed | top speed 3.6 mph (1.61 m/s), a ceiling; stop-and-go in the bed 0.89 m/s; travel on the headland 1.34 m/s | spec; placeholders |
| Batteries | 2 packs on board, plus 1 spare pair swapped by a person | lab |
| Return rule | at 20% charge the robot drives straight on to the end of the pass, goes home, the packs are swapped, and it returns to the same plant | lab |
| Bug zapper | a 15-min stop after every 30 min of field work | lab |
| Scan light | always on, except during zapper stops (the zapper has its own light) | lab |
| Working day | no limit; only the batteries limit a mission | lab |
| Field knowledge | the field is GPS-mapped beforehand (outline and every bed), so the route is fixed | lab |

### Sampling protocol (hard rules, enforced by action masks)
- The route drives **along the beds**, one pass every 16 beds (~30 m apart), snaking across the field.
- Each pass is cut into **10-ft blocks**, each holding 6 plant spots 0.508 m apart.
- **At least one plant is measured in every 10-ft block.** This can never be skipped. Energy masks guarantee the batteries always cover the protocol for the rest of the route, so the robot is never stranded.
- The field is also divided into **parts**: 50 per 30 acres, for part-level stress flags.

### Sensors and mount
| | Choice | Notes |
|---|---|---|
| Height | Intel RealSense **D455** depth at 1280×720 | Min-Z 0.52 m; noise ∝ z²; thin leaf tips lost; occlusion off-centre; field fill ~90%; systematic error 4.25 mm |
| Plant finding | D455 RGB + YOLO | never misses in simulation |
| NDVI | **JAI AD-130GE** (red + NIR prism camera), a documented stand-in until the lab picks one | 2.8 mm lens, f/2, focused at 1.31 m; pixel size, defocus, motion blur, mixed leaf/soil edge pixels, glare, viewpoint (leaf-angle) error, calibration offset per mission |
| Mount height | **1.5 m**, cameras looking down, wide side across the bed | the lowest height at which every camera sees the whole 1.4-m canopy at the tallest plants (65–70 cm); 1.85 m with a 4 mm lens scores slightly better on January NDVI |

Every camera parameter comes from datasheets or published papers; `docs/references.md` lists them. At 1.5 m, where the robot stops matters very little.

### Decisions the robot makes
1. **At every 10-ft block:** quick or careful, and how many of the 6 plants (1–6).
   - Quick = one frame per plant.
   - Careful = frames averaged for 1 s, and a second shot is allowed.
2. **At every plant:** where to shoot from (−30 … +30 cm in 10-cm moves, previewed while driving in), or DONE.
   - Learning this was tried, and it did not beat the hand-written positioning on held-out missions: the best spot depends on each plant's hidden leaf angles. So the hand-written positioning is used.
   - A plant counts as measured only once it has an NDVI reading.

### The robot's own map
- A block-level **Gaussian process** of NDVI and height, with each plant's own deviation on top.
- **TRUST** widens the map's uncertainty when readings keep surprising it.
- **ANOMALY** loosens a surprising plant from its neighbours.
- The robot never sees the drone map; it builds its own.

## 3. Accuracy (what "precision" means)
Accuracy is a weighted average of four parts. Each part is scored **per block, per plant or per part and then averaged**, so every measurement counts as soon as it is made.

| Part | Weight | How it is scored |
|---|---|---|
| 10-ft block values | 25% | each block's mean NDVI (tolerance 0.08) and height (tolerance 5 cm) |
| Stressed plants found | 30% | misses cost 1, false alarms 0.5, relative to the number of stressed plants (at least 2% of plants) |
| Parts | 25% | half the flag correct ("needs attention" = ≥10% stressed), half the part's mean NDVI (tolerance 0.05) |
| Protocol | 20% | share of blocks sampled; always 100% by design |

An earlier version used whole-field errors and floors at 0. That made early measurements worth nothing, and we changed it.

## 4. Data: real fields
- **Two Vidalia onion fields** (2024), flown by a DJI Mavic 3M at 33 m on **six dates**: Jan 31, Feb 14, Mar 7, Mar 18, Apr 4 and Apr 17.
  - **Field 1:** 30.0 acres, 307 beds about 1.92 m apart.
  - **Field 2:** 23.7 acres, 104 beds about 1.82 m apart.
  - 50 yield sample points per field.
- **Drone → robot:**
  - every furrow was detected, so each bed is the strip between two furrows;
  - every flight is registered to the Apr 4 flight;
  - NDVI comes from plant pixels only, and only relative differences are carried over (`DRONE_NDVI_GAIN` is a placeholder);
  - height = the typical height for the growth stage × √(canopy cover / median), because the drone elevation models are too coarse for onion leaves;
  - each plant gets its own random deviation on top.
- **Field types** (stress added on top of the real maps, so the ground truth is known exactly):

  | Type | What is added |
  |---|---|
  | `as_is` | nothing (the real map) |
  | `patches` | disease or pest foci, 6–30 m across |
  | `spots` | isolated stressed plants |
  | `poorer` | the whole field is poorer than the robot expects |

- **Variety every mission:** the route start, the conditions (speed, power, packs, noise, wind, zapper) and the field type all change.

## 5. Train / test split (spatial 80/20, both fields)
In each field, one compact end holding 20% of the crop is the **test region**. The rest, minus a **30-m buffer**, is the **training region**. The regions are the same on all six flights. Of the field's two ends, the one most typical of the whole field was held out.

| | Field 1 | Field 2 |
|---|---|---|
| Test region | 6.0 acres (a band of beds) | 4.7 acres (the first 128 m of every bed) |
| Training region | 21.5 acres | 17.6 acres |

- **Big fields made of real data.** A training region is laid out 2–3 times side by side, with every other copy mirrored so the beds continue across the seam, and cut into random **25–40-acre pieces**. Laying out a test region the same way gives **held-out ~30-acre test fields** (`f1~test*5`, `f2~test*3x2`). Tiled real data beat synthetic square fields in a direct comparison.
- **Training pool:** the whole training regions, plus 5/10/15-acre pieces of them, plus tiled big pieces (a third of missions). Every flight and every field type is used.
- **Model selection:** training fields with new routes and stress, split into small, medium and big groups.
- **Testing:** test regions only (small), tiled test regions (big), and the whole fields, reported separately because 80% of their area was seen in training.

## 6. How the policy is trained
**Why not PPO.** HAM-PPO (Khosravi et al. 2025) was the starting point. It tied the rules at best.
- Measured on this simulator: one block decision changes a mission score by **0.001–0.02 points**, while the noise PPO sees when comparing decisions is **0.1–0.5 points**.
- PPO's value estimate could not predict returns within a mission.

**The method now: simulation-based policy iteration with rollouts and common random numbers** (Bertsekas' rollout; Lagoudakis & Parr 2003; PEGASUS, Ng & Jordan 2000; the "vine" of TRPO, Schulman et al. 2015):
1. Run training missions. At **40 random block decisions per mission** (the same number for small and big fields), copy the mission and try **every legal choice** (12 of them).
2. Every copy gets the **same measurement noise** (random numbers keyed by plant and shot). This made comparisons 25–650× less noisy.
3. **Value each choice** by the accuracy it adds and the hours it costs over the **next 10 blocks**, with the base rule (minimum) doing those blocks. The protocol plan for the rest of the route is added (so battery swaps count), and the zapper counts as its average share of work. Two independent noise draws are used.
4. **Networks learn these values from the robot's observation** (52 inputs: map state, uncertainty, previous and next blocks, stress seen so far, battery, field size and so on). There are two ensembles of **5 networks** each, bagged over missions:
   - **raw** (accuracy in points): best on small fields;
   - **per-block** (accuracy per block's share of the field): best on big fields, and size-invariant.
5. **The decision:**
   - blend the two ensembles by field size (all raw ≤ 300 blocks ≈ 7 acres, all per-block ≥ 900 blocks ≈ 21 acres);
   - take the best quick choice;
   - switch to careful only if its advantage is positive across the networks (mean − 1 × spread > 0).
   - Accuracy and hours are kept apart, so the same model runs at any λ.
6. **3 rounds of 60,000 valued decisions.** Each round adds the states the newer policy visits and keeps earlier data (DAgger-style). The best round is picked on the selection fields.
7. **Safety** comes from the masks: the protocol always holds, and the robot is never stranded.

Things tested and rejected, with data:
- a longer valuation window (the labels were already unbiased);
- a smooth value shape in plant count;
- a "cautious" choice applied to all actions;
- weighting small fields up;
- learned positioning at 1.5 m.

Things that fixed earlier failures:
- the ensemble (one network's random errors were deciding close choices);
- balanced examples per mission;
- per-item accuracy scoring;
- training on big fields.

## 7. Results (best model `onion_rl3_it3`, λ = 1)
**Test regions (5–6 acres, never trained on), 24 missions per field type.** On small fields, "measure every plant quickly" is the right answer, and the RL ties it:

| | Patches | Spots | Poorer | As is |
|---|---|---|---|---|
| RL vs every plant, quick | −0.0 ± 0.2 | −0.3 ± 0.2 | −0.1 ± 0.4 | −0.1 ± 0.2 |

**Big fields:**

| | RL vs best rule (the minimum rule) | RL hours | Minimum rule hours | Every plant, quick: hours |
|---|---|---|---|---|
| Held-out ~30-acre tiled test fields (`validate`) | **+1.1 ± 0.8** (wins 83% of missions) | 8.3 | 5.5 | 13.6 |
| Same fields, 576 missions (`fieldwise`) | **+2.4 ± 0.4**; better on all 12 flights; wins 41 of 48 flight × type cases | 9.3 | 5.5 | 13.6 |
| Whole real fields (partly seen) | **+2.8 ± 1.8** (wins 92%) | 7.2 | 5.0 | – |

**On the held-out big fields, by field type:**

| Patches | Poorer | As is | Spots |
|---|---|---|---|
| +1.6 ± 0.5 | +1.3 ± 0.6 | +1.2 ± 0.7 | +1.0 ± 1.1 (weakest; −1.3 to −1.5 on Field 1's later flights) |

**What the robot does:**
- **Small fields:** about 5.9 plants per block, quick.
- **Big fields:** about 2 plants per block on average, more where stress or surprise shows, and almost never careful. On big fields it matches "every plant"'s accuracy (74.9 vs 75.6) in 4.3 fewer hours. It gets the part-level flags right more often than either rule.
- **Robustness:** not worse than the best rule under soft soil, aged packs (70%), strong wind, or a longer zapper stop.
- **Safety:** 0 stranded missions, 0 illegal actions, 0 unsampled blocks in more than 1,300 test missions.
- **Verdict:** READY for a supervised, shadow-mode field trial.

**The value of time (the same model at other λ):**

| λ (points per hour) | Small fields | Big fields |
|---|---|---|
| 0.5 | ties "every plant" | **+1.4** |
| 1 | ties "every plant" | **+2.6** |
| 2 | ties "every plant" | ≈ minimum rule (−0.6) |
| 3 | ties "every plant" | ≈ minimum rule (−0.2) |

A higher λ makes the minimum near-optimal on big fields, and the RL simply becomes it. λ should come from the lab's value of robot time; it is not a tuning knob.

## 8. Open issues and next steps
1. **The map overreacts to isolated stressed plants.** One low reading can drag 20–40 unmeasured neighbours below the stress threshold until the next reading, so spots fields are the weakest case. The principled fix is a robust (Student-t) plant model. A quick prototype helped spots but slowed patch detection, so it needs its own study.
2. **Placeholders to measure on the real Amiga:**
   - power draw, speeds, and move and shot times;
   - swap time;
   - the robot-vs-drone NDVI scale (at the 50 sample points);
   - onion height per growth stage with the D455;
   - the final NIR camera choice and its real noise and glare.
3. **Mount:** 1.5 m (default) vs 1.85 m (better January NDVI, but a taller mast and less lamp light).
4. **Scoring questions for the lab:**
   - Should an hour count the same on 5 and 30 acres?
   - Should a pack swap (a person's time) cost more than its minutes?
5. **Test regions are small (5–6 acres).** The big-field test uses tiled copies of them: real data, but repeated texture.
6. **Next milestone:** measure the placeholders, retrain, then run a shadow-mode field trial. In shadow mode the robot follows the protocol rule while the RL's choices are logged and compared.

## 9. How to run (from `SIM/onion_robot`, venv active)
```
python -m amiga_scout train --randomize --set hour_value_points=1 --states 60000 --iterations 3 --model-path models\onion_rl3
python -m amiga_scout curve --model-path models\onion_rl3 --set hour_value_points=1
python -m amiga_scout validate --models models\checkpoints\onion_rl3_it3.pt --set hour_value_points=1
python -m amiga_scout fieldwise --field "f1~test*5 f2~test*3x2" --models models\checkpoints\onion_rl3_it3.pt --set hour_value_points=1
python -m amiga_scout describe --field f1_05~test      # the field, route, batteries, all settings
python -m amiga_scout cameras                          # camera model: limits and per-shot errors
python -m amiga_scout mount                            # which camera height and lens
python -m amiga_scout watch --field f2_05~test --policy models\checkpoints\onion_rl3_it3.pt
```
Details: `README.md`. Sources: `docs/references.md`. All settings: `amiga_scout/config.py`, each tagged SPEC, LAB, DATA, PAPER or PLACEHOLDER.
