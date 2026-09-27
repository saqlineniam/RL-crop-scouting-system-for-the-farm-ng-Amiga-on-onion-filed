---
title: "Learning Where to Measure: RL crop scouting for the farm-ng Amiga on onion fields"
description: "Reinforcement learning for adaptive robotic crop scouting: a farm-ng Amiga robot learns where to measure plant height and NDVI in drone-mapped onion fields."
---

# Learning Where to Measure in Onion Fields with a farm-ng Amiga

**Reinforcement learning for adaptive robotic crop scouting** in precision agriculture. A **farm-ng Amiga** agricultural robot scouts **onion** fields, measuring plant height with an **Intel RealSense D455** depth camera and plant health (**NDVI**) with a red + NIR camera. It follows a fixed sampling protocol: at least one plant in every 10-ft block.

A policy trained in simulation decides **how many plants to measure where**, trading map accuracy against robot time. The simulator is built on **real drone maps** (DJI Mavic 3M, six flights in 2024) of two commercial onion fields.

## Key results
- **Small fields (5–6 acres):** ties the best hand-written rule, which is to measure every plant.
- **Big fields (~30 acres, held out):** beats every rule by +1.1 to +2.4 points, using 8–9 h against 13.6 h for measuring every plant.
- **Safety:** more than 1,300 test missions with no stranded robot, no illegal actions and no unsampled blocks.

## Method
- **Why not PPO:** on-policy PPO (HAM-PPO) could not learn this, because single decisions are 10–500× smaller than the mission noise.
- **What works instead:** simulation-based policy iteration with common random numbers. Every choice is tried in a copy of the mission with the same measurement noise.
- **The policy:** an ensemble of networks learns the value of each choice, blended by field size.

## Links
- Code, full results and documentation: [GitHub repository](https://github.com/saqlineniam/RL-crop-scouting-system-for-the-farm-ng-Amiga-on-onion-filed)
- [Design summary](PROJECT_SUMMARY.md)
- [References](references.md)

**Keywords:** reinforcement learning, agricultural robotics, crop scouting, precision agriculture, farm-ng Amiga, onion, NDVI, plant phenotyping, RealSense D455, drone mapping, active sensing, policy iteration, simulation.

Contact: Saklain Niam (saqlineniam@gmail.com).
