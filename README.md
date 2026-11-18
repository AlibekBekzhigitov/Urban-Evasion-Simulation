# QASH Engine: Quick Actions in Sudden Hazards

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version: v2 Stage 1.5](https://img.shields.io/badge/version-v2%20Stage%201.5-blue.svg)](#stage-15-2d-multi-agent-street-world-differentscenarios)

An agent-based stochastic simulation framework modeling urban pursuit-evasion dynamics under partial observability, spatial bottlenecks, and hidden-type threats in high-density courtyard topologies.

---

## QASH v2, Stage 1: Hidden-Type Pursuit and Evasion

**QASH v2 (Stage 1)** replaces the earlier abstract log-odds random-walk addendum with a physical, continuous-space encounter model in metres and seconds (`dt = 0.1 s`).

### Model Overview
A pedestrian walks along a 120 m street segment with courtyard entrances acting as refuges (`x = Y`). A van stops at `x = Δ` and is a *captor van* with prior probability `π = 0.15` (otherwise benign):

- **Van Timeline:** Both types start in `slowing` ~ `U(2, 4)` s. Benign vans pass with probability `p_pass = 0.35`; otherwise, and always for captor vans, the van enters `parking` ~ `U(1, 3)` s followed by `waiting`.
- **Emergence vs. Departure:** A captor team emerges after `τ ~ U(3, 8)` s of waiting and pursues at `5.0 m/s` for up to `15 s`; a benign van leaves after `U(6, 20)` s.
- **Closed-Form Bayesian Posterior:** Because the phase and phase age `a` are observed, the posterior `P(captor | phase, a)` updates analytically via uniform survival functions (peaking at `0.214` during parking and decaying to `0.000` after `8 s` of waiting without emergence).

### Policy Comparison (Base Case, `dt = 0.1 s`, Seed `20261114`)

Evaluated across a `26 × 17` grid (`Δ ∈ [-40, 60]` m, `Y ∈ [-64, 64]` m, 1,000 episodes per van type per cell):

| Policy | Trigger & Pre-Flight Behaviour | $P(\text{caught} \mid \text{captor})$ | $P(\text{FA} \mid \text{benign})$ | Mean Cost $J$ | Lowest-Cost Share |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `ignore` | Never flees; walks on | `0.981` | `0.000` | `1.471` | `0.0%` |
| `naive` | Walks on; flees at emergence | `0.314` | `0.000` | `0.518` | `28.5%` |
| `wait_deploy` | Holds still if van ahead; flees at emergence | `0.161` | `0.000` | `0.584` | `18.3%` |
| `flee_park` | Holds still; flees when van starts to park | `0.000` | `0.647` | `0.937` | `12.4%` |
| `flee_slow` | Flees at first sign of slowing (= static Bayes rule) | `0.000` | `1.000` | `1.275` | `1.1%` |
| **`optimal_dp`** | **Exact belief-MDP backward induction (`BackwardInduction/`)** | **`0.007`** | **`0.117`** | **`0.257`** | **Dominates** |
| *Tie (`naive = wait_deploy`)* | Van stops behind pedestrian (`Δ < 0`) | — | — | — | `39.6%` |

### Risk Maps & Policy Regions

![QASH v2 Stage 1 Risk Maps](UPD/qash_v2_riskmaps.png)

*(If viewing locally, the high-resolution vector figure is available in `UPD/qash_v2_riskmaps.pdf`.)*

---

## Development Stages & Model Attribution

### Stage 1: Exact Belief-MDP Solver (`BackwardInduction/`, `UPD_V2/`) — *Engineered with Gemini*
An exact single-threaded analytical backward induction solver (`qash_v2_dp.py`, exported via `qash_v2_dp_numbers.tex` and `qash_v2_dp_summary.csv`) solves the reduced belief-MDP over `(x_e, phase, a, Y)` with closed-form flight evaluation:
- **Grid-Mean Expected Cost:** $J = 0.257$ (strictly outperforming all static threshold heuristics).
- **Capture & False-Alarm Rates:** $P(\text{caught} \mid \text{captor}) = 0.007$, $P(\text{FA} \mid \text{benign}) = 0.117$.

### Stage 1.5: 2D Multi-Agent Street World (`DifferentScenarios/different_scenes.py`) — *Engineered with Claude*
Extends the Stage 1 encounter physics into a continuous 2D street environment with a decoupled Pillow GIF renderer:
- **2D Mechanics:** Crowd dilution (`n_ped = 30`), capture quota (`0/5`) with escorting back to the van, multi-stop van relocation across parking slots, departing benign vans, and posterior-based all-clear recovery (`smart_clear`).
- **Scenario Animations (`DifferentScenarios/`):**
  - `scene_A_empty.gif` — baseline empty sidewalk (`n_ped = 0`, `wait_deploy`).
  - `scene_B_crowd.gif` — crowd dilution effect (`n_ped = 30`, `wait_deploy`).
  - `scene_C_flee_park.gif` — early parking trigger in crowd (`n_ped = 30`, `flee_park`).

---

## Reproducibility

### Requirements
```bash
pip install numpy matplotlib pillow
