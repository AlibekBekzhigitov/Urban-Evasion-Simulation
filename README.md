# QASH Engine: Quick Actions in Sudden Hazards

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Simulation: Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-N%3D50k-green.svg)](#empirical-validation--monte-carlo-results)

An agent-based stochastic simulation engine modeling non-cooperative urban pursuit-evasion dynamics under partial observability, spatial bottlenecks, and asymmetric threat perception in high-density courtyard topologies ("Khrushchyovka" arch geometries).

---

## Overview & Core Problem

Urban evasion dynamics in restricted pedestrian domains operate under severe spatial constraints and partial information. Operational enforcement units mask transit movement within ordinary civilian traffic streams, deploying unexpectedly to execute local sweeps. 

The **QASH Engine** formalizes how an evader agent evaluates kinematic anomalies under cognitive noise, balancing the trade-off between **False Negatives** ($P_{\text{FN}}$, missed detection resulting in capture) and **False Positives** ($P_{\text{FP}}$, unnecessary flight/panic).

![Simulation Preview](simulation.gif)

---

## Mathematical & Theoretical Architecture

### 1. POMDP Formalization & Wald's SPRT Policy
The decision problem is modeled as a Partially Observable Markov Decision Process (POMDP). The evader does not directly observe the vehicle type $\theta \in \{\text{Threat}, \text{Benign}\}$, but updates a recursive log-odds belief $\ell_t$ via **Wald's Sequential Probability Ratio Test (SPRT)**:

$$\ell_t = \ell_0 + \sum_{i=1}^{t} \ln \Lambda(o_i), \quad \text{where } \Lambda(o) = \frac{P(o \mid \text{Threat})}{P(o \mid \text{Benign})}$$

The evader initiates an immediate sprint toward the escape archway $\Omega_{\text{arch}}$ once belief crosses an effective threshold $h^*$:

$$\text{Action}(t) = \begin{cases} \text{Sprint}(\Omega_{\text{arch}}), & \text{if } \ell_t \ge h^* \\ \text{Observe / Walk}, & \text{otherwise} \end{cases}$$

### 2. Analytical Decision Regimes & Spatial Deadlines
Under micro-local capture constraints ($\Delta X_{\text{slot}} \le 55\text{ px}$), the optimal threshold $h^*$ is bounded by the spatial deadline $D$ (steps remaining until entering the capture zone):

1. **Unconstrained Regime ($D \ge 50$):** High threshold ($h^* \approx 4.50\text{ nats}$), zero false alarms ($P_{\text{FP}} < 1\%$).
2. **Deadline-Constrained Regime ($D \in [5, 20]$):** Optimal threshold decays as $D \to 0$. The evader accepts higher false-alarm rates to prevent catastrophic missed detections ($P_{\text{FN}}$).
3. **Immediate Flight Regime ($D < 5$):** Threshold collapses ($h^* \to 0$). Information gather is impossible; immediate evasion is triggered.

---

## Empirical Validation & Monte Carlo Results

The repository includes a rigorous verification harness (`QashMonteCarlo.py`) executing **$N = 50,000$ seeded Monte Carlo rollouts** per deadline configuration.

### Optimal Threshold & Cost Summary

| Deadline ($D$, steps) | Optimal Threshold ($h^*$) | $P_{\text{FN}}$ (Missed / Captured) | $P_{\text{FP}}$ (False Panic) | Expected Cost $J(h^*)$ |
| :---: | :---: | :---: | :---: | :---: |
| **$D = 50$** (Far / Safe) | **$4.50$ nats** | $0.59\%$ | $0.94\%$ | **$0.0168$** |
| **$D = 20$** | **$2.10$ nats** | $5.26\%$ | $8.05\%$ | **$0.1473$** |
| **$D = 10$** | **$1.15$ nats** | $11.25\%$ | $18.67\%$ | **$0.3274$** |
| **$D = 5$** (Critical) | **$0.50$ nats** | $14.94\%$ | **$32.77\%$** | **$0.5027$** |

### Generated Visual Artifacts

| SPRT Trade-Off Curve | Cost Optimization $J(h)$ |
| :---: | :---: |
| ![FN/FP Trade-off](qash_fn_fp_curve.pdf) | ![Cost vs Threshold](qash_cost_vs_h.pdf) |
| *Convergence between Theory (Inverse Gaussian) and Monte Carlo.* | *Global minima showing optimal decision thresholds $h^*$.* |

---

## Repository Structure

```text
.
├── QashMonteCarlo.py     # Reproducible SPRT POMDP Monte Carlo harness (N=50k)
├── qash_results.csv       # Raw simulation output data with SE confidence intervals
├── simulation.py          # Core Pygame 2D interactive environment
├── simulation.gif         # Execution demonstration GIF
├── requirements.txt       # Python dependencies (NumPy, SciPy, Matplotlib, Pygame)
└── README.md              # Project documentation
