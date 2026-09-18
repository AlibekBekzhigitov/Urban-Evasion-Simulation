# Agent-Based Simulation of Urban Patrol Evasion Dynamics

A 2D agent-based stochastic simulation engine (built with Python, Pygame, and Pillow) modeling urban pursuit-evasion dynamics between an evasion agent and an operational patrol unit in a high-density courtyard topology ("Khrushchyovka" arch geometry).

![Simulation Preview](simulation.gif)

---

## Acknowledgments & Theoretical Attribution

> **Note & Citation:**
> The core conceptual framework and theoretical architecture of this project are inspired by and derived from the original repository:
> **[Qoyanshyq Engine by Alibek Bekzhigitov](https://github.com/AlibekBekzhigitov/Qoyanshyq)**

---

## Key Theoretical Features

* **POMDP Threat Perception:** Dynamic risk assessment with false positive noise (civilian minivans vs. operational patrol units).
* **Markov Multi-Stop Patrol Logic:** Non-deterministic patrol stop strategy based on population density and KPI quotas.
* **Micro-Local Intercept Kinematics:** Spatial capture triggers bound strictly to parking slot geometry.
* **Courtyard Arch Topology:** Geometric line-of-sight occlusion and sprint evasion mechanics.

---

## Repository Structure

```text
.
├── simulation.py        # Core Pygame simulation engine
├── simulation.gif       # Recorded execution preview
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
