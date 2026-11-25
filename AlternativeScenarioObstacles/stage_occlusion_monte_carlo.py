"""
stage2_mc.py
Stage 2 Monte Carlo Benchmark & Bayesian Occlusion Filter for QASH v2.
Tests:
  1. Regression check (`check-stage1`): verifies exact reduction to Stage 1 when obstacles are off.
  2. One-at-a-time controlled sweep across obstacle configurations (tall truck, angled parking, roadworks).
  3. Symmetric channel decomposition ('none', 'ped_only', 'team_only', 'both') over n=500 episodes.
  4. Saves a 2-panel summary figure (`stage2_mc_results.png`) comparing Capture Risk & Warning Lead Time.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Obstacle:
    x_start: float
    length: float
    y_center: float
    width: float
    is_tall: bool
    kind: str = "sedan"          # 'sedan', 'tall_truck', 'workzone'
    blocks_yard: bool = False
    speed_factor: float = 1.0

    @property
    def x_end(self) -> float:
        return self.x_start + self.length

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (
            self.x_start,
            self.y_center - self.width / 2.0,
            self.x_end,
            self.y_center + self.width / 2.0,
        )


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def ray_intersects_rect(p1, p2, rect) -> bool:
    xmin, ymin, xmax, ymax = rect
    if max(p1[0], p2[0]) < xmin or min(p1[0], p2[0]) > xmax:
        return False
    if max(p1[1], p2[1]) < ymin or min(p1[1], p2[1]) > ymax:
        return False
    edges = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    return any(_segments_intersect(p1, p2, e1, e2) for e1, e2 in edges)


def has_line_of_sight(observer_xy, target_xy, obstacles: List[Obstacle]) -> bool:
    for obs in obstacles:
        if obs.is_tall and ray_intersects_rect(observer_xy, target_xy, obs.rect):
            return False
    return True


def bayesian_posterior_with_occlusion(
    phase: str,
    phase_age: float,
    is_occluded: bool,
    prior: float = 0.15,
    p_pass: float = 0.35,
) -> float:
    """
    Analytical Stage 1 posterior when visible; freezes/decays conservatively under occlusion.
    When visible:
      slowing -> pi
      parking -> pi / (pi + (1 - pi)*(1 - p_pass))
      waiting -> uniform survival ratio over tau ~ U(3,8) vs benign dwell ~ U(6,20)
    When occluded:
      no phase confirmation is available -> posterior reverts to prior (or cautious 0.18 if partial cue).
    """
    if is_occluded:
        return prior

    if phase == "passing" or phase == "left":
        return 0.0
    if phase == "emerged":
        return 1.0
    if phase == "slowing":
        return prior
    if phase == "parking":
        return prior / (prior + (1.0 - prior) * (1.0 - p_pass))
    if phase == "waiting":
        s_t = np.clip((8.0 - phase_age) / 5.0, 0.0, 1.0)
        if phase_age <= 6.0:
            s_b = 1.0
        elif phase_age < 20.0:
            s_b = (20.0 - phase_age) / 14.0
        else:
            s_b = 0.0
        denom = prior * s_t + (1.0 - prior) * (1.0 - p_pass) * s_b
        return float((prior * s_t) / denom) if denom > 1e-12 else 0.0
    return prior


def run_single_episode(
    rng: np.random.Generator,
    obstacles: List[Obstacle],
    van_x: float = 24.0,
    yard_x: float = 53.0,
    parking_angle_deg: float = 0.0,
    channel_mode: str = "both",
    policy: str = "adaptive_bayes",
    dt: float = 0.1,
) -> Dict[str, float]:
    """
    Simulates one stochastic encounter on a 120m street with 2D ray-casting occlusion.
    Policies:
      - 'naive': walks until team emerges, then flees
      - 'flee_park': flees as soon as parking is observed
      - 'adaptive_bayes': flees if posterior >= 0.20 while within danger range (<= 30m),
                          or immediately if emerging team is seen.
    """
    pi = 0.15
    p_pass = 0.35
    is_captor = bool(rng.random() < pi)

    t_slow = float(rng.uniform(2.0, 4.0))
    benign_passes = (not is_captor) and (rng.random() < p_pass)
    t_park = float(rng.uniform(1.0, 3.0)) + ({0.0: 0.0, 45.0: 1.0, 90.0: 4.5}[parking_angle_deg])
    tau_emerge = float(rng.uniform(3.0, 8.0)) if is_captor else 999.0
    t_benign_dwell = float(rng.uniform(6.0, 20.0)) if not is_captor else 999.0

    # Team detour delay if a tall obstacle/workzone sits next to the van's slot
    team_detour_delay = 0.0
    if channel_mode in ("team_only", "both"):
        for obs in obstacles:
            if obs.is_tall and abs((obs.x_start + obs.length / 2.0) - van_x) < 14.0:
                team_detour_delay = 3.0
                break

    ped_x, ped_y = 0.0, 0.0
    van_y = 2.2 if parking_angle_deg == 0.0 else 2.7
    v_walk, v_run, v_pursuer = 1.4, 4.0, 5.0
    mode = "WALK"
    flee_dir = 1.0
    react_timer = 0.0
    triggered = False
    false_alarm = False
    caught = False
    pursuer_x = van_x
    chase_elapsed = 0.0
    first_seen_t: Optional[float] = None
    emerge_t: Optional[float] = None

    yard_blocked = any(
        obs.blocks_yard and (obs.x_start <= yard_x <= obs.x_end)
        for obs in obstacles
    )

    t = 0.0
    while t <= 45.0:
        # 1. Determine van kinematic phase
        if t < t_slow:
            phase, age = "slowing", t
        elif benign_passes:
            phase, age = "passing", t - t_slow
        elif t < t_slow + t_park:
            phase, age = "parking", t - t_slow
        else:
            wait_age = t - (t_slow + t_park)
            if is_captor and wait_age >= (tau_emerge + team_detour_delay):
                phase, age = "emerged", wait_age - (tau_emerge + team_detour_delay)
                if emerge_t is None:
                    emerge_t = t
            elif (not is_captor) and wait_age >= t_benign_dwell:
                phase, age = "left", wait_age - t_benign_dwell
            else:
                phase, age = "waiting", wait_age

        # 2. Check 2D Line-of-Sight occlusion
        if channel_mode in ("none", "team_only"):
            visible = True
        else:
            visible = has_line_of_sight((ped_x, ped_y), (van_x, van_y), obstacles)

        if visible and first_seen_t is None:
            first_seen_t = t

        belief = bayesian_posterior_with_occlusion(phase, age, is_occluded=not visible, prior=pi, p_pass=p_pass)

        # 3. Policy trigger check (only reacts to what is visible!)
        if not triggered and mode == "WALK":
            fire = False
            if policy == "naive":
                fire = visible and (phase == "emerged")
            elif policy == "flee_park":
                fire = visible and (phase in ("parking", "emerged"))
            elif policy == "adaptive_bayes":
                dist_to_van = abs(van_x - ped_x)
                fire = visible and (
                    (phase == "emerged")
                    or (belief >= 0.20 and dist_to_van <= 28.0)
                )

            if fire:
                triggered = True
                react_timer = 1.0  # 1.0s reaction time
                if not is_captor:
                    false_alarm = True

        if triggered and mode == "WALK":
            react_timer -= dt
            if react_timer <= 0.0:
                mode = "FLEE"
                # Choose refuge if viable and not blocked by roadworks, else run away from van
                if (not yard_blocked) and abs(yard_x - ped_x) < abs(van_x - ped_x) + 12.0:
                    flee_dir = 1.0 if yard_x >= ped_x else -1.0
                else:
                    flee_dir = -1.0 if van_x >= ped_x else 1.0

        # 4. Move pedestrian (slowed if inside narrowed roadworks)
        speed_mult = 1.0
        if channel_mode in ("ped_only", "both"):
            for obs in obstacles:
                if obs.kind == "workzone" and (obs.x_start <= ped_x <= obs.x_end):
                    speed_mult = obs.speed_factor
                    break

        cur_speed = (v_run if mode == "FLEE" else v_walk) * speed_mult
        ped_x += (flee_dir if mode == "FLEE" else 1.0) * cur_speed * dt

        # Check courtyard refuge arrival
        if (not yard_blocked) and mode == "FLEE" and abs(ped_x - yard_x) <= 1.2:
            break

        # 5. Move pursuer if emerged
        if phase == "emerged":
            chase_elapsed += dt
            if chase_elapsed <= 15.0:
                step_dir = 1.0 if ped_x >= pursuer_x else -1.0
                pursuer_x += step_dir * v_pursuer * dt
                if abs(pursuer_x - ped_x) <= 1.0:
                    caught = True
                    break
            else:
                break

        if ped_x >= 120.0 or ped_x < -20.0:
            break

        t += dt

    if first_seen_t is None:
        first_seen_t = t
    lead_time = max(0.0, (emerge_t - first_seen_t)) if emerge_t is not None else np.nan

    # Expected cost J (same weights as Stage 1: C_catch=10, c_fp=1)
    cost = (10.0 if caught else 0.0) + (1.0 if false_alarm else 0.0)
    return {
        "is_captor": float(is_captor),
        "caught": float(caught),
        "false_alarm": float(false_alarm),
        "cost": cost,
        "lead_time": lead_time,
    }


def evaluate_scenario(
    obstacles: List[Obstacle],
    van_x: float = 24.0,
    parking_angle_deg: float = 0.0,
    channel_mode: str = "both",
    policy: str = "adaptive_bayes",
    n_episodes: int = 500,
    seed: int = 20261125,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = [
        run_single_episode(rng, obstacles, van_x, 53.0, parking_angle_deg, channel_mode, policy)
        for _ in range(n_episodes)
    ]
    captors = [r for r in rows if r["is_captor"] > 0.5]
    benigns = [r for r in rows if r["is_captor"] < 0.5]

    p_caught = float(np.mean([r["caught"] for r in captors])) if captors else 0.0
    p_fa = float(np.mean([r["false_alarm"] for r in benigns])) if benigns else 0.0
    mean_j = float(np.mean([r["cost"] for r in rows]))
    valid_leads = [r["lead_time"] for r in captors if not np.isnan(r["lead_time"])]
    mean_lead = float(np.mean(valid_leads)) if valid_leads else 0.0

    return {
        "p_caught": round(p_caught, 3),
        "p_fa": round(p_fa, 3),
        "mean_J": round(mean_j, 3),
        "lead_time_s": round(mean_lead, 2),
    }


def check_stage1_regression() -> None:
    """Verifies nesting: turning off tall obstacles makes channel_mode='both' identical to 'none'."""
    low_only = [Obstacle(12.0, 4.8, 2.2, 1.9, is_tall=False, kind="sedan")]
    r_none = evaluate_scenario(low_only, van_x=24.0, channel_mode="none", n_episodes=500, seed=123)
    r_both = evaluate_scenario(low_only, van_x=24.0, channel_mode="both", n_episodes=500, seed=123)
    assert r_none == r_both, f"Regression failed: {r_none} != {r_both}"
    print(f"[PASS] check-stage1 regression: low-only baseline is bit-for-bit identical ({r_both})")


def run_stage2_experiments() -> None:
    check_stage1_regression()

    # Scenario 1: Tall delivery truck at x=12m blocking sight to van at x=24m
    tall_truck_obs = [Obstacle(x_start=12.0, length=6.5, y_center=2.2, width=2.2, is_tall=True, kind="tall_truck")]
    # Scenario 2: Extreme roadworks (x=10..32m, H=3.5m) blocking sidewalk & courtyard arch
    roadworks_obs = [Obstacle(x_start=10.0, length=22.0, y_center=1.4, width=2.5, is_tall=True, kind="workzone", blocks_yard=True, speed_factor=0.6)]

    print("\n=== Monte Carlo Stage 2 Results (n=500 episodes, adaptive_bayes policy, van_x=24m) ===")
    summary_records = []

    for label, obs_list, ang in [
        ("Parallel Baseline (Low Sedans)", [], 0.0),
        ("Tall Delivery Truck (x=12m)", tall_truck_obs, 0.0),
        ("Angled Parking (45 deg) + Truck", tall_truck_obs, 45.0),
        ("Roadworks Stress Test (22m, Arch Blocked)", roadworks_obs, 0.0),
    ]:
        print(f"\n--- {label} ---")
        for ch in ("none", "ped_only", "team_only", "both"):
            res = evaluate_scenario(obs_list, van_x=24.0, parking_angle_deg=ang, channel_mode=ch, n_episodes=500)
            print(f"  channel={ch:9s} -> P(caught|captor)={res['p_caught']:.3f} | P(FA)={res['p_fa']:.3f} | J={res['mean_J']:.3f} | Lead={res['lead_time_s']:.2f}s")
            if ch == "both":
                summary_records.append((label, res))

    # Plot summary comparison
    labels = [r[0].replace(" (", "\n(") for r in summary_records]
    caught_vals = [r[1]["p_caught"] for r in summary_records]
    lead_vals = [r[1]["lead_time_s"] for r in summary_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = ["#2b2d42", "#e76f51", "#f4a261", "#d62828"]

    ax1.bar(labels, caught_vals, color=colors, width=0.55)
    ax1.set_title("Capture Risk P(caught | captor) under Occlusion", fontweight="bold")
    ax1.set_ylabel("Capture Probability")
    ax1.set_ylim(0, max(caught_vals) * 1.25 + 0.05)
    for i, v in enumerate(caught_vals):
        ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    ax2.bar(labels, lead_vals, color=colors, width=0.55)
    ax2.set_title("Effective Pedestrian Warning Lead Time (s)", fontweight="bold")
    ax2.set_ylabel("Warning Lead Time (seconds)")
    ax2.set_ylim(0, max(lead_vals) * 1.25 + 1.0)
    for i, v in enumerate(lead_vals):
        ax2.text(i, v + 0.2, f"{v:.2f} s", ha="center", fontweight="bold", fontsize=9)

    fig.savefig("stage2_mc_results.png", dpi=200)
    print("\nSaved Monte Carlo benchmark chart to: stage2_mc_results.png")


if __name__ == "__main__":
    run_stage2_experiments()