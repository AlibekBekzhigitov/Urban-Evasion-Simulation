"""
stage2_occlusion_aware.py (v2 - Corner-Edge Peeking & Corridor-to-Refuge Guard)
Compares blind `adaptive_bayes` vs. refined `occlusion_aware` across Stage 2 scenarios
and the 10x10 ambush sensitivity heatmap.
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
    kind: str = "sedan"
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
    if is_occluded:
        return prior
    if phase in ("passing", "left"):
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
    policy: str = "adaptive_bayes",
    dt: float = 0.1,
) -> Dict[str, float]:
    pi = 0.15
    p_pass = 0.35
    is_captor = bool(rng.random() < pi)

    t_slow = float(rng.uniform(2.0, 4.0))
    benign_passes = (not is_captor) and (rng.random() < p_pass)
    t_park = float(rng.uniform(1.0, 3.0)) + ({0.0: 0.0, 45.0: 1.0, 90.0: 4.5}[parking_angle_deg])
    tau_emerge = float(rng.uniform(3.0, 8.0)) if is_captor else 999.0
    t_benign_dwell = float(rng.uniform(6.0, 20.0)) if not is_captor else 999.0

    team_detour_delay = 0.0
    for obs in obstacles:
        if obs.is_tall and abs((obs.x_start + obs.length / 2.0) - van_x) < 14.0:
            team_detour_delay = 3.0
            break

    planned_emerge_t = (t_slow + t_park + tau_emerge + team_detour_delay) if is_captor else np.nan

    ped_x = 0.0
    # Hug inner building edge (y = -0.8m) to maximize sightline angle past parked trucks
    ped_y = -0.8 if policy == "occlusion_aware" else 0.0
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
    delay_penalty = 0.0

    yard_blocked = any(
        obs.blocks_yard and (obs.x_start <= yard_x <= obs.x_end)
        for obs in obstacles
    )

    t = 0.0
    while t <= 45.0:
        # 1. Van kinematic phase
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
            elif (not is_captor) and wait_age >= t_benign_dwell:
                phase, age = "left", wait_age - t_benign_dwell
            else:
                phase, age = "waiting", wait_age

        # 2. 2D Line-of-Sight check
        visible = has_line_of_sight((ped_x, ped_y), (van_x, van_y), obstacles)
        if visible and first_seen_t is None:
            first_seen_t = t

        belief = bayesian_posterior_with_occlusion(phase, age, is_occluded=not visible, prior=pi, p_pass=p_pass)

        # 3. Policy trigger check
        if not triggered and mode == "WALK":
            dist_to_van = abs(van_x - ped_x)
            if policy == "adaptive_bayes":
                fire = visible and ((phase == "emerged") or (belief >= 0.20 and dist_to_van <= 28.0))
            elif policy == "occlusion_aware":
                # Guard the entire corridor between pedestrian and the courtyard refuge
                blocks_corridor = (ped_x <= van_x <= yard_x + 5.0)
                fire = visible and (
                    (phase == "emerged")
                    or (belief >= 0.20 and (dist_to_van <= 34.0 or blocks_corridor))
                )
            else:
                fire = visible and (phase == "emerged")

            if fire:
                triggered = True
                react_timer = 0.4 if policy == "occlusion_aware" else 1.0
                if not is_captor:
                    false_alarm = True

        if triggered and mode == "WALK":
            react_timer -= dt
            if react_timer <= 0.0:
                mode = "FLEE"
                if (not yard_blocked) and abs(yard_x - ped_x) < abs(van_x - ped_x) + 12.0:
                    flee_dir = 1.0 if yard_x >= ped_x else -1.0
                else:
                    flee_dir = -1.0 if van_x >= ped_x else 1.0

        # 4. Speed control: pass quickly past the blind wall, peek carefully ONLY at the front corner
        caution_mult = 1.0
        if policy == "occlusion_aware" and mode == "WALK":
            for obs in obstacles:
                if obs.is_tall and (obs.x_end - 1.5 <= ped_x <= obs.x_end + 0.5):
                    caution_mult = 0.35
                    delay_penalty += 0.004 * dt
                    break

        workzone_mult = 1.0
        for obs in obstacles:
            if obs.kind == "workzone" and (obs.x_start <= ped_x <= obs.x_end):
                workzone_mult = obs.speed_factor
                break

        cur_speed = (v_run if mode == "FLEE" else v_walk * caution_mult) * workzone_mult
        ped_x += (flee_dir if mode == "FLEE" else 1.0) * cur_speed * dt

        if (not yard_blocked) and mode == "FLEE" and abs(ped_x - yard_x) <= 1.2:
            break

        # 5. Move pursuer
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
        first_seen_t = planned_emerge_t if is_captor else t

    lead_time = max(0.0, planned_emerge_t - first_seen_t) if is_captor else np.nan
    cost = (10.0 if caught else 0.0) + (1.0 if false_alarm else 0.0) + delay_penalty

    return {
        "is_captor": float(is_captor),
        "caught": float(caught),
        "false_alarm": float(false_alarm),
        "cost": cost,
        "lead_time": lead_time,
    }


def evaluate_policy(
    obstacles: List[Obstacle],
    van_x: float = 24.0,
    parking_angle_deg: float = 0.0,
    policy: str = "adaptive_bayes",
    n_episodes: int = 500,
    seed: int = 20261125,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = [
        run_single_episode(rng, obstacles, van_x, 53.0, parking_angle_deg, policy)
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


def compare_policies_and_heatmaps() -> None:
    tall_truck_obs = [Obstacle(x_start=12.0, length=6.5, y_center=2.2, width=2.2, is_tall=True, kind="tall_truck")]
    roadworks_obs = [Obstacle(x_start=10.0, length=22.0, y_center=1.4, width=2.5, is_tall=True, kind="workzone", blocks_yard=True, speed_factor=0.6)]

    print("=== Head-to-Head Comparison: Blind `adaptive_bayes` vs. Refined `occlusion_aware` (n=500, van_x=24m) ===")
    for label, obs_list, ang in [
        ("Parallel Baseline (Low Sedans)", [], 0.0),
        ("Tall Truck (x=12m)", tall_truck_obs, 0.0),
        ("Angled (45 deg) + Tall Truck", tall_truck_obs, 45.0),
        ("Roadworks 22m (Arch Blocked)", roadworks_obs, 0.0),
    ]:
        r_blind = evaluate_policy(obs_list, van_x=24.0, parking_angle_deg=ang, policy="adaptive_bayes", n_episodes=500)
        r_aware = evaluate_policy(obs_list, van_x=24.0, parking_angle_deg=ang, policy="occlusion_aware", n_episodes=500)
        print(f"\n--- {label} ---")
        print(f"  adaptive_bayes  -> P(caught)={r_blind['p_caught']:.3f} | P(FA)={r_blind['p_fa']:.3f} | J={r_blind['mean_J']:.3f} | Lead={r_blind['lead_time_s']:.2f}s")
        print(f"  occlusion_aware -> P(caught)={r_aware['p_caught']:.3f} | P(FA)={r_aware['p_fa']:.3f} | J={r_aware['mean_J']:.3f} | Lead={r_aware['lead_time_s']:.2f}s")

    # Side-by-side 10x10 Heatmaps
    van_xs = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0]
    truck_xs: List[Optional[float]] = [None, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0]

    grid_blind = np.zeros((len(truck_xs), len(van_xs)))
    grid_aware = np.zeros((len(truck_xs), len(van_xs)))

    print("\nComputing side-by-side 10x10 Ambush Heatmaps...")
    for r_idx, tx in enumerate(truck_xs):
        for c_idx, vx in enumerate(van_xs):
            if tx is None:
                obs = []
            else:
                if abs((tx + 3.25) - vx) < 4.5:
                    grid_blind[r_idx, c_idx] = np.nan
                    grid_aware[r_idx, c_idx] = np.nan
                    continue
                obs = [Obstacle(x_start=tx, length=6.5, y_center=2.2, width=2.2, is_tall=True, kind="tall_truck")]

            seed_cell = 20261125 + r_idx * 100 + c_idx
            grid_blind[r_idx, c_idx] = evaluate_policy(obs, van_x=vx, policy="adaptive_bayes", n_episodes=300, seed=seed_cell)["p_caught"]
            grid_aware[r_idx, c_idx] = evaluate_policy(obs, van_x=vx, policy="occlusion_aware", n_episodes=300, seed=seed_cell)["p_caught"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d3d3d3")

    im1 = ax1.imshow(grid_blind, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax1.set_title("Blind `adaptive_bayes`: P(caught | captor)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Captor Van Stop Position van_x (m)")
    ax1.set_ylabel("Tall Truck Position truck_x (m)")
    ax1.set_xticks(range(len(van_xs)))
    ax1.set_xticklabels([f"{int(x)}" for x in van_xs], fontsize=8.5)
    ax1.set_yticks(range(len(truck_xs)))
    ax1.set_yticklabels(["No Truck" if x is None else f"{int(x)}m" for x in truck_xs], fontsize=8.5)
    fig.colorbar(im1, ax=ax1, label="Capture Probability")

    im2 = ax2.imshow(grid_aware, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax2.set_title("Corner-Peek `occlusion_aware`: P(caught | captor)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Captor Van Stop Position van_x (m)")
    ax2.set_ylabel("Tall Truck Position truck_x (m)")
    ax2.set_xticks(range(len(van_xs)))
    ax2.set_xticklabels([f"{int(x)}" for x in van_xs], fontsize=8.5)
    ax2.set_yticks(range(len(truck_xs)))
    ax2.set_yticklabels(["No Truck" if x is None else f"{int(x)}m" for x in truck_xs], fontsize=8.5)
    fig.colorbar(im2, ax=ax2, label="Capture Probability")

    fig.savefig("stage2_caution_comparison.png", dpi=200)
    print("Saved side-by-side comparison heatmap to: stage2_caution_comparison.png")


if __name__ == "__main__":
    compare_policies_and_heatmaps()