"""
stage2_preparked.py
Stage 2 Extension (Roadmap Item 2): Pre-Parked Active Captor Vans Behind Tall Occluders.
Models:
  - Van is already parked prior to evader arrival (`is_pre_parked=True`) with no slowing/parking cues.
  - Active captor vans have open doors / ongoing quota capture (k < Q): once line-of-sight opens,
    uncertainty immediately collapses to P(captor) = 1.0 (zero false alarms on benign parked vans).
  - Compares 3 policies (`naive`, `corner_peek`, `wide_berth_peek`) across truck-to-van clearances
    and saves a 2-panel benchmark figure (`stage2_preparked_results.png`).
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
    kind: str = "tall_truck"
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


def run_preparked_episode(
    rng: np.random.Generator,
    obstacles: List[Obstacle],
    van_x: float = 24.0,
    yard_x: float = 53.0,
    policy: str = "naive",
    dt: float = 0.1,
) -> Dict[str, float]:
    """
    Pre-parked active van scenario:
      - No slowing/parking phases: van is already stationary at `van_x` at t=0.
      - Captor van is actively filling quota (k < Q) for `t_active ~ U(18, 35)` s.
      - Benign parked van sits passively and leaves after `U(8, 25)` s.
      - Once line-of-sight opens (`visible=True`), active escorting is unambiguously identified:
        belief jumps immediately to 1.0 for captor and 0.0 for benign.
    """
    pi = 0.15
    is_captor = bool(rng.random() < pi)
    t_active_window = float(rng.uniform(18.0, 35.0)) if is_captor else 0.0
    t_benign_dwell = float(rng.uniform(8.0, 25.0)) if not is_captor else 999.0

    # Team detour delay if tall truck sits immediately adjacent to the van
    team_detour_delay = 0.0
    for obs in obstacles:
        if obs.is_tall and abs((obs.x_start + obs.length / 2.0) - van_x) < 12.0:
            team_detour_delay = 2.0
            break

    ped_x = 0.0
    # Lateral sidewalk offset:
    # - naive walks in the center of the sidewalk (y = 0.0m)
    # - corner_peek hugs inner lane (y = -0.8m)
    # - wide_berth_peek hugs the building wall (y = -1.4m) to open sightlines earlier
    ped_y = {"naive": 0.0, "corner_peek": -0.8, "wide_berth_peek": -1.4}[policy]
    van_y = 2.2

    v_walk, v_run, v_pursuer = 1.4, 4.0, 5.0
    mode = "WALK"
    flee_dir = 1.0
    react_timer = 0.0
    triggered = False
    false_alarm = False
    caught = False
    pursuer_x = van_x
    pursuer_dispatched = False
    chase_elapsed = 0.0
    first_seen_t: Optional[float] = None
    dispatch_t: Optional[float] = None
    delay_penalty = 0.0

    t = 0.0
    while t <= 45.0:
        # 1. Pre-parked van state
        if is_captor:
            phase = "active_capture" if t <= t_active_window else "left"
        else:
            phase = "benign_parked" if t <= t_benign_dwell else "left"

        # 2. 2D Line-of-Sight check
        visible = has_line_of_sight((ped_x, ped_y), (van_x, van_y), obstacles)
        if visible and first_seen_t is None and phase != "left":
            first_seen_t = t

        # Asymmetric identification: visible active capture collapses belief to 1.0
        if not visible or phase == "left":
            belief = pi if not visible else 0.0
        else:
            belief = 1.0 if phase == "active_capture" else 0.0

        # 3. Evader policy trigger
        if not triggered and mode == "WALK":
            if belief >= 0.99:
                triggered = True
                react_timer = 1.0 if policy == "naive" else 0.4
                if not is_captor:
                    false_alarm = True

        if triggered and mode == "WALK":
            react_timer -= dt
            if react_timer <= 0.0:
                mode = "FLEE"
                # Since the active van blocks the forward path to the courtyard, retreat backward
                flee_dir = -1.0 if van_x >= ped_x else 1.0

        # 4. Pedestrian speed control (corner peeking at front edge of tall truck)
        caution_mult = 1.0
        if policy in ("corner_peek", "wide_berth_peek") and mode == "WALK":
            for obs in obstacles:
                if obs.is_tall and (obs.x_end - 1.5 <= ped_x <= obs.x_end + 0.5):
                    caution_mult = 0.35
                    delay_penalty += 0.004 * dt
                    break

        cur_speed = v_run if mode == "FLEE" else (v_walk * caution_mult)
        ped_x += (flee_dir if mode == "FLEE" else 1.0) * cur_speed * dt

        if mode == "FLEE" and abs(ped_x - yard_x) <= 1.2:
            break

        # 5. Pre-parked team spots evader once evader enters their line of sight within 22m
        if phase == "active_capture" and not pursuer_dispatched:
            if visible and abs(van_x - ped_x) <= 22.0:
                pursuer_dispatched = True
                dispatch_t = t + team_detour_delay

        if pursuer_dispatched and dispatch_t is not None and t >= dispatch_t:
            chase_elapsed += dt
            if chase_elapsed <= 15.0:
                step_dir = 1.0 if ped_x >= pursuer_x else -1.0
                pursuer_x += step_dir * v_pursuer * dt
                if abs(pursuer_x - ped_x) <= 1.0:
                    caught = True
                    break
            else:
                break

        if ped_x >= 120.0 or ped_x < -25.0:
            break

        t += dt

    # Compute sighting distance (how far from van the evader was when first spotting it)
    if first_seen_t is not None:
        # Approximate x at sighting from initial speed or post-hoc gap
        sight_gap_m = max(0.0, van_x - (first_seen_t * v_walk))
    else:
        sight_gap_m = 0.0

    cost = (10.0 if caught else 0.0) + (1.0 if false_alarm else 0.0) + delay_penalty
    return {
        "is_captor": float(is_captor),
        "caught": float(caught),
        "false_alarm": float(false_alarm),
        "cost": cost,
        "sight_gap_m": sight_gap_m,
    }


def evaluate_preparked(
    truck_x: float,
    van_x: float = 24.0,
    policy: str = "naive",
    n_episodes: int = 500,
    seed: int = 20261125,
) -> Dict[str, float]:
    obs = [Obstacle(x_start=truck_x, length=6.5, y_center=2.2, width=2.2, is_tall=True, kind="tall_truck")]
    rng = np.random.default_rng(seed)
    rows = [
        run_preparked_episode(rng, obs, van_x=van_x, policy=policy)
        for _ in range(n_episodes)
    ]
    captors = [r for r in rows if r["is_captor"] > 0.5]
    benigns = [r for r in rows if r["is_captor"] < 0.5]

    p_caught = float(np.mean([r["caught"] for r in captors])) if captors else 0.0
    p_fa = float(np.mean([r["false_alarm"] for r in benigns])) if benigns else 0.0
    mean_j = float(np.mean([r["cost"] for r in rows]))
    mean_gap = float(np.mean([r["sight_gap_m"] for r in captors])) if captors else 0.0

    return {
        "p_caught": round(p_caught, 3),
        "p_fa": round(p_fa, 3),
        "mean_J": round(mean_j, 3),
        "sight_gap_m": round(mean_gap, 2),
    }


def run_preparked_benchmark() -> None:
    # Truck ends at truck_x + 6.5m. Vary clearance between truck front bumper and van at 24m.
    configs = [
        ("Tight Ambush\n(3.5m gap)", 14.0),    # truck: 14.0..20.5m, van at 24.0m
        ("Medium Gap\n(5.5m gap)", 12.0),      # truck: 12.0..18.5m, van at 24.0m
        ("Wide Gap\n(9.5m gap)", 8.0),         # truck: 8.0..14.5m, van at 24.0m
        ("Far Truck\n(13.5m gap)", 4.0),       # truck: 4.0..10.5m, van at 24.0m
    ]
    policies = ["naive", "corner_peek", "wide_berth_peek"]
    results = {p: [] for p in policies}

    print("=== Stage 2 (Item 2): Pre-Parked Active Van Behind Tall Truck (n=500, van_x=24m) ===")
    for label, tx in configs:
        clean_lbl = label.replace("\n", " ")
        print(f"\n--- {clean_lbl} (truck_x={tx}m, van_x=24m) ---")
        for pol in policies:
            res = evaluate_preparked(truck_x=tx, van_x=24.0, policy=pol, n_episodes=500)
            results[pol].append(res)
            print(
                f"  {pol:16s} -> P(caught|captor)={res['p_caught']:.3f} | "
                f"P(FA)={res['p_fa']:.3f} | J={res['mean_J']:.3f} | SightGap={res['sight_gap_m']:.2f}m"
            )

    # Plot 2-panel comparison: Capture Risk & Detection Distance
    x = np.arange(len(configs))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    palette = {"naive": "#d62828", "corner_peek": "#f4a261", "wide_berth_peek": "#2a9d8f"}
    titles = {
        "naive": "Naive (Center Sidewalk, 1.0s react)",
        "corner_peek": "Corner-Peek (y=-0.8m, 0.4s react)",
        "wide_berth_peek": "Wide-Berth Peek (Wall y=-1.4m)",
    }

    for idx, pol in enumerate(policies):
        offset = (idx - 1) * width
        caught_vals = [r["p_caught"] for r in results[pol]]
        gap_vals = [r["sight_gap_m"] for r in results[pol]]

        bars1 = ax1.bar(x + offset, caught_vals, width, label=titles[pol], color=palette[pol])
        for b in bars1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8, fontweight="bold")

        bars2 = ax2.bar(x + offset, gap_vals, width, label=titles[pol], color=palette[pol])
        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.3, f"{h:.1f}m", ha="center", fontsize=8, fontweight="bold")

    ax1.set_title("Pre-Parked Active Van: Capture Risk P(caught | captor)", fontweight="bold", fontsize=10.5)
    ax1.set_ylabel("Capture Probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in configs], fontsize=8.8)
    ax1.set_ylim(0, 1.18)
    ax1.legend(fontsize=8.2, loc="upper right")

    ax2.set_title("First Visual Detection Distance to Van (metres)", fontweight="bold", fontsize=10.5)
    ax2.set_ylabel("Sightline Detection Gap (m)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[0] for c in configs], fontsize=8.8)
    ax2.set_ylim(0, max(r["sight_gap_m"] for r in results["wide_berth_peek"]) * 1.28 + 2.0)

    fig.savefig("stage2_preparked_results.png", dpi=200)
    print("\nSaved pre-parked benchmark chart to: stage2_preparked_results.png")


if __name__ == "__main__":
    run_preparked_benchmark()