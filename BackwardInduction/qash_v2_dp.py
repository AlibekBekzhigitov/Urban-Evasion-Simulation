#!/usr/bin/env python3
"""
QASH v2, Stage 1 -- Single-Thread Analytical Backward Induction Solver (~4s, zero heat).
"""
from __future__ import annotations

import os
# Prevent NumPy/BLAS thread contention on macOS Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import math
import time
from dataclasses import dataclass

import numpy as np

POLICIES = ("ignore", "naive", "wait_deploy", "flee_park", "flee_slow")


@dataclass(frozen=True)
class Params:
    dt: float = 0.1
    t_max: float = 45.0
    street_len: float = 120.0
    pi: float = 0.15
    p_pass: float = 0.35
    slow: tuple = (2.0, 4.0)
    park: tuple = (1.0, 3.0)
    tau: tuple = (3.0, 8.0)
    leave: tuple = (6.0, 20.0)
    v_walk: float = 1.4
    v_run: float = 4.0
    v_catch: float = 5.0
    t_react: float = 1.0
    contact: float = 1.0
    t_chase: float = 15.0
    c_catch: float = 10.0
    c_fp: float = 1.0
    c_wait: float = 0.02

    @property
    def step_wait_cost(self) -> float:
        return self.c_wait * (self.dt / 0.5)


def _surv(a: float, lo: float, hi: float) -> float:
    return float(np.clip((hi - a) / (hi - lo), 0.0, 1.0))


def posterior_captor(p: Params, phase: str, age: float = 0.0) -> float:
    if phase == "slow":
        return p.pi
    pk_b = 1.0 - p.p_pass
    if phase == "park":
        return p.pi / (p.pi + (1.0 - p.pi) * pk_b)
    s_t = _surv(age, *p.tau)
    s_b = 1.0 if age <= p.leave[0] else _surv(age, *p.leave)
    num = p.pi * s_t
    den = num + (1.0 - p.pi) * pk_b * s_b
    return num / den if den > 0.0 else 0.0


def hazard_exit(a: float, dt: float, lo: float, hi: float) -> float:
    if a < lo - 1e-12:
        return 0.0
    rem = hi - a
    if rem <= dt + 1e-12:
        return 1.0
    return dt / rem


def flight_cost_closed_form(
    p: Params,
    x0: np.ndarray,
    delta: float,
    yard: float,
    captor: bool,
    rem_to_deploy: np.ndarray,
    already_deployed: bool,
) -> np.ndarray:
    step_c = p.step_wait_cost
    hold = delta > x0
    xe = np.where(hold, x0, x0 + p.v_walk * p.t_react)
    wait_cost = np.where(hold, (p.t_react / p.dt) * step_c, 0.0)

    if captor:
        t_em = np.where(already_deployed, 0.0, rem_to_deploy)
        em_early = t_em < p.t_react
        xe_em = np.where(hold, x0, x0 + p.v_walk * np.maximum(0.0, t_em))
        rem_r = np.maximum(0.0, p.t_react - t_em)
        dist_at_em = np.abs(delta - xe_em)
        v_ev_early = np.where(hold, 0.0, np.where(delta > x0, p.v_walk, -p.v_walk))
        closing = p.v_catch - v_ev_early
        caught_early = em_early & (
            (dist_at_em <= p.contact)
            | ((closing > 0) & ((dist_at_em - p.contact) / np.maximum(1e-9, closing) <= rem_r))
        )
        dir_to_e = np.sign(xe - delta)
        xc_react = np.where(
            em_early,
            delta + dir_to_e * np.minimum(dist_at_em, p.v_catch * rem_r),
            delta,
        )
    else:
        caught_early = np.zeros_like(x0, dtype=bool)
        xc_react = np.full_like(x0, delta)
        t_em = np.full_like(x0, np.inf)

    ref = np.where(already_deployed | (t_em <= p.t_react), xc_react, delta)
    d_van = ref - xe
    s_away = np.where(d_van < 0.0, 1.0, -1.0)
    dy = yard - xe
    h = np.where(already_deployed | (t_em <= p.t_react), 0.0, np.maximum(0.0, rem_to_deploy - p.t_react))
    meet = h + np.maximum(0.0, np.abs(d_van) - p.contact - p.v_run * h) / (p.v_run + p.v_catch)
    between = (dy * (-s_away) > 0.0) & (np.abs(dy) / p.v_run < meet)
    viable = (dy * s_away > 0.0) | between | (np.abs(dy) < 1e-9)

    t_to_yard = np.where(viable, np.abs(dy) / p.v_run, np.inf)
    t_to_end = np.where((~viable) & (s_away > 0), np.maximum(0.0, p.street_len - xe) / p.v_run, np.inf)
    t_safe = np.minimum(t_to_yard, t_to_end)

    if not captor:
        t_run = np.minimum(t_safe, p.t_chase)
        return p.c_fp + wait_cost + (t_run / p.dt) * step_c

    t_chase_start = np.maximum(0.0, t_em - p.t_react)
    t_chase_end = t_chase_start + p.t_chase
    dirf = np.where(viable, np.sign(dy), s_away)
    xe_chase0 = xe + dirf * p.v_run * t_chase_start
    gap0 = np.abs(xe_chase0 - xc_react)

    # Fixed sign: evader moves away from pursuer iff (xe - xc) and dirf share sign
    diff_pos = xe_chase0 - xc_react
    away_from_pursuer = (diff_pos * dirf > 0.0) | (np.abs(diff_pos) < 1e-9)
    v_rel = np.where(away_from_pursuer, p.v_catch - p.v_run, p.v_catch + p.v_run)

    t_intercept = np.where(
        gap0 <= p.contact,
        t_chase_start,
        np.where(v_rel > 0, t_chase_start + (gap0 - p.contact) / np.maximum(1e-9, v_rel), np.inf),
    )

    caught_late = (~caught_early) & (t_intercept <= np.minimum(t_safe, t_chase_end))
    caught = caught_early | caught_late

    t_run = np.where(
        caught_early,
        0.0,
        np.where(caught_late, np.maximum(0.0, t_intercept), np.minimum(t_safe, t_chase_end)),
    )
    return np.where(caught, p.c_catch, 0.0) + wait_cost + (t_run / p.dt) * step_c


def solve_cell_dp(p: Params, delta: float, yard: float):
    dt = p.dt
    dx = p.v_walk * dt
    n_x = int(math.ceil(p.street_len / dx)) + 1
    xs = np.arange(n_x) * dx

    # Coarse knots every 1.4 m (10 steps) -> tiny matrices that stay inside L1 cache
    knot_idx = np.arange(0, n_x, 10)
    if knot_idx[-1] != n_x - 1:
        knot_idx = np.append(knot_idx, n_x - 1)
    xk = xs[knot_idx]
    nk = xk.size

    n_s = int(round(p.slow[1] / dt))
    n_p = int(round(p.park[1] / dt))
    n_w = int(round(p.leave[1] / dt))

    v_emerged = flight_cost_closed_form(p, xs, delta, yard, True, np.zeros(n_x), True)
    c_benign_flee = flight_cost_closed_form(p, xs, delta, yard, False, np.full(n_x, np.inf), False)

    n_w_act = int(round(p.tau[1] / dt))
    q_flee_w = np.repeat(c_benign_flee[:, None], n_w, axis=1)
    u_quad = (np.arange(8) + 0.5) / 8.0
    ages_w = np.arange(n_w_act) * dt
    lo_rems = np.maximum(0.0, p.tau[0] - ages_w)[:, None]
    hi_rems = np.maximum(dt, p.tau[1] - ages_w)[:, None]
    rems_w = lo_rems + u_quad[None, :] * (hi_rems - lo_rems)

    X_w = np.broadcast_to(xk[:, None, None], (nk, n_w_act, 8))
    R_w = np.broadcast_to(rems_w[None, :, :], (nk, n_w_act, 8))
    c_cap_wk = flight_cost_closed_form(p, X_w, delta, yard, True, R_w, False).mean(axis=2)
    c_cap_w = np.array([np.interp(xs, xk, c_cap_wk[:, j]) for j in range(n_w_act)]).T
    b_w = np.array([posterior_captor(p, "wait", float(a)) for a in ages_w])[None, :]
    q_flee_w[:, :n_w_act] = b_w * c_cap_w + (1.0 - b_w) * c_benign_flee[:, None]

    ages_p = np.arange(n_p) * dt
    exp_p_rem = 0.5 * (np.maximum(p.park[0], ages_p) + p.park[1]) - ages_p
    tau_quad = p.tau[0] + ((np.arange(6) + 0.5) / 6.0) * (p.tau[1] - p.tau[0])
    rems_p = exp_p_rem[:, None] + tau_quad[None, :]
    X_p = np.broadcast_to(xk[:, None, None], (nk, n_p, 6))
    R_p = np.broadcast_to(rems_p[None, :, :], (nk, n_p, 6))
    c_cap_pk = flight_cost_closed_form(p, X_p, delta, yard, True, R_p, False).mean(axis=2)
    c_cap_p = np.array([np.interp(xs, xk, c_cap_pk[:, j]) for j in range(n_p)]).T
    b_p = posterior_captor(p, "park", 0.0)
    q_flee_p = b_p * c_cap_p + (1.0 - b_p) * c_benign_flee[:, None]

    ages_s = np.arange(n_s) * dt
    exp_s_rem = 0.5 * (np.maximum(p.slow[0], ages_s) + p.slow[1]) - ages_s
    rems_s = exp_s_rem[:, None] + 2.0 + tau_quad[None, :]
    X_s = np.broadcast_to(xk[:, None, None], (nk, n_s, 6))
    R_s = np.broadcast_to(rems_s[None, :, :], (nk, n_s, 6))
    c_cap_sk = flight_cost_closed_form(p, X_s, delta, yard, True, R_s, False).mean(axis=2)
    c_cap_s = np.array([np.interp(xs, xk, c_cap_sk[:, j]) for j in range(n_s)]).T
    b_s = posterior_captor(p, "slow", 0.0)
    q_flee_s = b_s * c_cap_s + (1.0 - b_s) * c_benign_flee[:, None]

    step_c = p.step_wait_cost
    V_w = np.zeros((n_x, n_w + 1))
    pi_w = np.zeros((n_x, n_w), dtype=np.int8)

    for j in range(n_w - 1, -1, -1):
        a = j * dt
        b = posterior_captor(p, "wait", a)
        p_emerge = b * hazard_exit(a, dt, *p.tau)
        p_leave = (1.0 - b) * hazard_exit(a, dt, *p.leave)
        p_stay = max(0.0, 1.0 - p_emerge - p_leave)

        v_next_stay = V_w[:, j + 1]
        v_next_adv = np.append(v_next_stay[1:], 0.0)
        v_em_adv = np.append(v_emerged[1:], 0.0)

        qs = np.stack([
            p_emerge * v_em_adv + p_stay * v_next_adv,
            step_c + p_emerge * v_emerged + p_stay * v_next_stay,
            q_flee_w[:, j],
        ], axis=1)
        pi_w[:, j] = np.argmin(qs, axis=1)
        V_w[:, j] = np.min(qs, axis=1)
        V_w[-1, j] = 0.0

    V_p = np.zeros((n_x, n_p + 1))
    V_p[:, n_p] = V_w[:, 0]
    pi_p = np.zeros((n_x, n_p), dtype=np.int8)

    for j in range(n_p - 1, -1, -1):
        a = j * dt
        p_exit = hazard_exit(a, dt, *p.park)
        v_cont_stay = p_exit * V_w[:, 0] + (1.0 - p_exit) * V_p[:, j + 1]
        v_cont_adv = np.append(v_cont_stay[1:], 0.0)

        qs = np.stack([v_cont_adv, step_c + v_cont_stay, q_flee_p[:, j]], axis=1)
        pi_p[:, j] = np.argmin(qs, axis=1)
        V_p[:, j] = np.min(qs, axis=1)
        V_p[-1, j] = 0.0

    V_s = np.zeros((n_x, n_s + 1))
    V_s[:, n_s] = V_p[:, 0]
    pi_s = np.zeros((n_x, n_s), dtype=np.int8)
    p_pass_given_exit = (1.0 - p.pi) * p.p_pass

    for j in range(n_s - 1, -1, -1):
        a = j * dt
        p_exit = hazard_exit(a, dt, *p.slow)
        p_to_park = p_exit * (1.0 - p_pass_given_exit)
        v_cont_stay = p_to_park * V_p[:, 0] + (1.0 - p_exit) * V_s[:, j + 1]
        v_cont_adv = np.append(v_cont_stay[1:], 0.0)

        qs = np.stack([v_cont_adv, step_c + v_cont_stay, q_flee_s[:, j]], axis=1)
        pi_s[:, j] = np.argmin(qs, axis=1)
        V_s[:, j] = np.min(qs, axis=1)
        V_s[-1, j] = 0.0

    return (pi_s, pi_p, pi_w), float(V_s[0, 0])


def eval_cell_mc_vec(p: Params, delta: float, yard: float, dp_tables, n: int, seed: int):
    pi_s, pi_p, pi_w = dp_tables
    dt = p.dt
    dx = p.v_walk * dt
    step_c = p.step_wait_cost
    t_min_global = p.slow[0] + p.park[0] + p.tau[0]

    all_pols = POLICIES + ("optimal_dp",)
    res = {pol: dict(pc=0.0, fa=0.0, J=0.0) for pol in all_pols}

    for is_captor, weight in ((True, p.pi), (False, 1.0 - p.pi)):
        rng = np.random.default_rng([seed, 0 if is_captor else 1])
        ds = rng.uniform(*p.slow, n)
        dp_dur = rng.uniform(*p.park, n)
        tau = rng.uniform(*p.tau, n)
        u = rng.uniform(*p.leave, n)
        passes = rng.random(n) < p.p_pass

        t_park = ds
        t_wait = ds + dp_dur
        if is_captor:
            parks = np.ones(n, bool)
            t_deploy = t_wait + tau
            t_clear = np.full(n, np.inf)
        else:
            parks = ~passes
            t_deploy = np.full(n, np.inf)
            t_clear = np.where(parks, t_wait + u, ds)

        for pol in all_pols:
            xe = np.zeros(n)
            xc = np.full(n, delta)
            done = np.zeros(n, bool)
            caught = np.zeros(n, bool)
            started = np.zeros(n, bool)
            fled = np.zeros(n, bool)
            deployed = np.zeros(n, bool)
            dirf = np.zeros(n)
            has_target = np.zeros(n, bool)
            flee_elapsed = np.zeros(n)
            wait_steps = np.zeros(n, dtype=np.int32)
            flee_steps = np.zeros(n, dtype=np.int32)

            if pol == "ignore":
                trig_t = np.full(n, np.inf)
            elif pol in ("naive", "wait_deploy"):
                trig_t = t_deploy.copy()
            elif pol == "flee_park":
                trig_t = np.where(parks, t_park, np.inf)
            elif pol == "flee_slow":
                trig_t = np.zeros(n)
            else:
                trig_t = np.full(n, np.inf)

            for k in range(int(round(p.t_max / dt))):
                t = k * dt
                active = ~done
                if not active.any():
                    break

                dp_act = np.zeros(n, dtype=np.int8)
                if pol == "optimal_dp":
                    untrig = active & (~started) & np.isinf(trig_t)
                    if untrig.any():
                        dep_now = untrig & (t >= t_deploy - 1e-12)
                        trig_t = np.where(dep_now, t, trig_t)

                        in_w = untrig & (~dep_now) & (t < t_clear - 1e-12) & (t >= t_wait - 1e-12)
                        in_p = untrig & (~dep_now) & (t < t_wait - 1e-12) & (t >= t_park - 1e-12)
                        in_s = untrig & (~dep_now) & (t < t_park - 1e-12)

                        ix = np.clip(np.rint(xe / dx).astype(np.int32), 0, pi_s.shape[0] - 1)
                        if in_w.any():
                            jw = np.clip(np.rint((t - t_wait) / dt).astype(np.int32), 0, pi_w.shape[1] - 1)
                            dp_act = np.where(in_w, pi_w[ix, jw], dp_act)
                        if in_p.any():
                            jp = np.clip(np.rint((t - t_park) / dt).astype(np.int32), 0, pi_p.shape[1] - 1)
                            dp_act = np.where(in_p, pi_p[ix, jp], dp_act)
                        if in_s.any():
                            js = np.clip(np.rint(t / dt).astype(np.int32), 0, pi_s.shape[1] - 1)
                            dp_act = np.where(in_s, pi_s[ix, js], dp_act)

                        trig_t = np.where(untrig & (dp_act == 2), t, trig_t)

                newly = active & (~started) & (t >= trig_t + p.t_react - 1e-12)
                if newly.any():
                    ref = np.where(deployed, xc, delta)
                    d_van = ref - xe
                    s_away = np.where(d_van < 0.0, 1.0, -1.0)
                    dy = yard - xe
                    h = np.where(deployed, 0.0, max(0.0, t_min_global - t))
                    meet = h + np.maximum(0.0, np.abs(d_van) - p.contact - p.v_run * h) / (p.v_run + p.v_catch)
                    between = (dy * (-s_away) > 0.0) & (np.abs(dy) / p.v_run < meet)
                    viable = (dy * s_away > 0.0) | between | (np.abs(dy) < 1e-9)
                    dirn = np.where(viable, np.sign(dy), s_away)
                    dirf = np.where(newly, dirn, dirf)
                    has_target = np.where(newly, viable, has_target)
                    started |= newly
                    fled |= newly

                fleeing = active & started
                normal = active & (~started)
                if pol == "optimal_dp":
                    hold = normal & ((dp_act == 1) | ((~np.isinf(trig_t)) & (delta > xe))) & (t < t_clear - 1e-12)
                elif pol in ("wait_deploy", "flee_park", "flee_slow"):
                    hold = normal & (t < t_clear - 1e-12) & (delta > xe)
                else:
                    hold = np.zeros(n, bool)

                dxe = np.where(fleeing, dirf * p.v_run * dt, np.where(hold, 0.0, p.v_walk * dt))
                to_y = dirf * (yard - xe)
                crossing = fleeing & has_target & (to_y <= np.abs(dxe) + 1e-12)
                s_end = np.where(crossing & (np.abs(dxe) > 0.0), np.clip(to_y / np.maximum(1e-12, np.abs(dxe)), 0.0, 1.0), 1.0)

                wait_steps += hold
                flee_steps += fleeing
                flee_elapsed += fleeing * dt

                if is_captor:
                    newly_dep = active & (~deployed) & (t >= t_deploy - 1e-12)
                    mov_dep = active & deployed
                    if mov_dep.any():
                        ga = xe - xc
                        dxc = np.minimum(p.v_catch * dt, np.abs(ga)) * np.sign(ga)
                        gb = ga + (dxe - dxc) * s_end
                        cap_now = mov_dep & ((ga * gb <= 0.0) | (np.minimum(np.abs(ga), np.abs(gb)) <= p.contact))
                        caught |= cap_now
                        done |= cap_now
                        xc = np.where(mov_dep & (~cap_now), xc + dxc, xc)
                    if newly_dep.any():
                        deployed |= newly_dep
                        xc = np.where(newly_dep, delta, xc)
                        cap_imm = newly_dep & (~done) & (np.abs(xe - xc) <= p.contact)
                        caught |= cap_imm
                        done |= cap_imm

                reach = (~done) & crossing
                xe = np.where(reach, yard, xe + np.where(~done, dxe, 0.0))
                done |= reach
                done |= (~done) & (~started) & (t >= t_clear - 1e-12)
                done |= (~done) & (xe >= p.street_len)

                if is_captor:
                    done |= (~done) & deployed & (t >= t_deploy + p.t_chase - 1e-12)
                else:
                    done |= (~done) & started & (flee_elapsed >= p.t_chase - 1e-12)

            cost = (p.c_catch * caught if is_captor else p.c_fp * fled) + step_c * (wait_steps + flee_steps)
            if is_captor:
                res[pol]["pc"] = float(caught.mean())
            else:
                res[pol]["fa"] = float(fled.mean())
            res[pol]["J"] += weight * float(cost.mean())

    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    p = Params()
    dgrid = np.arange(-40.0, 60.1, 4.0)
    ygrid = np.arange(-64.0, 64.1, 8.0)

    all_pols = POLICIES + ("optimal_dp",)
    grid_J = {pol: np.zeros((dgrid.size, ygrid.size)) for pol in all_pols}
    grid_pc = {pol: np.zeros((dgrid.size, ygrid.size)) for pol in all_pols}
    grid_fa = {pol: np.zeros((dgrid.size, ygrid.size)) for pol in all_pols}
    rows = []

    for i, d in enumerate(dgrid):
        for j, y in enumerate(ygrid):
            tables, v_dp = solve_cell_dp(p, float(d), float(y))
            cell_res = eval_cell_mc_vec(p, float(d), float(y), tables, args.n, args.seed)
            for pol in all_pols:
                grid_J[pol][i, j] = cell_res[pol]["J"]
                grid_pc[pol][i, j] = cell_res[pol]["pc"]
                grid_fa[pol][i, j] = cell_res[pol]["fa"]
            best_heur = min(cell_res[pol]["J"] for pol in POLICIES)
            rows.append([d, y, f"{v_dp:.4f}", f"{cell_res['optimal_dp']['J']:.4f}",
                         f"{best_heur:.4f}", f"{best_heur - cell_res['optimal_dp']['J']:.4f}",
                         f"{cell_res['flee_slow']['J'] - cell_res['optimal_dp']['J']:.4f}"])

    print(f"\n=== Grid-Mean Comparison (Stage 1, dt = 0.1 s, {time.time() - t0:.1f}s) ===")
    print(f"  {'Policy':<14}{'P(caught|cap)':>16}{'P(FA|benign)':>16}{'Mean Cost J':>14}")
    for pol in all_pols:
        print(f"  {pol:<14}{grid_pc[pol].mean():>16.3f}{grid_fa[pol].mean():>16.3f}{grid_J[pol].mean():>14.3f}")

    regret_thresh = grid_J["flee_slow"] - grid_J["optimal_dp"]
    regret_heur = np.minimum.reduce([grid_J[pol] for pol in POLICIES]) - grid_J["optimal_dp"]
    max_idx = np.unravel_index(np.argmax(regret_thresh), regret_thresh.shape)
    print("\n=== Prediction 3 Check (Threshold Regret vs Optimal Policy) ===")
    print(f"  Mean regret of constant threshold (flee_slow) over grid: {regret_thresh.mean():.3f}")
    print(f"  Max regret of constant threshold: {regret_thresh[max_idx]:.3f} at delta={dgrid[max_idx[0]]:.0f} m, y={ygrid[max_idx[1]]:.0f} m")
    print(f"  Mean gain of optimal_dp over best of 5 hand-built policies: {regret_heur.mean():.3f}")

    with open(os.path.join(args.out_dir, "qash_v2_dp_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_m", "yard_m", "V_dp_analytic", "J_optimal_dp", "J_best_heuristic", "gain_vs_best_heur", "regret_threshold"])
        w.writerows(rows)

    tex_lines = [
        "% generated by qash_v2_dp.py -- do not edit by hand",
        f"\\newcommand{{\\missGridOptimalDP}}{{{grid_pc['optimal_dp'].mean():.3f}}}",
        f"\\newcommand{{\\falseAlarmOptimalDP}}{{{grid_fa['optimal_dp'].mean():.3f}}}",
        f"\\newcommand{{\\costGridOptimalDP}}{{{grid_J['optimal_dp'].mean():.3f}}}",
        f"\\newcommand{{\\meanRegretThreshold}}{{{regret_thresh.mean():.3f}}}",
        f"\\newcommand{{\\maxRegretThreshold}}{{{regret_thresh[max_idx]:.3f}}}",
        f"\\newcommand{{\\maxRegretDelta}}{{{dgrid[max_idx[0]]:.0f}}}",
        f"\\newcommand{{\\maxRegretYard}}{{{ygrid[max_idx[1]]:.0f}}}",
        f"\\newcommand{{\\meanGainOverHeuristics}}{{{regret_heur.mean():.3f}}}",
    ]
    with open(os.path.join(args.out_dir, "qash_v2_dp_numbers.tex"), "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print("\nSaved qash_v2_dp_summary.csv and qash_v2_dp_numbers.tex")


if __name__ == "__main__":
    main()