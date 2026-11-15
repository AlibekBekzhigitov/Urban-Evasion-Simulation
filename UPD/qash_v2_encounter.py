#!/usr/bin/env python3
"""
QASH v2, stage 1 -- single encounter (no crowd, quota not yet modelled).

One van stops near an evader walking a street. The van is a captor van with
prior probability PI, otherwise benign. The evader sees the van's phase but not
its type. Everything is in metres and seconds (no pixels); dt = 0.5 s.

Van timeline (the same for both types until parking):
  slowing   S : duration ~ U(2, 4) s
  parking   P : duration ~ U(1, 3) s            (benign vans pass instead of
                                                 parking with prob P_PASS)
  waiting   W : captor -> catchers emerge after tau ~ U(3, 8) s
                benign -> van leaves after   u   ~ U(6, 20) s

Because the phase is observed, the type posterior is a closed-form function of
(phase, age in phase); see posterior_captor(). The only evidence is: does the
van park, and has a deployment happened by age a. A deployment is conclusive.

Evader policies (what makes the evader flee, and how he behaves before):
  ignore       never flees; walks on
  naive        walks on; flees when catchers emerge
  wait_deploy  holds still while a van ahead is active, else walks on;
               flees when catchers emerge
  flee_park    as wait_deploy, but flees as soon as the van starts to park
  flee_slow    as wait_deploy, but flees as soon as the van starts to slow

Flight: after t_react the evader runs at v_run to the yard if it is viable,
otherwise directly away from the van. A yard on the far side from the van is
always viable; a yard between him and the van is viable only if he can reach it
before the catchers, running at v_catch from where they now are (the van if they
have not yet emerged), would meet him. Captors run at v_catch toward the evader for at most
t_chase seconds and capture within `contact` metres. Reaching the yard entrance
makes the evader safe. Costs: caught = C_CATCH, fled from a benign van = C_FP,
plus C_WAIT per half-second spent waiting or fleeing.

All parameters are illustrative assumptions, not calibrated to any real
operation. Note: v_catch and t_chase default to 5.0 m/s and 15 s, not the 4.0
and 10 of the written spec, because with equal speeds a yard can never matter.
Run with --v-catch 4.0 --t-chase 10 to see the spec values.

Outputs (in --out-dir): qash_v2_riskmap.csv, qash_v2_numbers.tex,
qash_v2_riskmaps.pdf/png. Requires numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, replace

import numpy as np

Z95 = 1.959963984540054
POLICIES = ("ignore", "naive", "wait_deploy", "flee_park", "flee_slow")


@dataclass(frozen=True)
class Params:
    dt: float = 0.5
    t_max: float = 45.0
    street_len: float = 120.0
    pi: float = 0.15            # prior P(captor van)
    p_pass: float = 0.35        # benign vans that pass instead of parking
    slow: tuple = (2.0, 4.0)
    park: tuple = (1.0, 3.0)
    tau: tuple = (3.0, 8.0)     # captor deployment latency after parking
    leave: tuple = (6.0, 20.0)  # benign dwell after parking
    v_walk: float = 1.4
    v_run: float = 4.0
    v_catch: float = 5.0
    t_react: float = 1.0
    contact: float = 1.0
    t_chase: float = 15.0
    c_catch: float = 10.0
    c_fp: float = 1.0
    c_wait: float = 0.02


# --------------------------------------------------------------------------- #
# Posterior over van type, given what the evader has seen
# --------------------------------------------------------------------------- #
def _surv_uniform(a, lo, hi):
    return float(np.clip((hi - a) / (hi - lo), 0.0, 1.0))


def posterior_captor(p: Params, phase: str, age: float = 0.0) -> float:
    """P(captor | van seen in `phase`, `age` seconds into the waiting phase)."""
    if phase == "slow":
        return p.pi
    pk_b = 1.0 - p.p_pass
    if phase == "park":
        return p.pi / (p.pi + (1 - p.pi) * pk_b)
    s_t = _surv_uniform(age, *p.tau)
    s_b = 1.0 if age <= p.leave[0] else _surv_uniform(age, *p.leave)
    num = p.pi * s_t
    return num / (num + (1 - p.pi) * pk_b * s_b)


# --------------------------------------------------------------------------- #
# Van draws and policy triggers
# --------------------------------------------------------------------------- #
def draw_van(p: Params, captor: bool, n: int, rng):
    ds = rng.uniform(*p.slow, n)
    dp = rng.uniform(*p.park, n)
    tau = rng.uniform(*p.tau, n)
    u = rng.uniform(*p.leave, n)
    passes = rng.random(n) < p.p_pass
    t_park, t_wait = ds, ds + dp
    if captor:
        return dict(t_park=t_park, t_deploy=t_wait + tau,
                    t_clear=np.full(n, np.inf), parks=np.ones(n, bool))
    parks = ~passes
    return dict(t_park=t_park, t_deploy=np.full(n, np.inf),
                t_clear=np.where(parks, t_wait + u, ds), parks=parks)


def trigger(policy: str, van: dict, n: int):
    if policy == "ignore":
        return np.full(n, np.inf), False
    if policy == "naive":
        return van["t_deploy"], False
    if policy == "wait_deploy":
        return van["t_deploy"], True
    if policy == "flee_park":
        return np.where(van["parks"], van["t_park"], np.inf), True
    if policy == "flee_slow":
        return np.zeros(n), True
    raise ValueError(policy)


# --------------------------------------------------------------------------- #
# Simulation (vectorised over grid cells x episodes)
# --------------------------------------------------------------------------- #
def simulate(p: Params, policy: str, captor: bool, delta, yard, n: int, seed):
    """Return per-cell mean caught, mean fled, mean cost over n episodes.

    delta: van position relative to the evader's start (m, negative = behind).
    yard:  yard entrance position relative to the evader's start (m).
    The van draws depend only on `seed`, so every policy and cell sees the same vans.
    """
    rng = np.random.default_rng(seed)
    van = draw_van(p, captor, n, rng)
    trig, adaptive = trigger(policy, van, n)
    G = delta.size
    D, Yd = delta[:, None], yard[:, None]
    shape = (G, n)
    Db, Yb = np.broadcast_to(D, shape), np.broadcast_to(Yd, shape)

    xe = np.zeros(shape)
    xc = np.zeros(shape)
    done = np.zeros(shape, bool)
    caught = np.zeros(shape, bool)
    started = np.zeros(shape, bool)
    fled = np.zeros(shape, bool)
    deployed = np.zeros(shape, bool)
    has_target = np.zeros(shape, bool)
    dirf = np.zeros(shape)
    flee_elapsed = np.zeros(shape)
    wait_steps = np.zeros(shape)
    flee_steps = np.zeros(shape)

    flee_start = (trig + p.t_react)[None, :]
    t_deploy = van["t_deploy"][None, :]
    t_clear = van["t_clear"][None, :]

    for k in range(int(p.t_max / p.dt)):
        t = k * p.dt
        active = ~done

        newly = active & ~started & (t >= flee_start)
        if newly.any():
            ref = np.where(deployed, xc, Db)      # catchers if they have emerged, else the van
            d_van = ref - xe
            s_away = np.where(d_van < 0, 1.0, -1.0)
            dy = Yb - xe
            between = (dy * -s_away > 0) & (np.abs(dy) / p.v_run
                                            < (np.abs(d_van) - p.contact) / (p.v_run + p.v_catch))
            viable = (dy * s_away > 0) | between
            dirn = np.where(viable, np.sign(dy), s_away)
            dirf = np.where(newly, dirn, dirf)
            has_target = np.where(newly, viable, has_target)
            started |= newly
            fled |= newly

        fleeing = active & started
        normal = active & ~started
        if adaptive:
            wait = normal & (t < t_clear) & (Db > xe)
        else:
            wait = np.zeros(shape, bool)
        adv = normal & ~wait
        xe = xe + np.where(fleeing, dirf * p.v_run * p.dt, np.where(adv, p.v_walk * p.dt, 0.0))
        wait_steps += wait
        flee_steps += fleeing
        flee_elapsed += fleeing * p.dt

        newly_dep = active & ~deployed & (t >= t_deploy)
        moving = active & deployed
        gap = xe - xc
        xc = np.where(moving, xc + np.minimum(p.v_catch * p.dt, np.abs(gap)) * np.sign(gap), xc)
        xc = np.where(newly_dep, Db, xc)
        deployed |= newly_dep

        cap_now = active & deployed & (np.abs(xe - xc) <= p.contact)
        caught |= cap_now
        reach = started & has_target & (dirf * (xe - Yb) >= 0)
        timeout = deployed & (t >= t_deploy + p.t_chase)
        cleared = (t >= t_clear) & ~started
        arrived = xe >= p.street_len
        flight_end = started & (flee_elapsed >= p.t_chase)
        done |= active & (cap_now | reach | timeout | cleared | arrived | flight_end)
        if done.all():
            break

    cost = p.c_catch * caught + (0.0 if captor else p.c_fp) * fled \
        + p.c_wait * (wait_steps + flee_steps)
    return caught.mean(axis=1), fled.mean(axis=1), cost.mean(axis=1)


def evaluate(p: Params, dgrid, ygrid, n: int, seed: int) -> dict:
    Dm, Ym = np.meshgrid(dgrid, ygrid, indexing="ij")
    delta, yard = Dm.ravel(), Ym.ravel()
    res = {}
    for pol in POLICIES:
        pc, _, jt = simulate(p, pol, True, delta, yard, n, [seed, 0])
        _, fa, jb = simulate(p, pol, False, delta, yard, n, [seed, 1])
        res[pol] = dict(pc=pc.reshape(Dm.shape), fa=fa.reshape(Dm.shape),
                        J=(p.pi * jt + (1 - p.pi) * jb).reshape(Dm.shape))
    return res


TIE = len(POLICIES)  # code for "walk on and hold still behave identically"


def best_policy(res: dict):
    """Index of the lowest-cost policy per cell; naive/wait_deploy ties get their own code."""
    stack = np.stack([res[pol]["J"] for pol in POLICIES])
    best = stack.argmin(axis=0).ravel()
    j_naive, j_wait = stack[1].ravel(), stack[2].ravel()
    tie = (best == 1) & (np.abs(j_naive - j_wait) < 1e-12)
    best = np.where(tie, TIE, best)
    return best, stack.min(axis=0).ravel()


def wilson(k: float, n: int):
    ph = k / n
    den = 1 + Z95 ** 2 / n
    c = (ph + Z95 ** 2 / (2 * n)) / den
    h = Z95 * math.sqrt(ph * (1 - ph) / n + Z95 ** 2 / (4 * n * n)) / den
    return c - h, c + h


# --------------------------------------------------------------------------- #
# Reports, files, figure
# --------------------------------------------------------------------------- #
def cell(res, pol, dgrid, ygrid, d, y, key="pc"):
    i, j = int(np.abs(dgrid - d).argmin()), int(np.abs(ygrid - y).argmin())
    return float(res[pol][key][i, j]), dgrid[i], ygrid[j]


def report(p, res, dgrid, ygrid, n):
    print("\nPosterior P(captor) from the van timeline (prior %.2f)" % p.pi)
    thr = p.c_fp / (p.c_fp + p.c_catch)
    print(f"  slowing seen: {posterior_captor(p, 'slow'):.3f}   parking seen: {posterior_captor(p, 'park'):.3f}"
          f"   (textbook flee threshold c_fp/(c_fp+C_catch) = {thr:.3f})")
    for a in (0, 3, 5, 6, 7, 8):
        print(f"  waiting, age {a} s, no deployment yet: {posterior_captor(p, 'wait', a):.3f}")

    print("\nTrade-off by policy (grid means): miss vs false alarm")
    print(f"  {'policy':<12}{'P(caught|captor)':>18}{'P(false alarm|benign)':>24}{'mean cost J':>14}")
    for pol in POLICIES:
        print(f"  {pol:<12}{res[pol]['pc'].mean():>18.3f}{res[pol]['fa'].mean():>24.3f}{res[pol]['J'].mean():>14.3f}")

    print("\nPrediction 1: P(caught | captor) with the van beside the evader (delta = 0)")
    i0 = int(np.abs(dgrid).argmin())
    for pol in POLICIES:
        v = res[pol]["pc"][i0, :]
        print(f"  {pol:<12} mean over yard positions {v.mean():.3f}  (min {v.min():.3f}, max {v.max():.3f})")

    print("\nPrediction 2: van far ahead (delta = 40), yard between (y = 24)")
    for pol in ("naive", "wait_deploy"):
        v, d, y = cell(res, pol, dgrid, ygrid, 40, 24)
        lo, hi = wilson(v * n, n)
        print(f"  {pol:<12} P(caught|captor) = {v:.3f}  [{lo:.3f}, {hi:.3f}]  (cell delta={d:.0f}, y={y:.0f})")

    best, best_j = best_policy(res)
    print("\nLowest expected-cost policy, share of grid cells")
    for k, pol in enumerate(POLICIES):
        print(f"  {pol:<12}{100 * (best == k).mean():>6.1f}%")
    print(f"  {'walk = hold':<12}{100 * (best == TIE).mean():>6.1f}%   (naive and wait_deploy behave identically there)")
    return best, best_j


def write_files(p, res, dgrid, ygrid, best, out):
    with open(os.path.join(out, "qash_v2_riskmap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_m", "yard_m", "policy", "p_caught_given_captor", "p_false_alarm_given_benign", "cost_J"])
        for pol in POLICIES:
            for i, d in enumerate(dgrid):
                for j, y in enumerate(ygrid):
                    w.writerow([d, y, pol, f"{res[pol]['pc'][i, j]:.4f}",
                                f"{res[pol]['fa'][i, j]:.4f}", f"{res[pol]['J'][i, j]:.4f}"])
    i0 = int(np.abs(dgrid).argmin())
    names = dict(ignore="Ignore", naive="Naive", wait_deploy="WaitDeploy", flee_park="FleePark", flee_slow="FleeSlow")
    lines = ["% generated by qash_v2_encounter.py -- do not edit by hand",
             "\\newcommand{\\postSlow}{%.3f}" % posterior_captor(p, "slow"),
             "\\newcommand{\\postPark}{%.3f}" % posterior_captor(p, "park"),
             "\\newcommand{\\bayesThr}{%.3f}" % (p.c_fp / (p.c_fp + p.c_catch))]
    for k, pol in enumerate(POLICIES):
        lines.append("\\newcommand{\\best%s}{%.1f}" % (names[pol], 100 * (best == k).mean()))
        lines.append("\\newcommand{\\missBeside%s}{%.3f}" % (names[pol], res[pol]["pc"][i0, :].mean()))
        lines.append("\\newcommand{\\falseAlarm%s}{%.3f}" % (names[pol], res[pol]["fa"].mean()))
    lines.append("\\newcommand{\\bestTie}{%.1f}" % (100 * (best == TIE).mean()))
    with open(os.path.join(out, "qash_v2_numbers.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def make_figure(res, best, bestJ, dgrid, ygrid, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    hd, hy = (dgrid[1] - dgrid[0]) / 2, (ygrid[1] - ygrid[0]) / 2
    ext = [dgrid[0] - hd, dgrid[-1] + hd, ygrid[0] - hy, ygrid[-1] + hy]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    def heat(ax, arr, title, vmin, vmax, cmap, label):
        im = ax.imshow(arr.T, origin="lower", extent=ext, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axvline(0, color="white", lw=0.8, ls="--")
        ax.axhline(0, color="white", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Van stop position relative to evader (m)")
        ax.set_ylabel("Yard position relative to evader (m)")
        fig.colorbar(im, ax=ax, label=label, fraction=0.046)

    heat(axs[0, 0], res["naive"]["pc"], "Walk on, flee when catchers emerge", 0, 1, "viridis", "P(caught | captor)")
    heat(axs[0, 1], res["wait_deploy"]["pc"], "Hold still if van ahead, flee when catchers emerge", 0, 1, "viridis", "P(caught | captor)")
    heat(axs[0, 2], res["flee_park"]["pc"], "Flee as soon as the van starts to park", 0, 1, "viridis", "P(caught | captor)")

    ax = axs[1, 0]
    x = np.arange(len(POLICIES))
    miss = [res[pol]["pc"].mean() for pol in POLICIES]
    fa = [res[pol]["fa"].mean() for pol in POLICIES]
    ax.bar(x - 0.2, miss, 0.4, label="P(caught | captor), grid mean")
    ax.bar(x + 0.2, fa, 0.4, label="P(false alarm | benign)")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES, rotation=20, fontsize=8)
    ax.set_title("The miss / false-alarm trade-off", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    cols = ["#888780", "#378ADD", "#1D9E75", "#EF9F27", "#D4537E", "#B5D4F4"]
    labels = list(POLICIES) + ["walk = hold (tie)"]
    ax = axs[1, 1]
    ax.imshow(best.reshape(len(dgrid), len(ygrid)).T, origin="lower", extent=ext, aspect="auto",
              cmap=ListedColormap(cols), vmin=-0.5, vmax=5.5)
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.set_title("Lowest expected cost policy", fontsize=10)
    ax.set_xlabel("Van stop position relative to evader (m)")
    ax.set_ylabel("Yard position relative to evader (m)")
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(cols, labels)], fontsize=7, loc="upper left")

    heat(axs[1, 2], bestJ.reshape(len(dgrid), len(ygrid)), "Expected cost of the best policy", 0, None, "magma", "J")
    fig.suptitle("QASH v2 stage 1: single encounter (illustrative parameters)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qash_v2_riskmaps.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "qash_v2_riskmaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="episodes per cell, per van type")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--v-catch", type=float, default=5.0)
    ap.add_argument("--t-chase", type=float, default=15.0)
    ap.add_argument("--tau-lo", type=float, default=3.0, help="captor deployment latency after parking (s)")
    ap.add_argument("--tau-hi", type=float, default=8.0)
    ap.add_argument("--c-wait", type=float, default=0.02, help="cost per half-second spent waiting or fleeing")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p = replace(Params(), v_catch=args.v_catch, t_chase=args.t_chase,
                tau=(args.tau_lo, args.tau_hi), c_wait=args.c_wait)
    dgrid = np.arange(-40.0, 60.1, 4.0)
    ygrid = np.arange(-64.0, 64.1, 8.0)
    print(f"v_catch={p.v_catch} m/s, t_chase={p.t_chase} s, tau={p.tau} s, c_wait={p.c_wait}, "
          f"n={args.n} per cell and van type, seed={args.seed}")
    res = evaluate(p, dgrid, ygrid, args.n, args.seed)
    best, bestJ = report(p, res, dgrid, ygrid, args.n)
    write_files(p, res, dgrid, ygrid, best, args.out_dir)
    if not args.no_figure:
        make_figure(res, best, bestJ, dgrid, ygrid, args.out_dir)
    print("\nAll parameters are illustrative assumptions; CIs reflect sampling error only.")


if __name__ == "__main__":
    main()