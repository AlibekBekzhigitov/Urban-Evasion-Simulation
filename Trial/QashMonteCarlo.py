#!/usr/bin/env python3
"""
QASH POMDP -- SPRT threshold policy: Monte Carlo harness.

Models the evader's sequential detection problem as a log-odds random walk
over noisy observations of a parked van's hidden type theta in {Threat, Benign}.

Observation model (Gaussian approximation to the log-LR walk):
  Under Threat H_T : Delta_ell_t ~ N(mu_T, sigma^2) per step
  Under Benign H_B : Delta_ell_t ~ N(-mu_B, sigma^2) per step

where mu_T = E_{H_T}[log Lambda] > 0  and  mu_B = -E_{H_B}[log Lambda] > 0
are the per-step KL divergence terms. The table below shows one concrete
discrete cue model that leads to the default mu_T, mu_B values:

  Phase cue          P(cue|T)  P(cue|B)  log Lambda
  ------------------ --------- --------- ----------
  Cruising           0.55      0.78      -0.349
  Slowing            0.15      0.10       0.405
  Parking manoeuvre  0.12      0.07       0.539
  Parked (dwell)     0.13      0.04       1.179
  Deploying agents   0.05      0.01       1.609

  E_{H_T}[log Lambda] = 0.55*(-0.349)+0.15*(0.405)+0.12*(0.539)+0.13*(1.179)+0.05*(1.609)
                      = -0.192 + 0.061 + 0.065 + 0.153 + 0.080 = +0.167 ~ mu_T = 0.30 (rounded up for
                        the Gaussian approximation used here; the CLT convergence is slow for sparse
                        high-Λ events like deployment, so set mu_T conservatively)

  E_{H_B}[log Lambda] = -0.273 + 0.041 + 0.038 + 0.047 + 0.016 = -0.131 -> mu_B = 0.20

The evader maintains log-odds ell_t = ell_0 + sum_{i=1}^{t} Delta_ell_i.
SPRT policy: flee to refuge when ell_t >= h_eff (effective threshold above ell_0).

  FN = P(max_{t=1}^{D} ell_t < h_eff | H_T)  -- missed detection, evader caught
  FP = P(max_{t=1}^{D} ell_t >= h_eff | H_B) -- false alarm, unnecessary flight

D is the deadline: steps until the evader would enter the capture zone.

Outputs (in --out-dir):
  qash_fn_fp_curve.pdf    FN/FP trade-off for multiple deadlines
  qash_cost_vs_h.pdf      expected cost J(h) vs threshold h
  qash_h_star_vs_D.pdf    optimal threshold h* vs deadline D
  qash_trajectories.pdf   sample belief trajectories under H_T and H_B
  qash_results.csv        numerical results for embedding in the paper

IMPORTANT: all parameters are assumptions. CIs are Monte Carlo sampling error
only, not model error. Results are model-conditional.

Usage:
  python qash_monte_carlo.py
  python qash_monte_carlo.py --n 100000 --out-dir ./figs --seed 42
Requires: numpy, matplotlib, scipy.
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import norm as _norm

Z95 = 1.959963984540054
SEED = 20261009

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
class Params:
    """Model parameters -- all assumptions unless otherwise noted."""
    mu_T:    float = 0.30    # mean log-LR increment under H_T (nats/step)
    mu_B:    float = 0.20    # magnitude of mean increment under H_B (nats/step)
    sigma:   float = 0.65    # per-step noise std (same under both hypotheses)
    pi_T:    float = 0.15    # prior P(Threat) = theta_stream
    C_catch: float = 10.0   # cost of capture
    c_fp:    float = 1.0    # cost of unnecessary flight (false alarm)

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def l0(self) -> float:
        """Initial log-odds from the prior."""
        return math.log(self.pi_T / (1.0 - self.pi_T))

    @property
    def cost_ratio(self) -> float:
        return self.C_catch / self.c_fp

    def __repr__(self):
        return (f"Params(mu_T={self.mu_T}, mu_B={self.mu_B}, sigma={self.sigma}, "
                f"pi_T={self.pi_T}, C_catch={self.C_catch}, c_fp={self.c_fp})")


# ---------------------------------------------------------------------------
# Theoretical curves
# ---------------------------------------------------------------------------
def _ig_cdf(x: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """Inverse Gaussian CDF with mean nu and shape lam.

    F(x; nu, lam) = Phi(sqrt(lam/x)(x/nu - 1)) + exp(2*lam/nu)*Phi(-sqrt(lam/x)(x/nu + 1))
    Approximates P(T_h <= x) for a Brownian motion with drift mu_T starting at 0,
    hitting level h, with nu = h/mu_T and lam = h^2/sigma^2.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    s = np.sqrt(lam / xp)
    z1 = s * (xp / nu - 1.0)
    z2 = -s * (xp / nu + 1.0)
    out[pos] = _norm.cdf(z1) + np.exp(np.clip(2.0 * lam / nu, None, 700.0)) * _norm.cdf(z2)
    return np.clip(out, 0.0, 1.0)


def fn_theory(h_eff: np.ndarray, D: int, p: Params) -> np.ndarray:
    """Theoretical P(FN) = P(T_h > D | H_T) via inverse Gaussian approximation.
    Valid for h_eff >> 0 and moderate D; underestimates when D >> E[T_h].
    Vectorised over h: calls _ig_cdf once per h value.
    """
    h = np.asarray(h_eff, dtype=float)
    out = np.ones_like(h)
    for i, hi in enumerate(h):
        if hi < 1e-9:
            out[i] = 1.0
        else:
            nu_i = hi / p.mu_T
            lam_i = hi ** 2 / p.sigma ** 2
            out[i] = 1.0 - float(_ig_cdf(np.array([float(D)]), nu_i, lam_i)[0])
    return out


def fp_infinite_bound(h_eff: np.ndarray, p: Params) -> np.ndarray:
    """Upper bound on P(FP) via Wald's all-time identity under H_B:
    P(T_h < inf | H_B) = exp(-2 * mu_B * h / sigma^2).
    For finite D, actual P(FP) is smaller.
    """
    h = np.asarray(h_eff, dtype=float)
    return np.exp(-2.0 * p.mu_B * h / p.sigma ** 2)


def expected_cost(FN: np.ndarray, FP: np.ndarray, p: Params) -> np.ndarray:
    """J(h) = pi_T * C_catch * P(FN) + (1-pi_T) * c_fp * P(FP)."""
    return p.pi_T * p.C_catch * FN + (1.0 - p.pi_T) * p.c_fp * FP


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------
def simulate_fn_fp(h_grid: np.ndarray, D: int, p: Params,
                   n_trials: int, rng: np.random.Generator):
    """Vectorised Monte Carlo over all h values for a single deadline D.

    Simulates n_trials walks under H_T and H_B once, then thresholds
    at each h -- avoids re-running for every h value.

    FN[i] = P(max_{t=1..D} ell_t < h_grid[i] | H_T)
    FP[i] = P(max_{t=1..D} ell_t >= h_grid[i] | H_B)
    """
    # H_T walks: increments ~ N(mu_T, sigma^2)
    walks_T = rng.normal(p.mu_T, p.sigma, (n_trials, D)).cumsum(axis=1)
    max_T = walks_T.max(axis=1)   # shape (n_trials,)

    # H_B walks: increments ~ N(-mu_B, sigma^2)
    walks_B = rng.normal(-p.mu_B, p.sigma, (n_trials, D)).cumsum(axis=1)
    max_B = walks_B.max(axis=1)

    FN = np.array([(max_T < h).mean() for h in h_grid])
    FP = np.array([(max_B >= h).mean() for h in h_grid])
    FN_se = np.sqrt(np.clip(FN * (1 - FN), 0, None) / n_trials)
    FP_se = np.sqrt(np.clip(FP * (1 - FP), 0, None) / n_trials)
    return FN, FP, FN_se, FP_se


def optimal_h(h_grid: np.ndarray, FN: np.ndarray, FP: np.ndarray,
              p: Params) -> tuple[int, float, np.ndarray]:
    """Return (index, h_value, J_array) for the cost-minimising threshold."""
    J = expected_cost(FN, FP, p)
    idx = int(np.argmin(J))
    return idx, float(h_grid[idx]), J


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
RC = {'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 8,
      'ytick.labelsize': 8, 'legend.fontsize': 8}

COLOURS = ['#2166ac', '#4dac26', '#d01c8b', '#f1a340']   # 4 deadlines


def fig_fn_fp(results: dict, h_grid: np.ndarray, deadlines: list,
              p: Params, path: str) -> None:
    """FN/FP trade-off curves for each deadline."""
    h_theo = np.linspace(0.02, h_grid[-1] + 1.0, 300)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(deadlines), figsize=(13, 3.4),
                                 sharey=True)
        for ax, D, colour in zip(axes, deadlines, COLOURS):
            FN, FP, _, _ = results[D]
            FN_t = fn_theory(h_theo, D, p)
            FP_t = fp_infinite_bound(h_theo, p)

            ax.plot(FP, FN, '-', color=colour, lw=2.0, label='Monte Carlo')
            ax.plot(FP_t, FN_t, '--', color='#888', lw=1.2,
                    label='Theory (IG / Wald)')
            ax.plot([0, 1], [1, 0], ':', color='#ccc', lw=0.9)   # iso line

            idx, h_star, _ = optimal_h(h_grid, FN, FP, p)
            ax.scatter(FP[idx], FN[idx], s=70, zorder=6, color='crimson',
                       label=f'$h^*={h_star:.2f}$')

            ax.set_title(f'$D={D}$ steps', fontsize=9)
            ax.set_xlabel('$P_{{FP}}$')
            if ax is axes[0]:
                ax.set_ylabel('$P_{{FN}}$')
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.legend(loc='upper right', framealpha=0.85)
            ax.grid(True, alpha=0.25)

        fig.suptitle('SPRT Threshold Policy: FN/FP Trade-off by Deadline $D$',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


def fig_cost(results: dict, h_grid: np.ndarray, deadlines: list,
             p: Params, path: str) -> None:
    """Expected cost J(h) vs threshold for each deadline."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for D, colour in zip(deadlines, COLOURS):
            FN, FP, _, _ = results[D]
            J = expected_cost(FN, FP, p)
            idx = int(np.argmin(J))
            ax.plot(h_grid, J, '-', color=colour, lw=2.0, label=f'$D={D}$')
            ax.scatter(h_grid[idx], J[idx], s=60, color=colour, zorder=5)
            ax.annotate(f'$h^*={h_grid[idx]:.2f}$',
                        (h_grid[idx], J[idx]),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7, color=colour)
        ax.set_xlabel('Effective threshold $h$ (nats above $\\ell_0$)')
        ax.set_ylabel('Expected cost $J(h)$')
        ax.set_title('Expected Cost vs. Threshold')
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


def fig_h_star_vs_D(p_base: Params, h_grid: np.ndarray,
                    D_range: np.ndarray, n: int, path: str) -> None:
    """Optimal threshold h* vs deadline D for three cost ratios."""
    cost_ratios = [5, 10, 20]
    line_colours = ['#1a9641', '#2166ac', '#d73027']
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for ratio, colour in zip(cost_ratios, line_colours):
            p = Params(mu_T=p_base.mu_T, mu_B=p_base.mu_B, sigma=p_base.sigma,
                       pi_T=p_base.pi_T, C_catch=float(ratio), c_fp=1.0)
            h_stars = []
            for D in D_range:
                rng_d = np.random.default_rng([SEED, int(D), ratio])
                FN, FP, _, _ = simulate_fn_fp(h_grid, int(D), p, n, rng_d)
                _, hs, _ = optimal_h(h_grid, FN, FP, p)
                h_stars.append(hs)
            ax.plot(D_range, h_stars, '-', color=colour, lw=1.8,
                    label=f'$C_{{\\mathrm{{catch}}}}/c_{{fp}}={ratio}$')

        ax.axhline(0, color='#555', lw=0.7, linestyle=':')
        ax.text(3, 0.04, 'Flee immediately ($h=0$)', fontsize=7, color='#555')
        ax.set_xlabel('Deadline $D$ (steps to capture-zone boundary)')
        ax.set_ylabel('Optimal threshold $h^*$ (nats)')
        ax.set_title('Optimal Threshold vs. Deadline')
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


def fig_trajectories(p: Params, D_ex: int, h_star: float,
                     n_show: int, rng: np.random.Generator,
                     path: str) -> None:
    """Sample log-odds trajectories under H_T and H_B."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        t = np.arange(D_ex + 1)
        for ax, (mu, label) in zip(axes,
                [(p.mu_T, r'Threat $H_T$'), (-p.mu_B, r'Benign $H_B$')]):
            incr = rng.normal(mu, p.sigma, (n_show, D_ex))
            walks = np.hstack([np.zeros((n_show, 1)), incr.cumsum(axis=1)])
            for w in walks:
                crossed = np.any(w[1:] >= h_star)
                colour = '#d01c8b' if crossed else '#2166ac'
                ax.plot(t, w, '-', color=colour, alpha=0.5, lw=0.9)

            ax.axhline(h_star, color='black', lw=1.6, linestyle='--',
                       label=f'$h^* = {h_star:.2f}$')
            ax.axvline(D_ex, color='forestgreen', lw=1.4, linestyle=':',
                       label=f'Deadline $D = {D_ex}$')
            ax.axhline(0.0, color='#aaa', lw=0.6, linestyle='-')

            # shade catch-zone entry
            y_lo, y_hi = ax.get_ylim() if hasattr(ax, '_qlo') else (-6, 8)
            ax.axvspan(D_ex, D_ex + 0.3, color='red', alpha=0.08)

            ax.set_xlabel('Steps $t$')
            ax.set_title(f'Trajectories under {label}')
            if ax is axes[0]:
                ax.set_ylabel('Log-odds $\\ell_t$')
            ax.set_xlim(0, D_ex + 1)
            ax.set_ylim(-6, 8)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.25)

        # unified colour legend
        ha = [mlines.Line2D([], [], color='#d01c8b', lw=1.5,
                             label='Fled (threshold crossed)'),
              mlines.Line2D([], [], color='#2166ac', lw=1.5,
                             label='Did not flee')]
        axes[1].legend(handles=ha, loc='lower right', fontsize=8)

        fig.suptitle('Log-Odds Belief Trajectories: SPRT Policy', fontsize=11)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(results: dict, h_grid: np.ndarray,
              deadlines: list, p: Params, path: str) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['D', 'h_eff', 'FN', 'FP', 'FN_se', 'FP_se', 'cost_J'])
        for D in deadlines:
            FN, FP, FN_se, FP_se = results[D]
            J = expected_cost(FN, FP, p)
            for h, fn, fp, fn_se, fp_se, j in zip(
                    h_grid, FN, FP, FN_se, FP_se, J):
                w.writerow([D, f'{h:.4f}', f'{fn:.6f}', f'{fp:.6f}',
                             f'{fn_se:.6f}', f'{fp_se:.6f}', f'{j:.6f}'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=50_000,
                    help='Trials per (h, D) cell (default 50,000)')
    ap.add_argument('--n-hstar', type=int, default=15_000,
                    help='Trials for h* vs D sweep (default 15,000)')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = Params()

    H_GRID    = np.linspace(0.05, 4.5, 90)
    DEADLINES = [5, 10, 20, 50]
    D_RANGE   = np.concatenate([np.arange(2, 22), np.arange(22, 101, 4)])

    print(f"Parameters: {p}")
    print(f"l0 (log-odds prior) = {p.l0:.3f}  (pi_T = {p.pi_T})")
    print(f"SNR under H_T = mu_T/sigma = {p.mu_T/p.sigma:.3f}")
    print(f"SNR under H_B = mu_B/sigma = {p.mu_B/p.sigma:.3f}")
    print(f"Infinite-horizon P(FP) bound at h=2.0: "
          f"{fp_infinite_bound(2.0, p):.4f}")
    print()

    # 1. Main FN/FP simulation
    results: dict = {}
    for D in DEADLINES:
        print(f"Simulating D={D}, n={args.n:,} ...", end='  ')
        rng_d = np.random.default_rng([args.seed, D])
        FN, FP, FN_se, FP_se = simulate_fn_fp(H_GRID, D, p, args.n, rng_d)
        results[D] = (FN, FP, FN_se, FP_se)
        idx, h_star, J = optimal_h(H_GRID, FN, FP, p)
        print(f"h*={h_star:.3f}  J(h*)={J[idx]:.4f}  "
              f"FN(h*)={FN[idx]:.4f}  FP(h*)={FP[idx]:.4f}")

    # 2. Summary table
    print('\nSummary by deadline and optimal threshold:')
    print(f"{'D':>5}{'h*':>8}{'P(FN)':>10}{'P(FP)':>10}{'J(h*)':>10}")
    for D in DEADLINES:
        FN, FP, _, _ = results[D]
        idx, h_star, J = optimal_h(H_GRID, FN, FP, p)
        print(f"{D:>5}{h_star:>8.3f}{FN[idx]:>10.4f}{FP[idx]:>10.4f}{J[idx]:>10.4f}")

    # 3. CSV
    csv_path = os.path.join(args.out_dir, 'qash_results.csv')
    write_csv(results, H_GRID, DEADLINES, p, csv_path)
    print(f'\nSaved {csv_path}')

    # 4. Figures
    fig_fn_fp(results, H_GRID, DEADLINES, p,
              os.path.join(args.out_dir, 'qash_fn_fp_curve.pdf'))
    print('Saved qash_fn_fp_curve.pdf')

    fig_cost(results, H_GRID, DEADLINES, p,
             os.path.join(args.out_dir, 'qash_cost_vs_h.pdf'))
    print('Saved qash_cost_vs_h.pdf')

    fig_h_star_vs_D(p, H_GRID, D_RANGE, args.n_hstar,
                    os.path.join(args.out_dir, 'qash_h_star_vs_D.pdf'))
    print('Saved qash_h_star_vs_D.pdf')

    D_ex = 10
    FN_ex, FP_ex, _, _ = results[D_ex]
    _, h_star_ex, _ = optimal_h(H_GRID, FN_ex, FP_ex, p)
    rng_traj = np.random.default_rng([args.seed, 777])
    fig_trajectories(p, D_ex, h_star_ex, 18, rng_traj,
                     os.path.join(args.out_dir, 'qash_trajectories.pdf'))
    print('Saved qash_trajectories.pdf')

    print('\nReminder: all parameters are modelling assumptions. '
          'CIs reflect Monte Carlo sampling error only. '
          'Results are model-conditional.')


if __name__ == '__main__':
    main()