"""
Phase 1 - classical benchmarks for time-varying co-movement.

Why this phase exists:
    Before reaching for wavelets we establish what the *standard* toolkit says.
    Rolling correlation (and its "honest" econometric upgrade, DCC-GARCH) shows
    *that* co-movement rises in crises - but it collapses all investment
    horizons into a single number per day. The wavelet phase of the project
    answers the question classics cannot: *on which horizon* does the
    correlation spike live? This module provides the baseline for that contrast.

Contents
--------
    rolling_correlation(returns, pair, windows)   -> DataFrame of rolling corrs
    dcc_garch(returns, pair)                      -> Series of conditional corr
                                                     (two-step Engle 2002 DCC(1,1))
    plot_rolling_corr(...), plot_benchmark_pair(...) -> figures with episode bands

Notes on the DCC implementation:
    * Step 1: fit univariate GARCH(1,1) (zero-mean / constant-mean) to each leg
      via `arch`, take standardized residuals.
    * Step 2: maximise the DCC(1,1) correlation likelihood over (a, b) with
      correlation targeting (Q̄ = sample cov of std residuals), a, b ≥ 0,
      a + b < 1, using scipy's bounded optimiser.
    * For a 2-asset case this is fast and entirely standard.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger("benchmark")

ROLLING_WINDOWS = (30, 90, 250)


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------
def rolling_correlation(returns: pd.DataFrame, pair: tuple[str, str],
                        windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Rolling Pearson correlation of a pair over several window lengths.

    Returns a DataFrame with one column per window, indexed like `returns`.
    Rows where either leg is missing are dropped first (pairwise alignment),
    so late-listed assets (e.g. SOL) simply start later.
    """
    x, y = pair
    df = returns[[x, y]].dropna()
    out = {}
    for w in windows:
        out[f"corr_{w}d"] = df[x].rolling(w).corr(df[y])
    res = pd.DataFrame(out, index=df.index)
    return res


def mean_pairwise_rolling_corr(returns: pd.DataFrame, cols: list[str],
                               window: int = 90) -> pd.Series:
    """Average rolling correlation across all pairs in `cols` (portfolio-level
    'how connected is the book today' line; precursor of the Phase-4 early
    indicator)."""
    cols = [c for c in cols if c in returns.columns]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    acc = []
    for a, b in pairs:
        acc.append(rolling_correlation(returns, (a, b), windows=(window,))
                   .iloc[:, 0].rename(f"{a}-{b}"))
    mat = pd.concat(acc, axis=1)
    return mat.mean(axis=1).rename(f"mean_pairwise_corr_{window}d")


# ---------------------------------------------------------------------------
# DCC-GARCH (two-step, Engle 2002), 2 assets
# ---------------------------------------------------------------------------
def _garch_std_residuals(series: pd.Series) -> pd.Series:
    """Univariate GARCH(1,1) via arch; returns standardized residuals."""
    from arch import arch_model

    # scale to % for numerical stability of the optimiser (arch's own advice)
    am = arch_model(series * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    std_resid = res.std_resid.dropna()
    return std_resid


def _dcc_filter(z: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    """Run the DCC(1,1) recursion; return the path of conditional correlations
    rho_t for the 2-asset case. z: (T,2) standardized residuals."""
    T = z.shape[0]
    Q = Qbar.copy()
    rho = np.empty(T)
    for t in range(T):
        if t > 0:
            zt = z[t - 1][:, None]
            Q = (1 - a - b) * Qbar + a * (zt @ zt.T) + b * Q
        d = np.sqrt(np.diag(Q))
        rho[t] = Q[0, 1] / (d[0] * d[1])
    return rho


def _dcc_neg_loglik(params: np.ndarray, z: np.ndarray, Qbar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    rho = _dcc_filter(z, a, b, Qbar)
    rho = np.clip(rho, -0.999, 0.999)
    z1, z2 = z[:, 0], z[:, 1]
    ll = -0.5 * (np.log(1 - rho ** 2)
                 + (z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / (1 - rho ** 2))
    return -ll.sum()


def dcc_garch(returns: pd.DataFrame, pair: tuple[str, str]) -> tuple[pd.Series, dict]:
    """
    Two-step DCC(1,1)-GARCH(1,1) conditional correlation for a pair.

    Returns (rho_series, info) where info holds fitted (a, b) and the
    log-likelihood. Falls back to a, b = (0.05, 0.90) if the optimiser fails
    (logged loudly) so the pipeline never dies on one stubborn pair.
    """
    from scipy.optimize import minimize

    x, y = pair
    df = returns[[x, y]].dropna()
    zx = _garch_std_residuals(df[x])
    zy = _garch_std_residuals(df[y])
    common = zx.index.intersection(zy.index)
    z = np.column_stack([zx.loc[common].values, zy.loc[common].values])

    Qbar = np.cov(z.T)

    res = minimize(_dcc_neg_loglik, x0=np.array([0.05, 0.90]),
                   args=(z, Qbar), method="L-BFGS-B",
                   bounds=[(1e-6, 0.5), (1e-6, 0.999)])
    if res.success:
        a, b = res.x
    else:  # pragma: no cover
        logger.warning("DCC optimiser failed for %s-%s (%s); using fallback (0.05, 0.90)",
                       x, y, res.message)
        a, b = 0.05, 0.90

    rho = _dcc_filter(z, a, b, Qbar)
    info = {"a": float(a), "b": float(b),
            "loglik": float(-_dcc_neg_loglik(np.array([a, b]), z, Qbar)),
            "n_obs": int(len(common))}
    logger.info("DCC %s-%s: a=%.4f b=%.4f (a+b=%.4f), n=%d",
                x, y, a, b, a + b, len(common))
    return pd.Series(rho, index=common, name=f"dcc_{x}_{y}"), info


# ---------------------------------------------------------------------------
# Plot helpers (episode shading shared by all benchmark figures)
# ---------------------------------------------------------------------------
def shade_episodes(ax, alpha: float = 0.12, color: str = "red") -> None:
    for ep in C.EPISODES:
        ax.axvspan(pd.Timestamp(ep.start), pd.Timestamp(ep.end),
                   color=color, alpha=alpha, zorder=0)


def plot_benchmark_pair(returns: pd.DataFrame, pair: tuple[str, str],
                        dcc: pd.Series | None = None,
                        windows=ROLLING_WINDOWS, save: bool = True):
    """One figure per pair: rolling corr (all windows) + optional DCC line."""
    import matplotlib.pyplot as plt

    rc = rolling_correlation(returns, pair, windows)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col in rc.columns:
        ax.plot(rc.index, rc[col], lw=1.0, label=col.replace("corr_", "окно "))
    if dcc is not None:
        ax.plot(dcc.index, dcc.values, lw=1.2, color="black", ls="--",
                label="DCC-GARCH")
    shade_episodes(ax)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylim(-0.45, 1.05)
    ax.set_ylabel("корреляция")
    ax.set_title(f"{pair[0]} ↔ {pair[1]}: rolling-корреляция"
                 + (" и DCC-GARCH" if dcc is not None else "")
                 + " (красные полосы - стресс-эпизоды)")
    ax.legend(loc="lower left", ncols=4, fontsize=9)
    fig.tight_layout()
    if save:
        path = C.FIGURES / f"01_bench_{pair[0]}_{pair[1]}.png"
        fig.savefig(path, dpi=130)
        logger.info("saved %s", path)
    return fig, ax
