"""
Phase 4 - the bridge into risk: scale-dependent VaR/ES and the stress
correlation matrix.

Three deliverables (mirroring the project plan):

4.1 Scale-dependent VaR/ES
    The portfolio return is decomposed into frequency bands with a stationary
    wavelet transform (SWT, the à-trous/MODWT-style undecimated transform from
    PyWavelets, norm=True => variance-preserving). Detail level j carries
    fluctuations with periods ≈ 2^j .. 2^(j+1) days. Empirical VaR/ES per level
    shows *where on the horizon axis* the risk lives - a number ordinary VaR
    averages away.

4.2 Stress correlation matrix (the key result)
    Correlation/covariance of the crypto book estimated separately on
    * calm windows (config.CALM_WINDOWS),
    * stress windows (union of config.EPISODES),
    plus the *hybrid* matrix - calm marginal vols with stress correlations -
    which isolates the pure correlation-contagion effect on portfolio risk.
    VaR/ES (parametric normal + empirical where the sample allows) for each
    regime => results/var_es_table.csv.

4.3 Early-warning indicator (optional, honest)
    Mean significant-band coherence across core pairs as a "connectedness"
    signal; we check whether its spikes lead portfolio drawdowns and report
    the result whatever it is.

Normality caveat: parametric numbers use the normal quantile on purpose - the
point of the table is the *relative* jump of risk when the correlation regime
flips, holding the distributional assumption fixed. Empirical quantiles are
reported alongside; fat tails are listed in Limitations.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as C

logger = logging.getLogger("risk")

Z = {0.95: sps.norm.ppf(0.95), 0.99: sps.norm.ppf(0.99)}
# ES multiplier for the normal: E[X | X > z_a] = phi(z_a) / (1 - a)
ES_MULT = {a: float(sps.norm.pdf(sps.norm.ppf(a)) / (1 - a)) for a in (0.95, 0.99)}


# ---------------------------------------------------------------------------
# Portfolio assembly
# ---------------------------------------------------------------------------
def portfolio_returns(returns: pd.DataFrame,
                      weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted daily log-return of the crypto risk book (complete-case)."""
    weights = weights or C.RISK.portfolio_weights
    cols = list(weights)
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    df = returns[cols].dropna()
    port = pd.Series(df.values @ w, index=df.index, name="portfolio")
    logger.info("portfolio: %d obs (%s .. %s), weights=%s",
                len(port), port.index.min().date(), port.index.max().date(),
                dict(zip(cols, w.round(3))))
    return port


# ---------------------------------------------------------------------------
# 4.1 Scale-dependent VaR/ES (SWT decomposition)
# ---------------------------------------------------------------------------
def swt_decompose(x: pd.Series, levels: int | None = None,
                  wavelet: str | None = None) -> pd.DataFrame:
    """
    Undecimated wavelet decomposition of a series into detail bands.

    Returns a DataFrame: columns D1..D{levels} (periods 2^j..2^(j+1) days)
    plus S{levels} (the residual smooth). norm=True keeps total variance ≈
    sum of band variances (Parseval), which we verify in tests.
    """
    import pywt

    levels = levels or C.RISK.modwt_levels
    wavelet = wavelet or C.RISK.modwt_wavelet

    n = len(x)
    m = (n // 2 ** levels) * 2 ** levels        # SWT needs a multiple of 2^J
    trimmed = x.iloc[n - m:]
    coeffs = pywt.swt(trimmed.values, wavelet, level=levels,
                      trim_approx=True, norm=True)
    # coeffs = [cA_J, cD_J, cD_{J-1}, ..., cD_1]
    out = {f"S{levels}": coeffs[0]}
    for j, d in enumerate(coeffs[1:], start=0):
        lvl = levels - j
        out[f"D{lvl}"] = d
    df = pd.DataFrame(out, index=trimmed.index)
    ordered = [f"D{j}" for j in range(1, levels + 1)] + [f"S{levels}"]
    return df[ordered]


BAND_LABEL = {
    "D1": "2-4д", "D2": "4-8д", "D3": "8-16д",
    "D4": "16-32д", "D5": "32-64д", "D6": "64-128д",
}


def scale_var_table(port: pd.Series, levels: int | None = None) -> pd.DataFrame:
    """Empirical VaR/ES (95/99) and variance share per frequency band."""
    levels = levels or C.RISK.modwt_levels
    bands = swt_decompose(port, levels=levels)
    total_var = port.loc[bands.index].var()

    rows = []
    for col in bands.columns:
        d = bands[col]
        row = {
            "band": col,
            "periods": BAND_LABEL.get(col, f">{2**(levels+1)}д (тренд)"),
            "var_share_%": 100 * d.var() / total_var,
        }
        for a in (0.95, 0.99):
            q = d.quantile(1 - a)
            row[f"VaR{int(a*100)}_%"] = -100 * q
            tail = d[d <= q]
            row[f"ES{int(a*100)}_%"] = -100 * tail.mean() if len(tail) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("band").round(3)


# ---------------------------------------------------------------------------
# 4.2 Stress matrix and the VaR/ES comparison table
# ---------------------------------------------------------------------------
def _window_mask(index: pd.DatetimeIndex,
                 windows: list[tuple[str, str]]) -> np.ndarray:
    m = np.zeros(len(index), dtype=bool)
    for s, e in windows:
        m |= np.asarray((index >= s) & (index <= e))
    return m


def regime_samples(returns: pd.DataFrame,
                   weights: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    """Asset-return samples for calm / stress / full regimes (complete-case)."""
    weights = weights or C.RISK.portfolio_weights
    cols = list(weights)
    df = returns[cols].dropna()
    idx = df.index

    stress_windows = [(ep.start, ep.end) for ep in C.EPISODES]
    m_stress = _window_mask(idx, stress_windows)
    m_calm = _window_mask(idx, C.CALM_WINDOWS) & ~m_stress

    samples = {"full": df, "stress": df[m_stress], "calm": df[m_calm]}
    for k, v in samples.items():
        logger.info("regime %-6s: %d obs", k, len(v))
    return samples


def _para_var_es(w: np.ndarray, cov: np.ndarray) -> dict:
    """Zero-mean normal parametric portfolio VaR/ES from a covariance matrix."""
    sigma = float(np.sqrt(w @ cov @ w))
    out = {"vol_daily_%": 100 * sigma}
    for a in (0.95, 0.99):
        out[f"VaR{int(a*100)}_%"] = 100 * Z[a] * sigma
        out[f"ES{int(a*100)}_%"] = 100 * ES_MULT[a] * sigma
    return out


def _empirical_var_es(port: pd.Series) -> dict:
    out = {}
    for a in (0.95, 0.99):
        q = port.quantile(1 - a)
        tail = port[port <= q]
        out[f"empVaR{int(a*100)}_%"] = -100 * q
        out[f"empES{int(a*100)}_%"] = -100 * tail.mean() if len(tail) else np.nan
    return out


def stress_vs_normal_table(returns: pd.DataFrame,
                           weights: dict[str, float] | None = None
                           ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    The headline deliverable: portfolio VaR/ES across correlation regimes.

    Rows:
      calm              - calm-window covariance,
      full              - whole-sample covariance,
      stress            - stress-window covariance (vols AND correlations move),
      calm_vol×stress_corr - hybrid: calm marginal vols, stress correlation
                          matrix. The pure 'diversification evaporates' effect.
    """
    weights = weights or C.RISK.portfolio_weights
    cols = list(weights)
    w = np.array([weights[c] for c in cols]); w = w / w.sum()

    samples = regime_samples(returns, weights)
    cov_calm = samples["calm"].cov().values
    cov_full = samples["full"].cov().values
    cov_stress = samples["stress"].cov().values

    # hybrid: calm vols, stress correlations
    d_calm = np.sqrt(np.diag(cov_calm))
    corr_stress = samples["stress"].corr().values
    cov_hybrid = np.outer(d_calm, d_calm) * corr_stress

    rows = {}
    rows["calm"] = {**_para_var_es(w, cov_calm),
                    **_empirical_var_es(samples["calm"] @ w)}
    rows["full"] = {**_para_var_es(w, cov_full),
                    **_empirical_var_es(samples["full"] @ w)}
    rows["stress"] = {**_para_var_es(w, cov_stress),
                      **_empirical_var_es(samples["stress"] @ w)}
    rows["calm_vol×stress_corr"] = _para_var_es(w, cov_hybrid)

    table = pd.DataFrame(rows).T
    table.index.name = "regime"

    mats = {
        "corr_calm": samples["calm"].corr(),
        "corr_stress": samples["stress"].corr(),
        "cov_calm": pd.DataFrame(cov_calm, cols, cols),
        "cov_stress": pd.DataFrame(cov_stress, cols, cols),
        "cov_hybrid": pd.DataFrame(cov_hybrid, cols, cols),
    }
    return table, mats


# ---------------------------------------------------------------------------
# 4.3 Early-warning indicator (optional, honest)
# ---------------------------------------------------------------------------
def connectedness_signal(returns: pd.DataFrame,
                         pairs=(("BTC", "ETH"), ("BTC", "SOL")),
                         band: tuple[float, float] = (8.0, 32.0)) -> pd.Series:
    """Mean inside-COI coherence in a band, averaged across core pairs."""
    from . import wavelet as W

    series = []
    for pair in pairs:
        res = W.run_pair(returns, pair)
        series.append(W.band_mean_coherence(res, band))
    sig = pd.concat(series, axis=1).mean(axis=1)
    return sig.rename("connectedness")


def indicator_check(signal: pd.Series, port: pd.Series,
                    threshold_q: float = 0.85, horizon: int = 20,
                    drawdown_q: float = 0.10) -> dict:
    """
    Does a connectedness spike precede bad 20-day portfolio returns?

    Alarm: signal above its rolling 1y `threshold_q` quantile.
    Event: forward `horizon`-day portfolio return in its worst `drawdown_q` tail.
    Reports hit rate, base rate, lift and the count of alarms. No tuning -
    a single pre-registered configuration, reported as-is.
    """
    df = pd.concat([signal, port], axis=1).dropna()
    sig, p = df.iloc[:, 0], df.iloc[:, 1]
    fwd = p.rolling(horizon).sum().shift(-horizon)
    thresh = sig.rolling(252, min_periods=126).quantile(threshold_q)
    alarm = sig > thresh
    bad_cut = fwd.quantile(drawdown_q)
    bad = fwd < bad_cut

    valid = fwd.notna() & thresh.notna()
    alarm, bad = alarm[valid], bad[valid]
    hit_rate = float(bad[alarm].mean()) if alarm.sum() else np.nan
    base_rate = float(bad.mean())
    return {
        "n_alarm_days": int(alarm.sum()),
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": hit_rate / base_rate if base_rate else np.nan,
        "horizon_days": horizon,
    }
