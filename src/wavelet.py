"""
Phase 2 core - wavelet coherence (WTC) with COI and Monte-Carlo significance.

Built on top of ``pycwt`` (Grinsted/Torrence-Compo lineage). We use pycwt's fast
``wct(..., sig=False)`` for the coherence itself and implement the Monte-Carlo
significance test ourselves, because pycwt's built-in ``wct_significance`` is
orders of magnitude slower (it re-derives the null per call without vectorised
reuse) and, in the installed version, raises on string wavelet names.

Method (Grinsted, Moore & Jevrejeva 2004):
  1. Estimate AR(1) persistence of each input series (pycwt.ar1).
  2. Generate ``mc_count`` pairs of independent AR(1) ("red noise") surrogates
     with the same lag-1 autocorrelation and variance.
  3. Compute WTC for every surrogate pair; pool coherence values *inside the
     COI* per scale; the null 95th percentile per scale is the significance
     threshold. Observed R² > threshold => locally significant at 5%.

Caching: every (pair, input-content, params) result is saved to
``data/processed/wavelet/`` as an .npz so notebooks never recompute a
1-minute MC run. Old inputs are pruned per pair.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pycwt
from pycwt import Morlet

from . import config as C

logger = logging.getLogger("wavelet")

CACHE_DIR = C.DATA_PROCESSED / "wavelet"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MOTHER = Morlet(C.WAVELET.omega0)
CACHE_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def ar1_coef(x: np.ndarray) -> float:
    """Lag-1 autocorrelation via pycwt's unbiased estimator (Allen & Smith)."""
    try:
        g, _, _ = pycwt.ar1(x)
        return float(g)
    except Exception:                      # very short / degenerate series
        x0, x1 = x[:-1], x[1:]
        return float(np.corrcoef(x0, x1)[0, 1])


def ar1_surrogate(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian AR(1) surrogate of length n with lag-1 autocorr phi, unit var.
    Vectorised via lfilter (an AR(1) is an IIR filter on white noise)."""
    from scipy.signal import lfilter

    phi = float(np.clip(phi, -0.99, 0.99))
    eps = rng.normal(0, np.sqrt(1 - phi ** 2), n + 100)  # 100-sample burn-in
    eps[0] = rng.normal()                                # stationary start
    out = lfilter([1.0], [1.0, -phi], eps)
    return out[100:]


def _coi_mask(coi: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Boolean (n_scales, n_times): True where the point is INSIDE the COI
    (interpretable region): period <= coi(t)."""
    return periods[:, None] <= coi[None, :]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def wct_pair(x: pd.Series, y: pd.Series,
             dt: float = None, dj: float = None) -> dict:
    """
    Wavelet coherence of two aligned series.

    Returns dict with:
      R2 (n_scales, n_times), phase (radians, same shape), periods (days),
      coi (days), times (DatetimeIndex), inside_coi (bool mask), dt, dj.
    """
    dt = dt or C.WAVELET.dt
    dj = dj or C.WAVELET.dj

    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 200:
        raise ValueError(f"too few overlapping points: {len(df)}")
    xa = _standardize(df.iloc[:, 0].values)
    ya = _standardize(df.iloc[:, 1].values)

    R2, phase, coi, freq, _ = pycwt.wct(xa, ya, dt=dt, dj=dj,
                                        s0=C.WAVELET.s0_factor * dt,
                                        sig=False, wavelet=MOTHER,
                                        normalize=False)
    periods = 1.0 / freq
    return {
        "R2": R2, "phase": phase, "periods": periods, "coi": coi,
        "times": df.index, "inside_coi": _coi_mask(coi, periods),
        "dt": dt, "dj": dj, "n": len(df),
        "ar1_x": ar1_coef(xa), "ar1_y": ar1_coef(ya),
    }


def mc_significance(n: int, ar1_x: float, ar1_y: float,
                    periods: np.ndarray, coi: np.ndarray,
                    mc_count: int = None, dt: float = None, dj: float = None,
                    seed: int = C.RANDOM_SEED,
                    significance_level: float = None) -> np.ndarray:
    """
    Monte-Carlo null for WTC against AR(1) red-noise surrogate pairs.

    Returns sig (n_scales,) - the per-scale null quantile of coherence
    (e.g. 95th). Pools null values inside the COI only, so edge-inflated
    coherence does not contaminate the threshold.
    """
    mc_count = mc_count or C.WAVELET.mc_count
    dt = dt or C.WAVELET.dt
    dj = dj or C.WAVELET.dj
    significance_level = significance_level or C.WAVELET.significance_level

    rng = np.random.default_rng(seed)
    inside = _coi_mask(coi, periods)
    n_scales = len(periods)

    # Accumulate per-scale samples of null coherence. Memory note: storing every
    # inside-COI cell for 1000 surrogates is ~1 GB; we instead keep a thinned
    # subsample (~50 values per scale per surrogate, float32). Null values are
    # strongly autocorrelated along time, so a thinned pooled sample estimates
    # the 95% quantile with negligible extra error (~50k draws per scale).
    per_surrogate_per_scale = 50
    acc: list[list[np.ndarray]] = [[] for _ in range(n_scales)]
    for i in range(mc_count):
        sx = ar1_surrogate(n, ar1_x, rng)
        sy = ar1_surrogate(n, ar1_y, rng)
        R2s, _, _, _, _ = pycwt.wct(sx, sy, dt=dt, dj=dj,
                                    s0=C.WAVELET.s0_factor * dt,
                                    sig=False, wavelet=MOTHER, normalize=False)
        # surrogate grid can differ by one scale row on rare lengths; align
        k = min(R2s.shape[0], n_scales)
        for s in range(k):
            vals = R2s[s][inside[s][:R2s.shape[1]]]
            if vals.size:
                step = max(1, vals.size // per_surrogate_per_scale)
                acc[s].append(vals[::step].astype(np.float32))
        if (i + 1) % 100 == 0:
            logger.info("MC %d/%d", i + 1, mc_count)

    sig = np.full(n_scales, np.nan)
    for s in range(n_scales):
        if acc[s]:
            sig[s] = np.quantile(np.concatenate(acc[s]), significance_level)
    # fill any untouched deep scales with the last valid value
    valid = ~np.isnan(sig)
    if valid.any():
        sig[~valid] = sig[valid][-1]
    return sig


# ---------------------------------------------------------------------------
# Cached pair runner
# ---------------------------------------------------------------------------
def _data_fingerprint(df: pd.DataFrame) -> str:
    """Stable digest of an aligned pair, including dates, values and columns."""
    h = hashlib.sha256()
    h.update(json.dumps(
        {"columns": list(df.columns), "dtypes": [str(d) for d in df.dtypes]},
        sort_keys=True,
    ).encode())
    hashed = pd.util.hash_pandas_object(df, index=True, categorize=True)
    h.update(hashed.to_numpy(dtype=np.uint64, copy=False).tobytes())
    return h.hexdigest()


def _cache_key(pair: tuple[str, str], df: pd.DataFrame, mc_count: int) -> Path:
    params = dict(version=CACHE_FORMAT_VERSION, pair=pair, n=len(df),
                  data=_data_fingerprint(df), mc=mc_count, dj=C.WAVELET.dj,
                  omega0=C.WAVELET.omega0, s0=C.WAVELET.s0_factor,
                  lvl=C.WAVELET.significance_level, seed=C.RANDOM_SEED,
                  pycwt=getattr(pycwt, "__version__", "unknown"))
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"wtc_{pair[0]}_{pair[1]}_{h}.npz"


def prune_pair_cache(pair: tuple[str, str], keep: int | None = None,
                     preserve: Path | None = None) -> list[Path]:
    """Delete stale cache files for a pair, retaining the newest ``keep``."""
    keep = C.WAVELET.cache_files_per_pair if keep is None else keep
    if keep < 0:
        raise ValueError("keep must be non-negative")

    prefix = f"wtc_{pair[0]}_{pair[1]}_"
    paths = sorted(
        CACHE_DIR.glob(f"{prefix}*.npz"),
        key=lambda p: (p == preserve, p.stat().st_mtime_ns),
        reverse=True,
    )
    removed = []
    for path in paths[keep:]:
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            logger.warning("could not prune cache %s: %r", path, exc)
    if removed:
        logger.info("WTC %s-%s pruned %d stale cache file(s)",
                    pair[0], pair[1], len(removed))
    return removed


def run_pair(returns: pd.DataFrame, pair: tuple[str, str],
             mc_count: int = None, force: bool = False) -> dict:
    """
    WTC + MC significance for a pair, cached on disk.

    Result dict adds: sig (per-scale threshold), sig_ratio = R2 / sig[:,None]
    (>1 == significant), and the cache path.
    """
    mc_count = mc_count or C.WAVELET.mc_count
    x, y = pair
    df = returns[[x, y]].dropna()
    path = _cache_key(pair, df, mc_count)

    if path.exists() and not force:
        z = np.load(path, allow_pickle=True)
        res = {k: z[k] for k in z.files}
        res["times"] = pd.DatetimeIndex(res["times"])
        res["pair"] = pair
        res["cache"] = path
        logger.info("WTC %s-%s loaded from cache (%s)", x, y, path.name)
        return res

    logger.info("WTC %s-%s: computing (n=%d, mc=%d)...", x, y, len(df), mc_count)
    res = wct_pair(df[x], df[y])
    sig = mc_significance(res["n"], res["ar1_x"], res["ar1_y"],
                          res["periods"], res["coi"], mc_count=mc_count)
    res["sig"] = sig
    res["sig_ratio"] = res["R2"] / sig[:, None]
    res["mc_count"] = mc_count

    np.savez_compressed(
        path,
        R2=res["R2"], phase=res["phase"], periods=res["periods"],
        coi=res["coi"], times=res["times"].values.astype("datetime64[ns]"),
        inside_coi=res["inside_coi"], sig=sig, sig_ratio=res["sig_ratio"],
        dt=res["dt"], dj=res["dj"], n=res["n"],
        ar1_x=res["ar1_x"], ar1_y=res["ar1_y"], mc_count=mc_count,
        cache_format=CACHE_FORMAT_VERSION,
    )
    prune_pair_cache(pair, preserve=path)
    res["pair"] = pair
    res["cache"] = path
    logger.info("WTC %s-%s saved -> %s", x, y, path.name)
    return res


# ---------------------------------------------------------------------------
# Summary metrics used by later phases
# ---------------------------------------------------------------------------
def significant_share(res: dict, t_slice: slice | None = None) -> float:
    """Share of (time x scale) cells inside the COI that are significant.
    The single-number 'how connected on all horizons' summary per window."""
    sig = res["sig_ratio"] > 1
    inside = res["inside_coi"].astype(bool)
    if t_slice is not None:
        sig = sig[:, t_slice]
        inside = inside[:, t_slice]
    denom = inside.sum()
    return float((sig & inside).sum() / denom) if denom else np.nan


def band_mean_coherence(res: dict, band: tuple[float, float]) -> pd.Series:
    """Mean R² over a period band (days), masked to inside-COI; time series."""
    lo, hi = band
    rows = (res["periods"] >= lo) & (res["periods"] <= hi)
    masked = np.where(res["inside_coi"], res["R2"], np.nan)
    return pd.Series(np.nanmean(masked[rows], axis=0),
                     index=pd.DatetimeIndex(res["times"]), name=f"R2_{lo:g}-{hi:g}d")


# ---------------------------------------------------------------------------
# Phase 3 - lead-lag from the phase difference
# ---------------------------------------------------------------------------
# Convention (Torrence & Webster / Grinsted, with W_xy = W_x · W_y*):
#   φ = 0      arrows →   in phase
#   φ = ±π     arrows ←   anti-phase
#   φ > 0                 X leads Y (for the pair (X, Y) as passed to run_pair)
#   φ < 0                 Y leads X
# Lag in days at a cell with period T: lag = φ/(2π) · T.
# IMPORTANT: phase gives lead-lag, NOT causality. A third common driver hitting
# the two assets with different speed produces the same picture.

PHASE_BANDS = {
    "короткий (2-8д)": (2.0, 8.0),
    "средний (8-32д)": (8.0, 32.0),
    "длинный (32-128д)": (32.0, 128.0),
}


def _circular_mean(phi: np.ndarray) -> tuple[float, float]:
    """Mean angle and resultant length R∈[0,1] (R→1 = phases tightly aligned)."""
    z = np.exp(1j * phi)
    m = z.mean()
    return float(np.angle(m)), float(np.abs(m))


def phase_summary(res: dict, band: tuple[float, float],
                  t_mask: np.ndarray | None = None,
                  only_significant: bool = True) -> dict:
    """
    Aggregate the phase difference inside one period band.

    Uses only cells inside the COI (and, by default, inside the 95% contour).
    A raw ``wct_pair`` result has no significance contour; in that case the
    function logs a warning and aggregates all inside-COI cells.
    Returns circular mean phase, concentration R, the share of cells in each
    qualitative regime, and the mean lead in days (positive => X leads Y).
    """
    periods = np.asarray(res["periods"])
    rows = (periods >= band[0]) & (periods <= band[1])
    mask = res["inside_coi"].astype(bool) & rows[:, None]
    if only_significant:
        if "sig_ratio" in res:
            mask &= np.asarray(res["sig_ratio"]) > 1
        else:
            logger.warning(
                "phase_summary: sig_ratio is absent; using all inside-COI cells"
            )
    if t_mask is not None:
        mask &= t_mask[None, :]

    phi = np.asarray(res["phase"])[mask]
    if phi.size < 30:
        return {"n_cells": int(phi.size), "valid": False}

    mean_phi, R = _circular_mean(phi)

    # qualitative buckets
    in_phase = float((np.abs(phi) < np.pi / 4).mean())
    anti_phase = float((np.abs(phi) > 3 * np.pi / 4).mean())
    x_leads = float(((phi > np.pi / 16) & (phi < 3 * np.pi / 4)).mean())
    y_leads = float(((phi < -np.pi / 16) & (phi > -3 * np.pi / 4)).mean())

    # mean lead in days computed cell-wise (phase scaled by its own period)
    T = np.broadcast_to(periods[:, None], res["phase"].shape)[mask]
    lead_days = float(np.mean(np.asarray(res["phase"])[mask] / (2 * np.pi) * T))

    return {
        "n_cells": int(phi.size), "valid": True,
        "mean_phase_deg": float(np.degrees(mean_phi)),
        "concentration_R": R,
        "share_in_phase": in_phase, "share_anti_phase": anti_phase,
        "share_x_leads": x_leads, "share_y_leads": y_leads,
        "mean_lead_days": lead_days,
    }


def phase_table(res: dict, bands: dict | None = None,
                t_mask: np.ndarray | None = None,
                only_significant: bool = True) -> pd.DataFrame:
    """phase_summary over all bands -> tidy DataFrame (index = band name)."""
    bands = bands or PHASE_BANDS
    rows = {}
    for name, band in bands.items():
        rows[name] = phase_summary(
            res, band, t_mask=t_mask, only_significant=only_significant
        )
    df = pd.DataFrame(rows).T
    return df
