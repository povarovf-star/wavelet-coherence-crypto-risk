"""
Preprocessing: prices -> clean, aligned log-returns.

Why log-returns and not prices:
    Price levels are non-stationary (random-walk-ish); feeding them to a wavelet
    would mostly pick up the trend. Log-returns r_t = ln(P_t / P_{t-1}) are far
    closer to stationary and additive across time, which is what CWT assumes.

The 24/7 nuance:
    Crypto trades every calendar day; equities/gold/DXY only on weekdays. For
    *cross* pairs we must put both on the same clock. ``align_calendar`` offers
    two honest modes:
      - 'trading_days' : restrict crypto to traditional trading days (use for
                          crypto<->macro pairs).
      - 'crypto'       : keep the full 24/7 crypto calendar, forward-fill the
                          traditional series (only safe for crypto-only work;
                          flagged because it injects zero-return weekend bars).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------
def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log-returns; first row (NaN) dropped."""
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"non-positive prices in {bad}; cannot take log")
    r = np.log(prices).diff()
    return r.iloc[1:]


# ---------------------------------------------------------------------------
# Calendar alignment
# ---------------------------------------------------------------------------
def align_calendar(prices: pd.DataFrame, mode: str = "trading_days") -> pd.DataFrame:
    """
    Put crypto and traditional columns on one calendar.

    'trading_days' (default): keep only weekdays (Mon-Fri). Traditional assets
        define trading days; crypto is sampled on those same days. This is the
        correct choice whenever a traditional asset is part of the pair.
    'crypto': keep every calendar day, forward-fill traditional columns over
        weekends/holidays. Convenient for crypto-only analysis but it fabricates
        flat weekend bars for macro assets - never use it to draw crypto<->macro
        conclusions.
    """
    crypto_cols = [c for c in C.CRYPTO_COLS if c in prices.columns]
    trad_cols = [c for c in C.TRADITIONAL_COLS if c in prices.columns]

    if mode == "trading_days":
        weekday = prices[prices.index.dayofweek < 5].copy()
        # drop rows where traditional data is entirely missing (holidays)
        if trad_cols:
            weekday = weekday.dropna(subset=trad_cols, how="all")
        out = weekday
    elif mode == "crypto":
        out = prices.copy()
        if trad_cols:
            out[trad_cols] = out[trad_cols].ffill()
            logger.warning("mode='crypto': traditional columns forward-filled over "
                           "non-trading days - do not use for crypto<->macro claims.")
    else:
        raise ValueError("mode must be 'trading_days' or 'crypto'")

    return out.sort_index()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, winsor_q: float = C.WINSOR_QUANTILE,
          winsorize: bool = True, winsor_exempt: list[str] | None = None,
          dropna: str = "any") -> pd.DataFrame:
    """
    Clean a returns (or price) frame:
      * drop duplicate index rows (keep last),
      * sort by time,
      * report & forward-fill short internal gaps (<=2 days),
      * drop rows according to ``dropna`` ('any' for complete-case, 'all' for
        panel data where assets have different listing dates),
      * optionally winsorize the extreme winsor_q tails per column (clip, don't
        delete - a true crash is signal, not error, so we only tame the very
        fattest outliers symmetrically).

    winsor_exempt: columns NEVER winsorized. Stablecoins live here on purpose:
        their defining events are de-pegs (a -9% day on USDC in Mar-2023), which
        a global winsorizer would flatten into noise. Clipping them would erase
        the exact phenomenon the project studies - so we keep their tails intact.
    """
    out = df[~df.index.duplicated(keep="last")].sort_index()

    # gap report
    n_missing = out.isna().sum()
    if n_missing.any():
        logger.info("missing values per column before fill:\n%s",
                    n_missing[n_missing > 0].to_string())
    # only fill very short gaps; leave long holes to be dropped
    out = out.ffill(limit=2)
    if dropna not in {"any", "all", "none"}:
        raise ValueError("dropna must be 'any', 'all', or 'none'")

    before = len(out)
    if dropna != "none":
        out = out.dropna(how=dropna)
    if len(out) < before:
        logger.info("dropped %d rows with unfilled gaps", before - len(out))

    if winsorize:
        exempt = set(winsor_exempt or [])
        cols = [c for c in out.columns if c not in exempt]
        if cols:
            lo = out[cols].quantile(winsor_q)
            hi = out[cols].quantile(1 - winsor_q)
            out[cols] = out[cols].clip(lower=lo, upper=hi, axis=1)
        if exempt:
            logger.info("winsorization skipped for: %s (de-peg tails preserved)",
                        sorted(exempt & set(out.columns)))

    return out


# ---------------------------------------------------------------------------
# One-shot pipeline
# ---------------------------------------------------------------------------
def build_returns(prices: pd.DataFrame, mode: str = "trading_days",
                  winsorize: bool = True, dropna: str = "all") -> pd.DataFrame:
    """prices -> aligned, cleaned log-returns (the Phase-0 deliverable)."""
    aligned = align_calendar(prices, mode=mode)
    aligned = clean(aligned, winsorize=False,          # clean prices (no winsor on levels)
                    dropna=dropna)
    returns = to_log_returns(aligned)
    returns = clean(returns, winsorize=winsorize,      # winsorize returns...
                    winsor_exempt=C.STABLE_COLS,       # ...but never the stablecoins
                    dropna=dropna)
    logger.info("returns built: %d rows x %d cols (%s .. %s)",
                *returns.shape, returns.index.min().date(), returns.index.max().date())
    return returns


# ---------------------------------------------------------------------------
# Sanity statistics
# ---------------------------------------------------------------------------
def sanity_stats(returns: pd.DataFrame, periods_per_year: int = 365) -> pd.DataFrame:
    """Per-column summary: n, missing share, mean, std, skew, excess kurtosis,
    annualised vol, min/max."""
    from scipy.stats import kurtosis, skew

    stats = pd.DataFrame({
        "n_obs": returns.count(),
        "missing_%": (returns.isna().mean() * 100).round(3),
        "mean": returns.mean(),
        "std": returns.std(),
        "ann_vol": returns.std() * np.sqrt(periods_per_year),
        "skew": returns.apply(lambda s: skew(s.dropna())),
        "excess_kurt": returns.apply(lambda s: kurtosis(s.dropna())),  # Fisher (excess)
        "min": returns.min(),
        "max": returns.max(),
    })
    return stats
