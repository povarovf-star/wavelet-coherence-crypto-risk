"""
Data loading layer.

Design goal: the *same* call site works whether or not the machine has network
access. Each loader tries the live source, transparently caches raw pulls to
``data/raw/``, and - if the source is unreachable - falls back to a realistic
synthetic generator so the whole pipeline stays runnable end-to-end (e.g. in a
sandbox or CI). Every fallback is logged loudly so synthetic data is never
mistaken for the real thing.

Live sources (used on a networked machine):
    - all markets : TradingView WebSocket via the Node.js collector
    - crypto      : ccxt (Binance spot klines)
    - traditional : yfinance (^GSPC, GLD, DX-Y.NYB, ...)
    - long crypto : CoinGecko (pycoingecko)

Public API
----------
    load_prices(use_synthetic=None) -> pd.DataFrame
        Wide daily price frame, columns = ALL_COLS, DatetimeIndex.
    load_crypto(symbols, timeframe, start, end) -> pd.DataFrame
    load_traditional(tickers, start, end) -> pd.DataFrame
    make_synthetic_prices(...) -> pd.DataFrame
"""
from __future__ import annotations

import logging
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger("data_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
def _cache_path(kind: str, key: str, timeframe: str) -> Path:
    safe = key.replace("/", "-").replace("^", "").replace("=", "")
    return C.DATA_RAW / f"{kind}_{safe}_{timeframe}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:  # pragma: no cover
            logger.warning("cache read failed for %s: %r", path.name, e)
    return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception as e:  # pragma: no cover
        logger.warning("cache write failed for %s: %r", path.name, e)


# ---------------------------------------------------------------------------
# Live: TradingView via the Node.js collector
# ---------------------------------------------------------------------------
def _run_tradingview_collector(start: str, end: str | None) -> None:
    """Run the Node.js TradingView collector and write a wide close-price CSV."""
    cli = C.ROOT / "tradingview" / "cli.js"
    if not cli.exists():
        raise RuntimeError(f"TradingView collector not found: {cli}")

    cmd = [
        "node", str(cli), "history-all",
        "--start", start,
        "--output", str(C.TRADINGVIEW_PRICES_CSV),
        "--quiet",
    ]
    if end:
        cmd.extend(["--end", end])

    try:
        completed = subprocess.run(
            cmd,
            cwd=C.ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as e:
        raise RuntimeError("Node.js is required for the TradingView data source") from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"TradingView collector failed: {detail}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("TradingView collector timed out after 300 seconds") from e

    if completed.stderr.strip():
        logger.info("TradingView collector: %s", completed.stderr.strip())


def load_tradingview(start=None, end=None, refresh: bool = False) -> pd.DataFrame:
    """Load the research universe from TradingView, refreshing through Node.js."""
    start = start or C.START_DATE
    end = end or C.END_DATE

    if refresh or not C.TRADINGVIEW_PRICES_CSV.exists():
        _run_tradingview_collector(start, end)

    prices = pd.read_csv(C.TRADINGVIEW_PRICES_CSV, parse_dates=["date"])
    prices = prices.set_index("date").sort_index()

    missing = [c for c in C.ALL_COLS if c not in prices.columns]
    if missing:
        details = ""
        if C.TRADINGVIEW_METADATA.exists():
            try:
                metadata = json.loads(C.TRADINGVIEW_METADATA.read_text())
                details = f"; collector errors: {metadata.get('errors', {})}"
            except Exception:
                pass
        raise RuntimeError(f"TradingView load missing columns: {missing}{details}")

    prices = prices[C.ALL_COLS]
    logger.info("Loaded TradingView prices: %d rows x %d cols", *prices.shape)
    return prices


# ---------------------------------------------------------------------------
# Live: crypto via ccxt
# ---------------------------------------------------------------------------
def _fetch_ccxt_ohlcv(symbol: str, timeframe: str, start: str, end: str | None) -> pd.DataFrame:
    import ccxt  # imported lazily so the module loads without ccxt installed

    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end}T00:00:00Z") if end else exchange.milliseconds()
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    rows: list[list] = []
    cursor = since
    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
    if not rows:
        raise RuntimeError(f"ccxt returned no data for {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.set_index("date")["close"].rename(C.crypto_colname(symbol)).to_frame()


def load_crypto(symbols=None, timeframe=None, start=None, end=None) -> pd.DataFrame:
    """Daily close prices for crypto symbols via ccxt, cached per symbol."""
    symbols = symbols or C.CRYPTO_SYMBOLS
    timeframe = timeframe or C.TIMEFRAME
    start = start or C.START_DATE
    end = end or C.END_DATE

    series = []
    for sym in symbols:
        cache = _cache_path("crypto", sym, timeframe)
        cached = _read_cache(cache)
        if cached is not None:
            logger.info("crypto %s loaded from cache", sym)
            series.append(cached)
            continue
        col = _fetch_ccxt_ohlcv(sym, timeframe, start, end)
        _write_cache(col, cache)
        logger.info("crypto %s fetched live (%d bars)", sym, len(col))
        series.append(col)

    return pd.concat(series, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Live: traditional via yfinance
# ---------------------------------------------------------------------------
def load_traditional(tickers=None, start=None, end=None) -> pd.DataFrame:
    """Daily close prices for traditional assets via yfinance, cached per ticker."""
    import yfinance as yf  # lazy import

    tickers = tickers or C.TRADITIONAL_TICKERS
    start = start or C.START_DATE
    end = end or C.END_DATE

    series = []
    for name, tk in tickers.items():
        cache = _cache_path("trad", name, "1d")
        cached = _read_cache(cache)
        if cached is not None:
            logger.info("traditional %s loaded from cache", name)
            series.append(cached)
            continue
        raw = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        col = raw["Close"].copy()
        col.index = pd.to_datetime(col.index).normalize()
        col = col.rename(name).to_frame()
        _write_cache(col, cache)
        logger.info("traditional %s fetched live (%d bars)", name, len(col))
        series.append(col)

    return pd.concat(series, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def make_synthetic_prices(start=None, end=None, seed=C.RANDOM_SEED) -> pd.DataFrame:
    """
    Build a realistic daily price panel with *embedded contagion structure*.

    Construction:
      * One market factor (a proxy for systemic crypto risk) drives every asset
        with an asset-specific beta.
      * In calm times asset returns are mostly idiosyncratic (low correlation).
      * During each configured EPISODE the factor's volatility jumps and every
        asset's loading on it rises -> correlation spikes across the board, which
        is exactly the "diversification evaporates in a crisis" effect the
        project sets out to measure.
      * Stablecoins sit at ~1.0 with tiny noise, but de-peg sharply during the
        Terra and USDC/SVB windows.
      * Traditional assets share a smaller, partly-overlapping factor so crypto
        <-> macro coupling is weak in calm and stronger in COVID/March-2023.

    This is for development & end-to-end testing only. On a networked machine,
    ``load_prices`` prefers the live ccxt / yfinance sources.
    """
    start = start or C.START_DATE
    end = end or "2024-12-31"
    rng = np.random.default_rng(seed)

    # crypto trades 24/7 -> calendar-day index; we trim traditional to weekdays later
    idx = pd.bdate_range(start=start, end=end, freq="C", weekmask="Mon Tue Wed Thu Fri Sat Sun")
    n = len(idx)

    # --- episode mask & intensity -----------------------------------------
    stress = np.zeros(n)
    for ep in C.EPISODES:
        m = (idx >= pd.Timestamp(ep.start)) & (idx <= pd.Timestamp(ep.end))
        stress[m] = 1.0
    # smooth the stress envelope so spikes ramp in/out rather than switch
    kernel = np.ones(5) / 5
    stress = np.convolve(stress, kernel, mode="same")
    stress = np.clip(stress, 0, 1)

    # --- common crypto market factor --------------------------------------
    # Calm-period factor vol ~2.2%/day; rises ~2x in full stress. Multipliers
    # are deliberately moderate so calm vs stress regimes are distinct without
    # producing absurd (>15%/day, kurtosis>30) tails.
    base_vol = 0.022
    factor_vol = base_vol * (1 + 1.0 * stress)        # vol roughly doubles in stress
    factor = rng.normal(0, 1, n) * factor_vol
    factor += -0.025 * stress                         # crises drag prices down

    # --- macro factor (for traditional assets) ----------------------------
    macro_vol = 0.010 * (1 + 1.2 * stress)
    macro = rng.normal(0, 1, n) * macro_vol
    macro += -0.015 * stress

    # per-asset betas on the crypto factor; calm idio vol; drift
    crypto_spec = {
        "BTC": dict(beta=1.00, idio=0.012, drift=0.0040),
        "ETH": dict(beta=1.10, idio=0.015, drift=0.0042),
        "SOL": dict(beta=1.35, idio=0.026, drift=0.0050),
        "BNB": dict(beta=0.85, idio=0.016, drift=0.0038),
        "XRP": dict(beta=0.90, idio=0.022, drift=0.0034),
    }

    returns = {}
    for name, p in crypto_spec.items():
        # loading on the factor rises with stress (correlation contagion):
        # idiosyncratic share shrinks relative to the common factor -> pairwise
        # correlation climbs toward 1 exactly when it hurts most.
        load = p["beta"] * (1 + 0.7 * stress)
        idio = rng.normal(0, p["idio"], n) * (1 - 0.4 * stress)
        returns[name] = p["drift"] + load * factor + idio

    # --- stablecoins: ~peg with episode-specific de-pegs ------------------
    def stable_series(depeg_windows):
        r = rng.normal(0, 0.0004, n)  # micro noise around 0 return
        for (s, e, depth, recover) in depeg_windows:
            m = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
            days = np.where(m)[0]
            if len(days) == 0:
                continue
            # sharp drop then partial mean-reversion back toward peg
            shock = np.linspace(0, -depth, len(days))
            r[days] += shock / max(len(days), 1)
            # recovery just after the window
            rec_days = days[-1] + np.arange(1, min(recover, n - days[-1]))
            if len(rec_days):
                r[rec_days] += depth / max(len(rec_days), 1)
        return r

    # USDC de-pegs hard in Mar-2023; mild stress wobble during Terra
    returns["USDC"] = stable_series([
        ("2022-05-09", "2022-05-14", 0.010, 6),
        ("2023-03-10", "2023-03-13", 0.090, 8),
    ])
    # DAI is partly USDC-collateralised -> echoes USDC
    returns["DAI"] = stable_series([
        ("2023-03-10", "2023-03-13", 0.060, 8),
    ]) + 0.3 * returns["USDC"]

    # --- traditional assets ----------------------------------------------
    trad_spec = {
        "SP500": dict(beta_macro=1.0, beta_crypto=0.15, idio=0.008, drift=0.0004),
        "NASDAQ": dict(beta_macro=1.25, beta_crypto=0.22, idio=0.011, drift=0.0005),
        "GOLD": dict(beta_macro=-0.20, beta_crypto=0.05, idio=0.007, drift=0.0002),
        "DXY": dict(beta_macro=-0.35, beta_crypto=-0.08, idio=0.004, drift=0.0000),
    }
    for name, p in trad_spec.items():
        cross = p["beta_crypto"] * stress  # crypto<->macro link strengthens in stress
        idio = rng.normal(0, p["idio"], n)
        returns[name] = (p["drift"] + p["beta_macro"] * macro
                         + cross * factor + idio)

    # --- integrate returns into prices ------------------------------------
    ret_df = pd.DataFrame(returns, index=idx)[C.ALL_COLS]
    # stablecoins anchored at 1.0; others at sensible start levels
    start_levels = {
        "BTC": 4000, "ETH": 150, "SOL": 2.0, "BNB": 15, "XRP": 0.35,
        "USDC": 1.0, "DAI": 1.0,
        "SP500": 2500, "NASDAQ": 6700, "GOLD": 120, "DXY": 96,
    }
    prices = ret_df.copy()
    for col in prices.columns:
        prices[col] = start_levels[col] * np.exp(ret_df[col].cumsum())
    # re-anchor stables tightly to 1.0 (cumsum drift removed)
    for col in ("USDC", "DAI"):
        prices[col] = start_levels[col] * np.exp(ret_df[col].cumsum()
                                                 - ret_df[col].cumsum().mean())

    logger.warning("USING SYNTHETIC PRICES (no live data source reachable). "
                   "Do NOT use for real conclusions - for pipeline/dev only.")
    return prices.sort_index()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def load_prices(use_synthetic: bool | None = None,
                start=None, end=None, source: str = "auto",
                refresh: bool = False) -> pd.DataFrame:
    """
    Return a wide daily price frame (columns = C.ALL_COLS).

    source:
        'auto'        -> TradingView, then legacy ccxt/yfinance, then synthetic.
        'tradingview' -> TradingView only.
        'legacy'      -> ccxt/yfinance only.
        'synthetic'   -> deterministic development data.

    use_synthetic (deprecated compatibility flag):
        None  -> try live; on any failure, fall back to synthetic (default).
        True  -> force synthetic.
        False -> force live (raise if unreachable).
    """
    if use_synthetic is True:
        source = "synthetic"
    elif use_synthetic is False and source == "auto":
        source = "tradingview"

    if source == "synthetic":
        return make_synthetic_prices(start=start, end=end)

    if source in {"auto", "tradingview"}:
        try:
            return load_tradingview(start=start, end=end, refresh=refresh)
        except Exception as e:
            if source == "tradingview":
                raise
            warnings.warn(f"TradingView load failed ({e!r}); trying legacy sources.")
            logger.warning("TradingView load failed: %r -> legacy sources", e)

    if source not in {"auto", "legacy"}:
        raise ValueError("source must be 'auto', 'tradingview', 'legacy', or 'synthetic'")

    try:
        crypto = load_crypto(start=start, end=end)
        trad = load_traditional(start=start, end=end)
        prices = pd.concat([crypto, trad], axis=1).sort_index()
        missing = [c for c in C.ALL_COLS if c not in prices.columns]
        if missing:
            raise RuntimeError(f"missing columns from live load: {missing}")
        logger.info("Loaded LIVE prices: %d rows x %d cols", *prices.shape)
        return prices[C.ALL_COLS]
    except Exception as e:
        if source == "legacy":
            raise
        warnings.warn(f"Live load failed ({e!r}); falling back to synthetic data.")
        logger.warning("Live load failed: %r -> synthetic fallback", e)
        return make_synthetic_prices(start=start, end=end)


if __name__ == "__main__":
    df = load_prices()
    print(df.tail())
    print(df.describe().T[["mean", "std", "min", "max"]])
