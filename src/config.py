"""
Central configuration: assets, date range, stress episodes, wavelet & risk params.

Everything that another module might need to agree on lives here, so the rest of
the pipeline never hard-codes a ticker or a date window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

PROCESSED_RETURNS = DATA_PROCESSED / "returns.parquet"
PROCESSED_PRICES = DATA_PROCESSED / "prices.parquet"
TRADINGVIEW_RAW = DATA_RAW / "tradingview"
TRADINGVIEW_PRICES_CSV = TRADINGVIEW_RAW / "prices.csv"
TRADINGVIEW_METADATA = TRADINGVIEW_RAW / "metadata.json"

# ---------------------------------------------------------------------------
# Sample window
# ---------------------------------------------------------------------------
# Start before the COVID crash so we capture every killer episode; end is open
# (None -> "today" at load time).
START_DATE = "2019-01-01"
END_DATE = None  # None => up to the latest available bar

# Daily bars are the workhorse; hourly is available for zoom-ins on episodes.
TIMEFRAME = "1d"

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
# Crypto symbols are given in ccxt "BASE/QUOTE" form (USDT-quoted spot).
CRYPTO_CORE = ["BTC/USDT", "ETH/USDT"]
CRYPTO_ALTS = ["SOL/USDT", "BNB/USDT", "XRP/USDT"]
# Stablecoins quoted vs USD to study the peg itself (USDT/USDC against fiat USD).
STABLES = ["USDC/USDT", "DAI/USDT"]  # USDT≈peg reference; these track de-pegs

CRYPTO_SYMBOLS = CRYPTO_CORE + CRYPTO_ALTS + STABLES

# Traditional / cross-asset contour (yfinance tickers).
TRADITIONAL_TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "GOLD": "GLD",      # SPDR Gold Shares as gold proxy
    "DXY": "DX-Y.NYB",  # US Dollar Index
}

# TradingView symbols used by the Node.js collector in ``tradingview/``.
TRADINGVIEW_SYMBOLS = {
    "BTC": "BINANCE:BTCUSDT",
    "ETH": "BINANCE:ETHUSDT",
    "SOL": "BINANCE:SOLUSDT",
    "BNB": "BINANCE:BNBUSDT",
    "XRP": "BINANCE:XRPUSDT",
    "USDC": "BINANCE:USDCUSDT",
    "DAI": "KRAKEN:DAIUSDT",
    "SP500": "SP:SPX",
    "NASDAQ": "NASDAQ:IXIC",
    "GOLD": "AMEX:GLD",
    "DXY": "TVC:DXY",
}

# Friendly column names used everywhere downstream.
def crypto_colname(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'. Stable pairs keep the base ('USDC', 'DAI')."""
    return symbol.split("/")[0]


CRYPTO_COLS = [crypto_colname(s) for s in CRYPTO_SYMBOLS]
CRYPTO_CORE_COLS = [crypto_colname(s) for s in CRYPTO_CORE]  # ['BTC', 'ETH']
STABLE_COLS = [crypto_colname(s) for s in STABLES]  # ['USDC', 'DAI']
TRADITIONAL_COLS = list(TRADITIONAL_TICKERS.keys())
ALL_COLS = CRYPTO_COLS + TRADITIONAL_COLS

# ---------------------------------------------------------------------------
# Key pairs to analyse (drives the wavelet runs in later phases)
# ---------------------------------------------------------------------------
KEY_PAIRS = [
    ("BTC", "ETH"),     # diversification illusion inside crypto
    ("BTC", "SOL"),
    ("BTC", "BNB"),
    ("BTC", "XRP"),
    ("BTC", "NASDAQ"),  # "risk asset?"
    ("BTC", "GOLD"),    # "digital gold?"
    ("BTC", "DXY"),     # dollar link
    ("USDC", "USDT_PEG"),  # handled specially: USDC vs its own peg
    ("USDC", "DAI"),
]

# ---------------------------------------------------------------------------
# Stress / contagion episodes (natural stress windows)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    name: str
    start: str
    end: str
    note: str


EPISODES: list[Episode] = [
    Episode("COVID crash", "2020-03-01", "2020-04-15",
            "crypto falls with everything -> not a safe haven"),
    Episode("Terra/LUNA + UST de-peg", "2022-05-05", "2022-05-20",
            "systemic stablecoin collapse"),
    Episode("3AC / Celsius", "2022-06-10", "2022-07-05",
            "leverage-driven contagion wave"),
    Episode("FTX collapse", "2022-11-06", "2022-11-21",
            "exchange failure, market-wide contagion"),
    Episode("USDC de-peg / SVB", "2023-03-08", "2023-03-17",
            "crypto's link to banking risk"),
]

# Calm reference windows (used to estimate the 'normal' covariance regime).
CALM_WINDOWS = [
    ("2019-06-01", "2019-12-31"),
    ("2021-04-01", "2021-10-31"),
    ("2023-07-01", "2024-06-30"),
]

# ---------------------------------------------------------------------------
# Wavelet parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaveletParams:
    mother: str = "morlet"   # complex Morlet - phase is required
    omega0: float = 6.0      # standard non-dimensional frequency (Grinsted 2004)
    dt: float = 1.0          # 1 bar = 1 day
    dj: float = 1 / 12       # 12 sub-octaves per octave
    s0_factor: float = 2.0   # smallest scale = s0_factor * dt
    mc_count: int = 500      # Monte-Carlo surrogates for significance
    # 500 gives ~25k pooled null draws per scale - ample for a stable 95%
    # quantile; raise to 1000 locally for the final publication run.
    significance_level: float = 0.95
    cache_files_per_pair: int = 1  # keep the newest input, prune stale results


WAVELET = WaveletParams()

# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskParams:
    var_levels: tuple[float, ...] = (0.95, 0.99)
    modwt_levels: int = 6           # frequency bands for scale-dependent VaR
    modwt_wavelet: str = "db4"      # Daubechies-4 for MODWT decomposition
    portfolio_weights: dict[str, float] = field(default_factory=lambda: {
        # equal-ish risk-asset book; stables excluded from the risk book
        "BTC": 0.35, "ETH": 0.25, "SOL": 0.15, "BNB": 0.15, "XRP": 0.10,
    })


RISK = RiskParams()

# ---------------------------------------------------------------------------
# Cleaning parameters
# ---------------------------------------------------------------------------
WINSOR_QUANTILE = 0.001  # clip the most extreme 0.1% of returns on each tail
RANDOM_SEED = 42
