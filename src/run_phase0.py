"""
Phase 0 runner: load -> align/clean -> log-returns -> save + sanity report.

Run from the repo root:
    python3 -m src.run_phase0                # try TradingView, fall back to synthetic
    python3 -m src.run_phase0 --synthetic    # force synthetic
    python3 -m src.run_phase0 --live         # force fresh TradingView data

Outputs:
    data/processed/prices.parquet
    data/processed/returns.parquet
    results/phase0_sanity.csv
"""
from __future__ import annotations

import argparse
import logging

from . import config as C
from . import data_loader as dl
from . import preprocess as pp

logger = logging.getLogger("run_phase0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0: data & sanity check")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true", help="force synthetic data")
    g.add_argument("--live", action="store_true",
                   help="force a fresh TradingView load (no fallback)")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "tradingview", "legacy", "synthetic"],
                    help="price source; auto prefers TradingView")
    ap.add_argument("--refresh", action="store_true",
                    help="refresh TradingView raw CSV even if a cache exists")
    ap.add_argument("--mode", default="trading_days",
                    choices=["trading_days", "crypto"],
                    help="calendar alignment for cross-asset work")
    ap.add_argument("--dropna", default="all", choices=["all", "any", "none"],
                    help="'all' preserves assets with shorter histories; "
                         "'any' is complete-case")
    args = ap.parse_args()

    source = "synthetic" if args.synthetic else ("tradingview" if args.live else args.source)
    refresh = args.refresh or args.live

    prices = dl.load_prices(source=source, refresh=refresh)
    prices.to_parquet(C.PROCESSED_PRICES)
    logger.info("saved prices -> %s", C.PROCESSED_PRICES)

    returns = pp.build_returns(prices, mode=args.mode, winsorize=True,
                               dropna=args.dropna)
    returns.to_parquet(C.PROCESSED_RETURNS)
    logger.info("saved returns -> %s", C.PROCESSED_RETURNS)

    periods_per_year = 252 if args.mode == "trading_days" else 365
    stats = pp.sanity_stats(returns, periods_per_year=periods_per_year)
    out_csv = C.RESULTS / "phase0_sanity.csv"
    stats.to_csv(out_csv)
    logger.info("saved sanity stats -> %s", out_csv)

    print("\n=== Phase 0 sanity statistics ===")
    with __import__("pandas").option_context("display.width", 160,
                                             "display.max_columns", 20):
        print(stats.round(4))
    print(f"\nrows: {len(returns)}  | period: {returns.index.min().date()} "
          f".. {returns.index.max().date()}")


if __name__ == "__main__":
    main()
