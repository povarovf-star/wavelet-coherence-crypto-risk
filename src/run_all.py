"""
One-command reproduction of the whole pipeline (Phases 0-4).

    python3 -m src.run_all                # use cached raw data if present
    python3 -m src.run_all --live         # refresh TradingView data first
    python3 -m src.run_all --synthetic    # fully offline demo run

Steps:
    0. data  -> data/processed/returns.parquet, results/phase0_sanity.csv
    1. bench -> figures/01_*.png, results/phase1_corr_by_episode.csv
    2. WTC   -> figures/02_wtc_*.png, results/phase2_sig_share.csv (+ cache)
    3. phase -> results/phase3_leadlag.csv
    4. risk  -> results/var_es_table.csv, results/phase4_scale_var.csv,
                results/phase4_indicator_check.csv, figures/04_*.png

Wavelet MC runs are cached in data/processed/wavelet/, so re-runs are fast.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("run_all")

KEY_WTC_PAIRS = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "NASDAQ"),
                 ("BTC", "GOLD"), ("USDC", "DAI")]


def phase0(source: str, refresh: bool) -> pd.DataFrame:
    from . import data_loader as dl
    from . import preprocess as pp

    logger.info("=== Phase 0: data ===")
    prices = dl.load_prices(source=source, refresh=refresh)
    prices.to_parquet(C.PROCESSED_PRICES)
    returns = pp.build_returns(prices, mode="trading_days", winsorize=True,
                               dropna="all")
    returns.to_parquet(C.PROCESSED_RETURNS)
    pp.sanity_stats(returns).to_csv(C.RESULTS / "phase0_sanity.csv")
    return returns


def phase1(returns: pd.DataFrame) -> None:
    from . import benchmark as B

    logger.info("=== Phase 1: benchmark ===")
    for pair in [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "NASDAQ"), ("BTC", "GOLD")]:
        B.plot_benchmark_pair(returns, pair)
    mpc = B.mean_pairwise_rolling_corr(returns, ["BTC", "ETH", "SOL", "BNB", "XRP"])
    rows, union = [], pd.Index([])
    for ep in C.EPISODES:
        m = (mpc.index >= ep.start) & (mpc.index <= ep.end)
        union = union.union(mpc.index[m])
        rows.append({"эпизод": ep.name, "средняя corr": round(float(mpc[m].mean()), 3)})
    rows.append({"эпизод": "- вне эпизодов -",
                 "средняя corr": round(float(mpc.drop(index=union).mean()), 3)})
    pd.DataFrame(rows).to_csv(C.RESULTS / "phase1_corr_by_episode.csv", index=False)


def phase2(returns: pd.DataFrame) -> dict:
    from . import wavelet as W
    from . import plots as P

    logger.info("=== Phase 2: wavelet coherence (cached MC) ===")
    results, rows = {}, []
    for pair in KEY_WTC_PAIRS:
        res = W.run_pair(returns, pair)          # cache-aware
        P.plot_coherence(res, save=f"02_wtc_{pair[0]}_{pair[1]}.png")
        results[pair] = res
        rows.append({"pair": f"{pair[0]}-{pair[1]}", "n": int(res["n"]),
                     "mc": int(res["mc_count"]),
                     "sig_share_total": round(W.significant_share(res), 4)})
    pd.DataFrame(rows).to_csv(C.RESULTS / "phase2_sig_share.csv", index=False)
    return results


def phase3(results: dict) -> None:
    from . import wavelet as W

    logger.info("=== Phase 3: lead-lag phases ===")
    frames = []
    for pair, res in results.items():
        t = W.phase_table(res)
        t.insert(0, "pair", f"{pair[0]}-{pair[1]}")
        frames.append(t)
    pd.concat(frames).to_csv(C.RESULTS / "phase3_leadlag.csv")


def phase4(returns: pd.DataFrame) -> None:
    from . import risk as R

    logger.info("=== Phase 4: risk bridge ===")
    port = R.portfolio_returns(returns)
    R.scale_var_table(port).to_csv(C.RESULTS / "phase4_scale_var.csv")
    tbl, _ = R.stress_vs_normal_table(returns)
    tbl.round(3).to_csv(C.RESULTS / "var_es_table.csv")
    signal = R.connectedness_signal(returns)
    pd.Series(R.indicator_check(signal, port)).to_csv(
        C.RESULTS / "phase4_indicator_check.csv")


EXPECTED_ARTIFACTS = [
    C.PROCESSED_RETURNS,
    C.RESULTS / "phase0_sanity.csv",
    C.RESULTS / "phase1_corr_by_episode.csv",
    C.RESULTS / "phase2_sig_share.csv",
    C.RESULTS / "phase3_leadlag.csv",
    C.RESULTS / "phase4_scale_var.csv",
    C.RESULTS / "var_es_table.csv",
    C.RESULTS / "phase4_indicator_check.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole pipeline (Phases 0-4)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="refresh TradingView data")
    g.add_argument("--synthetic", action="store_true", help="offline demo run")
    args = ap.parse_args()

    source = "synthetic" if args.synthetic else ("tradingview" if args.live else "auto")
    returns = phase0(source, refresh=args.live)
    phase1(returns)
    results = phase2(returns)
    phase3(results)
    phase4(returns)

    missing = [p for p in EXPECTED_ARTIFACTS if not p.exists()]
    if missing:
        logger.error("MISSING artifacts: %s", [str(m) for m in missing])
        sys.exit(1)
    logger.info("ALL PHASES DONE - %d artifacts in place.", len(EXPECTED_ARTIFACTS))
    print("\nFinal VaR/ES table (results/var_es_table.csv):")
    print(pd.read_csv(C.RESULTS / "var_es_table.csv", index_col=0).round(2).to_string())


if __name__ == "__main__":
    main()
