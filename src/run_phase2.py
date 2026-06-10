"""
Phase 2 runner: wavelet coherence + MC significance for the key pairs.

    python3 -m src.run_phase2                 # default pairs, mc from config
    python3 -m src.run_phase2 --mc 300        # faster dev run
    python3 -m src.run_phase2 --pairs BTC:ETH BTC:NASDAQ

Results are cached in data/processed/wavelet/ and figures in figures/02_*.png.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from . import config as C
from . import wavelet as W
from . import plots as P

logger = logging.getLogger("run_phase2")

DEFAULT_PAIRS = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "NASDAQ"), ("BTC", "GOLD")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2: wavelet coherence maps")
    ap.add_argument("--mc", type=int, default=None, help="MC surrogate count")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="pairs as A:B (default: key four)")
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    pairs = ([tuple(p.split(":")) for p in args.pairs]
             if args.pairs else DEFAULT_PAIRS)

    returns = pd.read_parquet(C.PROCESSED_RETURNS)
    rows = []
    for pair in pairs:
        res = W.run_pair(returns, pair, mc_count=args.mc, force=args.force)
        P.plot_coherence(res, save=f"02_wtc_{pair[0]}_{pair[1]}.png")
        share_all = W.significant_share(res)
        rows.append({"pair": f"{pair[0]}-{pair[1]}",
                     "n": int(res["n"]),
                     "mc": int(res["mc_count"]),
                     "sig_share_total": round(share_all, 4)})
        logger.info("%s: significant share inside COI = %.1f%%",
                    pair, share_all * 100)

    summary = pd.DataFrame(rows)
    out = C.RESULTS / "phase2_sig_share.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
