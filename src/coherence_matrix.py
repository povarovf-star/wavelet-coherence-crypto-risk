"""
Stretch goal - full pairwise coherence matrix of the portfolio.

For every asset pair we compute the wavelet coherence and average R² inside the
COI, separately over calm windows and over the union of stress episodes. The
two matrices are the network view of contagion: in calm times crypto is a
moderately connected cluster loosely attached to macro; in stress the crypto
block saturates.

No Monte-Carlo here on purpose: averaging raw R² over thousands of cells is a
*descriptive* connectedness measure (pair ordering is what matters), while the
inferential heavy lifting (significance) already lives in Phase 2 for the key
pairs. This keeps the full 11x11 sweep to seconds instead of hours.

    python3 -m src.coherence_matrix
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C
from . import wavelet as W

logger = logging.getLogger("coherence_matrix")


def _mean_R2(res: dict, t_mask: np.ndarray | None = None) -> float:
    masked = np.where(res["inside_coi"].astype(bool), res["R2"], np.nan)
    if t_mask is not None:
        masked = masked[:, t_mask]
    return float(np.nanmean(masked))


def coherence_matrices(returns: pd.DataFrame,
                       cols: list[str] | None = None
                       ) -> dict[str, pd.DataFrame]:
    """Mean inside-COI R² per pair, for calm and stress sub-periods."""
    cols = cols or [c for c in C.ALL_COLS if c in returns.columns]
    n = len(cols)
    calm = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    stress = pd.DataFrame(np.eye(n), index=cols, columns=cols)

    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            b = cols[j]
            try:
                res = W.wct_pair(returns[a], returns[b])
            except ValueError as e:           # too little overlap
                logger.warning("skip %s-%s: %s", a, b, e)
                calm.loc[a, b] = calm.loc[b, a] = np.nan
                stress.loc[a, b] = stress.loc[b, a] = np.nan
                continue
            times = pd.DatetimeIndex(res["times"])
            m_stress = np.zeros(len(times), bool)
            for ep in C.EPISODES:
                m_stress |= np.asarray((times >= ep.start) & (times <= ep.end))
            m_calm = np.zeros(len(times), bool)
            for s, e in C.CALM_WINDOWS:
                m_calm |= np.asarray((times >= s) & (times <= e))
            m_calm &= ~m_stress

            calm.loc[a, b] = calm.loc[b, a] = _mean_R2(res, m_calm)
            stress.loc[a, b] = stress.loc[b, a] = _mean_R2(res, m_stress)
        logger.info("row %s done", a)

    return {"calm": calm, "stress": stress}


def plot_matrices(mats: dict[str, pd.DataFrame], save: str = "05_coherence_matrix.png"):
    import matplotlib.pyplot as plt

    cols = mats["calm"].columns
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, key, ttl in [(axes[0], "calm", "Спокойные окна"),
                         (axes[1], "stress", "Стресс-эпизоды")]:
        M = mats[key].values
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        off = M[np.triu_indices(len(cols), 1)]
        ax.set_title(f"{ttl}: средняя R² (внутри COI) = {np.nanmean(off):.2f}")
        for i in range(len(cols)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=axes, fraction=0.03)
    path = C.FIGURES / save
    fig.savefig(path, dpi=140, bbox_inches="tight")
    logger.info("saved %s", path)
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    returns = pd.read_parquet(C.PROCESSED_RETURNS)
    mats = coherence_matrices(returns)
    mats["calm"].round(3).to_csv(C.RESULTS / "phase5_coherence_calm.csv")
    mats["stress"].round(3).to_csv(C.RESULTS / "phase5_coherence_stress.csv")
    plot_matrices(mats)

    crypto = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    iu = np.triu_indices(5, 1)
    print("crypto block mean R²: calm =",
          round(float(np.nanmean(mats['calm'].loc[crypto, crypto].values[iu])), 3),
          "| stress =",
          round(float(np.nanmean(mats['stress'].loc[crypto, crypto].values[iu])), 3))


if __name__ == "__main__":
    main()
