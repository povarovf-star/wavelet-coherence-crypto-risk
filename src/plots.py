"""
Phase 2+ plotting: wavelet-coherence maps with COI, significance and phase arrows.

The canonical map (Grinsted 2004 style):
  * heatmap of R² in time x period axes, period on log2 scale (top = short
    horizons / days, bottom = long horizons / months),
  * hatched/shaded Cone of Influence - no interpretation outside it,
  * thick contour where R² exceeds the AR(1) Monte-Carlo 95% null,
  * phase arrows inside significant zones: → in-phase, ← anti-phase,
    ↓ X leads Y by 90°, ↑ Y leads X by 90° (pycwt/Grinsted convention),
  * red top-bars marking the project's stress episodes.
"""
from __future__ import annotations

import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

logger = logging.getLogger("plots")


def plot_coherence(res: dict, title: str | None = None,
                   arrow_density: tuple[int, int] = (60, 14),
                   save: str | None = None, ax=None):
    """
    Render a WTC result dict (from wavelet.run_pair) as the canonical map.

    arrow_density: approx number of arrows along (time, scale) axes.
    save: filename inside figures/ (e.g. '02_wtc_BTC_ETH.png').
    """
    R2 = res["R2"]
    phase = res["phase"]
    periods = np.asarray(res["periods"])
    coi = np.asarray(res["coi"])
    times = pd.DatetimeIndex(res["times"])
    sig_ratio = np.asarray(res["sig_ratio"])

    t = mdates.date2num(times.to_pydatetime())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
    else:
        fig = ax.figure

    # --- heatmap ----------------------------------------------------------
    im = ax.pcolormesh(t, periods, R2, vmin=0, vmax=1, cmap="jet",
                       shading="auto", rasterized=True)

    # --- significance contour (R2 / null95 > 1) ---------------------------
    ax.contour(t, periods, sig_ratio, levels=[1.0], colors="black",
               linewidths=1.4)

    # --- COI: shade the unreliable region ---------------------------------
    coi_clip = np.clip(coi, periods.min(), periods.max())
    ax.fill_between(t, coi_clip, periods.max(), color="white", alpha=0.45,
                    hatch="x", edgecolor="grey", linewidth=0.0)
    ax.plot(t, coi_clip, color="white", lw=1.0, ls="--")

    # --- phase arrows inside significant & inside-COI cells ---------------
    ns, nt = R2.shape
    step_t = max(nt // arrow_density[0], 1)
    step_s = max(ns // arrow_density[1], 1)
    sl_t = slice(step_t // 2, None, step_t)
    sl_s = slice(step_s // 2, None, step_s)
    show = (sig_ratio > 1) & res["inside_coi"].astype(bool)
    U = np.cos(phase)
    V = np.sin(phase)
    U = np.where(show, U, np.nan)[sl_s, sl_t]
    V = np.where(show, V, np.nan)[sl_s, sl_t]
    ax.quiver(t[sl_t], periods[sl_s], U, V, units="height", angles="uv",
              pivot="mid", scale=28, width=0.0022, headwidth=4,
              color="black", alpha=0.85)

    # --- episodes: red bars along the top edge ----------------------------
    ymin = periods.min()
    for ep in C.EPISODES:
        s, e = pd.Timestamp(ep.start), pd.Timestamp(ep.end)
        if s > times.max() or e < times.min():
            continue
        ax.axvspan(mdates.date2num(s), mdates.date2num(e),
                   ymin=0.965, ymax=1.0, color="red", alpha=0.9)
        ax.axvline(mdates.date2num(s), color="red", lw=0.5, alpha=0.35)

    # --- axes ---------------------------------------------------------------
    ax.set_yscale("log", base=2)
    ax.set_ylim(periods.max(), periods.min())   # short horizons on top
    yticks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    yticks = [y for y in yticks if periods.min() <= y <= periods.max()]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])
    ax.set_ylabel("период, дней (≈ горизонт)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    pair = res.get("pair", ("X", "Y"))
    if isinstance(pair, np.ndarray):
        pair = tuple(pair)
    ax.set_title(title or
                 f"Вейвлет-когерентность {pair[0]} ↔ {pair[1]} "
                 f"(контур - 95% против AR(1)-шума, штриховка - вне COI, "
                 f"красное - стресс-эпизоды)",
                 fontsize=11)

    if own_fig:
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.set_label("R²")
        fig.tight_layout()
    if save:
        path = C.FIGURES / save
        fig.savefig(path, dpi=140, bbox_inches="tight")
        logger.info("saved %s", path)
    return fig, ax
