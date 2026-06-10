"""Fast checks for the scientific invariants claimed in the README."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pywt

from src.risk import swt_decompose
from src.wavelet import (
    _cache_key,
    phase_summary,
    phase_table,
    prune_pair_cache,
    wct_pair,
)


class PhaseConventionTests(unittest.TestCase):
    def test_positive_phase_means_first_series_leads(self) -> None:
        n = 512
        period_days = 32.0
        known_lead_days = 4.0
        t = np.arange(n)
        index = pd.date_range("2020-01-01", periods=n)
        x = pd.Series(np.sin(2 * np.pi * t / period_days), index=index)
        y = pd.Series(
            np.sin(2 * np.pi * (t - known_lead_days) / period_days),
            index=index,
        )

        raw = wct_pair(x, y)
        with self.assertLogs("wavelet", level="WARNING"):
            summary = phase_summary(raw, (24.0, 40.0))

        self.assertTrue(summary["valid"])
        self.assertAlmostEqual(summary["mean_phase_deg"], 45.0, delta=2.0)
        self.assertAlmostEqual(summary["mean_lead_days"], known_lead_days, delta=0.5)

    def test_phase_table_accepts_raw_wct_result(self) -> None:
        n = 256
        t = np.arange(n)
        index = pd.date_range("2020-01-01", periods=n)
        x = pd.Series(np.sin(2 * np.pi * t / 16), index=index)
        y = pd.Series(np.sin(2 * np.pi * (t - 2) / 16), index=index)

        raw = wct_pair(x, y)
        with self.assertLogs("wavelet", level="WARNING"):
            table = phase_table(raw, bands={"16-day cycle": (12.0, 20.0)})

        self.assertTrue(bool(table.loc["16-day cycle", "valid"]))


class StationaryWaveletTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.values = rng.normal(size=1024)
        self.series = pd.Series(self.values)

    def test_swt_parseval_variance_partition(self) -> None:
        bands = swt_decompose(self.series, levels=6, wavelet="db4")
        ratio = bands.var().sum() / self.series.var()

        self.assertAlmostEqual(float(ratio), 1.0, places=12)

    def test_swt_iswt_reconstruction(self) -> None:
        coeffs = pywt.swt(self.values, "db4", level=6, norm=True)
        reconstructed = pywt.iswt(coeffs, "db4", norm=True)

        np.testing.assert_allclose(reconstructed, self.values, rtol=1e-12, atol=1e-12)


class WaveletCacheTests(unittest.TestCase):
    def test_cache_key_changes_when_same_length_input_changes(self) -> None:
        index = pd.date_range("2020-01-01", periods=4)
        first = pd.DataFrame({"BTC": [1.0, 2.0, 3.0, 4.0],
                              "ETH": [4.0, 3.0, 2.0, 1.0]}, index=index)
        changed = first.copy()
        changed.iloc[0, 0] = 99.0
        shifted = first.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)

        first_key = _cache_key(("BTC", "ETH"), first, mc_count=10)
        changed_key = _cache_key(("BTC", "ETH"), changed, mc_count=10)
        shifted_key = _cache_key(("BTC", "ETH"), shifted, mc_count=10)

        self.assertNotEqual(first_key, changed_key)
        self.assertNotEqual(first_key, shifted_key)

    def test_cache_pruning_preserves_current_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"wtc_BTC_ETH_{i}.npz" for i in range(3)]
            other = root / "wtc_BTC_SOL_0.npz"
            for path in [*paths, other]:
                path.touch()
            for i, path in enumerate(paths):
                path.touch()
                path.chmod(0o600)
                path.write_bytes(bytes([i]))

            with patch("src.wavelet.CACHE_DIR", root):
                removed = prune_pair_cache(
                    ("BTC", "ETH"), keep=1, preserve=paths[0]
                )

            self.assertEqual(len(removed), 2)
            self.assertEqual(list(root.glob("wtc_BTC_ETH_*.npz")), [paths[0]])
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
