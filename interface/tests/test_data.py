"""Meaningful invariants for source timing, holiday logic and forecast leakage."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from flowpilot.data import (
    AXES, PARIS, _forecast_from_history, get_forecast, is_holiday,
    load_history, load_quality, prepare_map_rows, resolve_data_root,
)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_history()
        cls.forecast = _forecast_from_history(date(2026, 11, 7), cls.history)

    def test_data_root_supports_checkout_without_hiding_missing_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "upstream" / "datathon_bgcx"
            (nested / "data" / "processed_data").mkdir(parents=True)
            self.assertEqual(resolve_data_root(root), nested)
            # A fresh GitHub checkout stores the data directly beside app.py.
            # Its own directory wins even if the expected CSVs are not present,
            # rather than silently borrowing potentially different nested data.
            (root / "data" / "processed_data").mkdir(parents=True)
            self.assertEqual(resolve_data_root(root), root)

    def test_forecast_has_each_real_hour_and_valid_units(self):
        frame = self.forecast
        self.assertEqual(len(frame), 72)
        self.assertEqual(set(frame.axe), set(AXES))
        self.assertTrue(frame.groupby("axe").size().eq(24).all())
        self.assertTrue(frame.occupation.between(0, 100).all())
        self.assertTrue(frame.debit.ge(0).all())
        self.assertTrue(frame.lower_occupation.le(frame.occupation).all())
        self.assertTrue(frame.upper_occupation.ge(frame.occupation).all())
        self.assertTrue(frame.samples.gt(0).all())

    def test_source_end_hour_is_converted_to_start(self):
        first = self.history[self.history.axe.eq("Convention")].iloc[0]
        self.assertEqual(first.datetime, pd.Timestamp("2014-01-01 00:00", tz=PARIS))
        # CSV ends 01:00 Paris; the measured delivery slot starts 00:00.
        self.assertEqual(first.debit, 205.0)

    def test_invalid_sensor_is_missing_not_free_flow(self):
        recent = self.history[
            self.history.axe.eq("Champs-Elysees") &
            self.history.datetime.ge(pd.Timestamp("2026-10-01", tz=PARIS))
        ]
        self.assertGreater(len(recent), 0)
        self.assertTrue(recent[["debit", "occupation"]].isna().all().all())
        self.assertGreater(self.forecast[self.forecast.axe.eq("Champs-Elysees")].stale_days.min(), 300)

    def test_quality_and_forecast_metadata_describe_same_last_reading(self):
        quality = load_quality().set_index("axe")
        last_forecast = self.forecast.groupby("axe").last_observation.first()
        for axe in AXES:
            self.assertEqual(quality.loc[axe, "last_observation"], last_forecast.loc[axe])
            axis_history = self.history[self.history.axe.eq(axe)]
            complete = axis_history[["debit", "occupation"]].notna().all(axis=1)
            self.assertEqual(quality.loc[axe, "n_observations"], int(complete.sum()))
            self.assertAlmostEqual(quality.loc[axe, "missing_pct"], 100 * (1 - complete.mean()))

    def test_french_holidays_include_armistice_and_easter(self):
        self.assertTrue(is_holiday(date(2026, 11, 11)))
        self.assertFalse(is_holiday(date(2026, 11, 10)))
        self.assertTrue(is_holiday(date(2027, 3, 29)))
        self.assertFalse(is_holiday(date(2027, 3, 30)))

    def test_predictions_do_not_read_future_observations(self):
        target = date(2025, 11, 7)
        original = _forecast_from_history(target, self.history)
        altered = self.history.copy()
        future = altered.datetime.ge(pd.Timestamp(target, tz=PARIS))
        altered.loc[future, "occupation"] = 99.0
        altered.loc[future, "debit"] = 11_999.0
        comparison = _forecast_from_history(target, altered)
        pd.testing.assert_frame_equal(original, comparison)

    def test_dst_day_has_real_hours_and_map_uses_exact_arc(self):
        spring = _forecast_from_history(date(2026, 3, 29), self.history)
        autumn = _forecast_from_history(date(2026, 10, 25), self.history)
        self.assertEqual(len(spring), 3 * 23)
        self.assertEqual(len(autumn), 3 * 25)
        rows = prepare_map_rows(self.forecast, 9)
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows.arc_id), {4264, 5671, 191})
        self.assertTrue(rows.length_m.between(100, 300).all())


if __name__ == "__main__":
    unittest.main()
