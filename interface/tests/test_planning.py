import unittest

import pandas as pd

from flowpilot.planning import build_schedule, estimate_impact, export_calendar, rank_slots, traffic_risk


def forecast():
    rows = []
    for axe in ["Convention", "Saints-Peres"]:
        for hour in range(24):
            rows.append({
                "axe": axe,
                "datetime": pd.Timestamp("2026-11-07", tz="Europe/Paris") + pd.Timedelta(hours=hour),
                "occupation": 5.0 if 10 <= hour <= 12 else 35.0,
                "debit": 100.0 if 10 <= hour <= 12 else 400.0,
            })
    return pd.DataFrame(rows)


class PlanningTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual([traffic_risk(x) for x in [0, 14.9, 15, 30, 50]], ["Fluide", "Fluide", "Pré-saturé", "Saturé", "Bloqué"])

    def test_slots_favor_quiet_complete_windows(self):
        result = rank_slots(forecast(), ["Convention", "Saints-Peres"])
        self.assertEqual(result.iloc[0]["start_hour"], 10)
        self.assertTrue(result["score"].between(0, 100).all())
        frame = forecast()
        frame = frame[~(frame["axe"].eq("Saints-Peres") & frame["datetime"].dt.hour.eq(10))]
        result = rank_slots(frame, ["Convention", "Saints-Peres"])
        self.assertNotIn(9, result["start_hour"].tolist())
        self.assertNotIn(10, result["start_hour"].tolist())

    def test_impact_identical_and_negative(self):
        result = estimate_impact(forecast(), ["Convention"], 10, 10)
        self.assertEqual(result["daily_savings"], 0)
        self.assertEqual(result["time_saved_hours"], 0)
        result = estimate_impact(forecast(), ["Convention"], 10, 8)
        self.assertLess(result["daily_savings"], 0)
        result = estimate_impact(forecast(), ["Convention"], 8, 10)
        self.assertGreater(result["daily_savings"], 0)
        self.assertAlmostEqual(result["annual_savings"], result["daily_savings"] * 220)

    def test_schedule_capacity_and_deadlines(self):
        requests = pd.DataFrame([{"axe": "Convention", "earliest_hour": 8, "latest_hour": 12, "service_minutes": 20}] * 12)
        result = build_schedule(forecast(), requests, fleet_size=2, start_hour=8, end_hour=12, base_travel_minutes=20)
        self.assertEqual(len(result), 12)
        self.assertTrue(result["request_id"].is_unique)
        planned = result[result["status"].eq("Planifiée")]
        self.assertFalse(planned.empty)
        self.assertGreater((result["status"] == "Non planifiée").sum(), 0)
        for _, group in planned.groupby("vehicle"):
            group = group.sort_values("start")
            intervals = list(zip(group["start"], group["end"]))
            for (_, end), (next_start, _) in zip(intervals, intervals[1:]):
                self.assertLessEqual(end, next_start)
        start = pd.Timestamp("2026-11-07 08:00", tz="Europe/Paris")
        end = pd.Timestamp("2026-11-07 12:00", tz="Europe/Paris")
        self.assertTrue((planned["start"] >= start).all())
        self.assertTrue((planned["end"] <= end).all())

    def test_schedule_too_short_and_invalid(self):
        requests = pd.DataFrame([
            {"axe": "Convention", "earliest_hour": 8, "latest_hour": 8.25, "service_minutes": 40},
            {"axe": "Unknown", "earliest_hour": 8, "latest_hour": 12, "service_minutes": 5},
            {"axe": "Convention", "earliest_hour": 10, "latest_hour": 9, "service_minutes": 5},
        ])
        result = build_schedule(forecast(), requests)
        self.assertTrue(result["status"].eq("Non planifiée").all())
        self.assertTrue(result["reason"].str.len().gt(0).all())

    def test_schedule_respects_civil_window_on_both_dst_transition_days(self):
        for day in ["2026-03-29", "2026-10-25"]:
            with self.subTest(day=day):
                midnight = pd.Timestamp(day, tz="Europe/Paris")
                next_midnight = (midnight.tz_localize(None) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
                frame = pd.DataFrame({
                    "datetime": pd.date_range(midnight, next_midnight, freq="h", inclusive="left"),
                    "axe": "Convention", "occupation": 5.0, "debit": 100.0,
                })
                requests = pd.DataFrame([{
                    "axe": "Convention", "earliest_hour": 9, "latest_hour": 10,
                    "service_minutes": 5,
                }])
                result = build_schedule(frame, requests, fleet_size=1, base_travel_minutes=5)
                self.assertEqual(result.iloc[0]["status"], "Planifiée")
                self.assertEqual(result.iloc[0]["start"], pd.Timestamp(day + " 09:00", tz="Europe/Paris"))
                self.assertLessEqual(result.iloc[0]["end"], pd.Timestamp(day + " 10:00", tz="Europe/Paris"))

    def test_schedule_skips_nonexistent_hour_and_preserves_repeated_hour(self):
        def day_forecast(day):
            start = pd.Timestamp(day, tz="Europe/Paris")
            stop = (start.tz_localize(None) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
            return pd.DataFrame({
                "datetime": pd.date_range(start, stop, freq="h", inclusive="left"),
                "axe": "Convention", "occupation": 5.0, "debit": 100.0,
            })

        requests = pd.DataFrame([{
            "axe": "Convention", "earliest_hour": 2, "latest_hour": 3.5,
            "service_minutes": 5,
        }])
        result = build_schedule(day_forecast("2026-03-29"), requests, fleet_size=1,
                                start_hour=0, end_hour=24, base_travel_minutes=5)
        self.assertEqual(result.iloc[0]["start"], pd.Timestamp("2026-03-29 03:00", tz="Europe/Paris"))
        autumn = day_forecast("2026-10-25")
        first_two = autumn["datetime"].eq(pd.Timestamp("2026-10-25 02:00").tz_localize("Europe/Paris", ambiguous=True))
        second_two = pd.Timestamp("2026-10-25 02:00").tz_localize("Europe/Paris", ambiguous=False)
        autumn.loc[first_two, "occupation"] = 90.0
        result = build_schedule(autumn, requests, fleet_size=1, start_hour=0,
                                end_hour=24, base_travel_minutes=5)
        self.assertEqual(result.iloc[0]["start"], second_two)
        self.assertEqual(result.iloc[0]["occupation"], 5.0)

    def test_calendar_timezone_and_folding(self):
        requests = pd.DataFrame([{"axe": "Convention", "earliest_hour": 10, "latest_hour": 12, "service_minutes": 10}])
        schedule = build_schedule(forecast(), requests, fleet_size=1)
        exported = export_calendar(schedule)
        self.assertIsInstance(exported, bytes)
        calendar = exported.decode("utf-8")
        self.assertIn("BEGIN:VEVENT\r\n", calendar)
        self.assertIn("DTSTART:20261107T090000Z", calendar)
        self.assertTrue(all(len(line) <= 75 for line in exported.split(b"\r\n")))
        self.assertIn("END:VCALENDAR\r\n", calendar)


if __name__ == "__main__":
    unittest.main()
