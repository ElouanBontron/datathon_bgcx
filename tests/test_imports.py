import unittest

import pandas as pd

from flowpilot.imports import IMPORTED_METHOD, parse_predictions


HEADER = "arc,datetime,debit_horaire,taux_occupation\n"


def csv_bytes(*rows):
    return (HEADER + "\n".join(rows)).encode("utf-8")


class PredictionImportTests(unittest.TestCase):
    def test_challenge_names_and_forecast_contract(self):
        frame = parse_predictions(csv_bytes(
            "Champs-Elysées,2026-11-11 09:00,1234,18.5",
            "Convention,2026-11-11 09:00,200,10",
            "Saint-Pères,2026-11-11 09:00,90,2",
        ))
        self.assertEqual(set(frame.axe), {"Champs-Elysees", "Convention", "Saints-Peres"})
        self.assertEqual(str(frame.datetime.dt.tz), "Europe/Paris")
        # Import timestamps are START times and must not be shifted back an hour.
        self.assertTrue(frame.datetime.dt.hour.eq(9).all())
        self.assertTrue(frame.is_holiday.all())
        self.assertTrue(frame.method.eq(IMPORTED_METHOD).all())
        self.assertTrue(frame.samples.eq(0).all())
        self.assertTrue(frame.last_observation.isna().all())
        self.assertTrue(frame.stale_days.isna().all())
        pd.testing.assert_series_equal(frame.lower_occupation, frame.occupation, check_names=False)
        pd.testing.assert_series_equal(frame.upper_occupation, frame.occupation, check_names=False)

    def test_semicolon_bom_and_french_decimal(self):
        frame = parse_predictions((
            "\ufeffarc;datetime;debit_horaire;taux_occupation\r\n"
            " saints_peres ;2026-11-10 08:00;120,5;5,25\r\n"
        ).encode("utf-8"))
        self.assertEqual(frame.at[0, "axe"], "Saints-Peres")
        self.assertEqual(frame.at[0, "debit"], 120.5)
        self.assertEqual(frame.at[0, "occupation"], 5.25)
        self.assertFalse(frame.at[0, "is_holiday"])

    def test_numeric_validity_and_bounds(self):
        for debit, occupation in [
            ("", "10"), ("NaN", "10"), ("inf", "10"), ("-1", "10"),
            ("50000.01", "10"), ("10", "NaN"), ("10", "-0.1"), ("10", "100.01"),
        ]:
            with self.subTest(debit=debit, occupation=occupation), self.assertRaises(ValueError):
                parse_predictions(csv_bytes(f"Convention,2026-11-07 08:00,{debit},{occupation}"))
        frame = parse_predictions(csv_bytes("Convention,2026-11-07 08:00,50000,100"))
        self.assertEqual(frame.at[0, "debit"], 50_000)

    def test_duplicate_names_are_normalized_before_validation(self):
        with self.assertRaisesRegex(ValueError, "double"):
            parse_predictions(csv_bytes(
                "Saint-Pères,2026-11-07 08:00,100,10",
                "Saints-Peres,2026-11-07 08:00,110,12",
            ))

    def test_unknown_axis(self):
        with self.assertRaisesRegex(ValueError, "Axes inconnus"):
            parse_predictions(csv_bytes("Rivoli,2026-11-07 08:00,100,10"))

    def test_dates_must_be_valid_and_hourly(self):
        for stamp in ["", "NaT", "not-a-date", "07/11/2026 08:00", "2026-11-07",
                      "2026-11-07 08:15", "2026-11-07 08:00:01",
                      "2026-11-07 08:00:00.000001"]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                parse_predictions(csv_bytes(f"Convention,{stamp},100,10"))

    def test_timezone_offsets_must_correspond_to_paris(self):
        good = parse_predictions(csv_bytes("Convention,2026-11-07T08:00:00+01:00,100,10"))
        self.assertEqual(good.at[0, "datetime"].hour, 8)
        for stamp in ["2026-11-07T08:00:00+02:00", "2026-11-07T08:00:00Z"]:
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, "Paris"):
                parse_predictions(csv_bytes(f"Convention,{stamp},100,10"))
        with self.assertRaisesRegex(ValueError, "mélangez"):
            parse_predictions(csv_bytes(
                "Convention,2026-11-07T08:00:00+01:00,100,10",
                "Convention,2026-11-07 09:00,100,10",
            ))

    def test_dst_requires_explicit_offset_for_ambiguous_hour(self):
        for stamp in ["2026-03-29 02:00", "2026-10-25 02:00"]:
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, "Paris"):
                parse_predictions(csv_bytes(f"Convention,{stamp},100,10"))
        frame = parse_predictions(csv_bytes(
            "Convention,2026-10-25T02:00:00+02:00,100,10",
            "Convention,2026-10-25T02:00:00+01:00,110,12",
        ))
        self.assertEqual(len(frame), 2)
        self.assertEqual((frame.datetime.iloc[1] - frame.datetime.iloc[0]).total_seconds(), 3600)

    def test_bad_files_raise_readable_value_errors(self):
        for content in [b"", b"\xff", HEADER.encode(), b"arc,datetime\nConvention,2026-11-07 08:00",
                        b"arc,arc,datetime,debit_horaire,taux_occupation\na,b,c,d,e"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_predictions(content)


if __name__ == "__main__":
    unittest.main()
