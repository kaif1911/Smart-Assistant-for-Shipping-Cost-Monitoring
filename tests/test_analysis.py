import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis import monday_of, aggregate_weekly, build_baselines, note_applies, note_is_positive_support


class FreightAnalysisTests(unittest.TestCase):
    def test_week_starts_monday(self):
        dates = pd.Series(pd.to_datetime(["2025-02-24", "2025-03-02", "2025-03-03"]))
        self.assertEqual([d.strftime("%Y-%m-%d") for d in monday_of(dates)],
                         ["2025-02-24", "2025-02-24", "2025-03-03"])

    def test_weighted_weekly_ratio(self):
        df = pd.DataFrame({
            "origin": ["A", "A"], "destination": ["B", "B"], "route_type": ["Short", "Short"],
            "route": ["A-B", "A-B"], "week_of": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "freight_cost_inr": [100, 300], "tonne_km": [10, 90], "shipment_id": ["1", "2"]
        })
        got = aggregate_weekly(df).iloc[0]["cost_per_tonne_km"]
        self.assertAlmostEqual(got, 4.0)  # (100+300)/(10+90), not mean(10, 3.333...)

    def test_history_excludes_current_week_and_limits_eight(self):
        rows = []
        for i in range(10):
            rows.append({"route":"A-B","route_type":"Short","week_of":pd.Timestamp("2025-01-06")+pd.Timedelta(weeks=i),
                         "cost_per_tonne_km":float(i+1)})
        weekly = build_baselines(pd.DataFrame(rows))
        # Week 10 uses the previous eight costs 2..9, not current cost 10.
        self.assertAlmostEqual(weekly.iloc[9]["own_history_baseline"], 5.5)
        self.assertEqual(weekly.iloc[9]["own_history_weeks"], 8)

    def test_note_must_explicitly_support_increase(self):
        self.assertTrue(note_is_positive_support("A surcharge drove higher trip costs."))
        self.assertFalse(note_is_positive_support("Costs were not significantly affected."))

    def test_route_and_time_scope(self):
        n = pd.Series({"note_id":"N001","date":"2025-02-24","applies_to":"Chennai-Bangalore","note":"higher trip costs"})
        self.assertTrue(note_applies(n,"Chennai-Bangalore",pd.Timestamp("2025-02-24")))
        self.assertFalse(note_applies(n,"Delhi-Jaipur",pd.Timestamp("2025-02-24")))
        self.assertFalse(note_applies(n,"Chennai-Bangalore",pd.Timestamp("2025-03-10")))


if __name__ == "__main__":
    unittest.main()
