import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare import DATA_PATH, features, load


class TelcoDataTests(unittest.TestCase):
    def test_loads_all_customers_with_binary_target(self):
        frame = load()
        self.assertEqual(len(frame), 7043)
        self.assertFalse(frame["customerID"].duplicated().any())
        self.assertEqual(set(frame["Churn"]), {"Yes", "No"})

    def test_blank_total_charges_only_for_new_customers(self):
        raw = pd.read_csv(DATA_PATH)
        blank = pd.to_numeric(raw["TotalCharges"], errors="coerce").isna()
        self.assertEqual(int(blank.sum()), 11)
        self.assertTrue((raw.loc[blank, "tenure"] == 0).all())
        self.assertEqual(float(load().loc[blank, "TotalCharges"].sum()), 0.0)

    def test_features_are_numeric_with_no_missing_values(self):
        x, y = features(load())
        self.assertEqual(len(x), len(y))
        self.assertFalse(x.isna().any().any())
        self.assertIn("avg_monthly_spend", x)
        self.assertTrue(x["add_on_services"].between(0, 6).all())
        self.assertNotIn("customerID", x)

    def test_rejects_duplicate_ids(self):
        frame = pd.read_csv(DATA_PATH).head(20)
        frame = pd.concat([frame, frame.head(1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dupes.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load(path)


if __name__ == "__main__":
    unittest.main()
