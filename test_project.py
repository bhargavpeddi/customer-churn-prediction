import unittest

from generate_data import FEATURES, make_rows


class SyntheticDataTests(unittest.TestCase):
    def test_reproducible_and_unique(self):
        rows = make_rows(500, seed=12)
        self.assertEqual(rows, make_rows(500, seed=12))
        self.assertEqual(len({row["customer_id"] for row in rows}), 500)

    def test_schema_and_classes(self):
        rows = make_rows(500, seed=12)
        self.assertEqual(set(rows[0]), {"customer_id", *FEATURES, "churned"})
        self.assertEqual({row["churned"] for row in rows}, {0, 1})

    def test_rejects_tiny_dataset(self):
        with self.assertRaises(ValueError):
            make_rows(9)


if __name__ == "__main__":
    unittest.main()
