"""
Unit tests for MarineMine Pro — Preprocessing Module
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PreprocessingPipeline, quick_clean, get_cleaning_stats


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create dataset with known duplicates and missing values
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 3, 4],  # duplicate row 3 and 4
            "val": [10.0, np.nan, 30.0, 30.0, 50.0],
            "cat": ["A", "B", "A", "A", None],
            "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-03-01", "2023-05-01"],
        })

    def test_original_not_modified(self):
        """Verify pipeline operates on copy and leaves original dataframe unchanged."""
        orig_copy = self.df.copy(deep=True)
        pipeline = PreprocessingPipeline(self.df)
        pipeline.remove_duplicates()
        pipeline.handle_missing_values(strategy="mean")
        _ = pipeline.get_processed_df()

        pd.testing.assert_frame_equal(self.df, orig_copy)

    def test_remove_duplicates(self):
        """Verify duplicate rows are accurately dropped."""
        pipeline = PreprocessingPipeline(self.df)
        pipeline.remove_duplicates()
        cleaned = pipeline.get_processed_df()
        self.assertEqual(len(cleaned), 4)

    def test_handle_missing_values(self):
        """Verify missing values are imputed."""
        pipeline = PreprocessingPipeline(self.df)
        pipeline.handle_missing_values(strategy="mean")
        cleaned = pipeline.get_processed_df()
        self.assertEqual(cleaned["val"].isna().sum(), 0)

    def test_scale_numerics(self):
        """Verify numerical scaling creates standard normal properties."""
        df_num = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        pipeline = PreprocessingPipeline(df_num)
        pipeline.scale_numerics(cols=["x"], method="standard")
        res = pipeline.get_processed_df()
        self.assertAlmostEqual(res["x"].mean(), 0.0, places=5)
        self.assertAlmostEqual(res["x"].std(ddof=0), 1.0, places=5)

    def test_quick_clean(self):
        """Verify quick_clean wrapper functionality."""
        cleaned = quick_clean(self.df)
        self.assertLessEqual(len(cleaned), len(self.df))
        stats = get_cleaning_stats(self.df, cleaned)
        self.assertIn("rows_removed", stats)


if __name__ == "__main__":
    unittest.main()
