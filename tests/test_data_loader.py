"""
Unit tests for MarineMine Pro — Data Loader Module
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_sample_dataset,
    validate_dataset,
    get_dataset_info,
    infer_column_roles,
    get_summary_statistics,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a small valid test dataframe
        self.valid_df = pd.DataFrame({
            "record_id": ["MM-001", "MM-002", "MM-003", "MM-004"],
            "observation_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
            "latitude": [25.5, 30.1, -12.4, 45.0],
            "longitude": [-70.2, 50.3, 110.5, 12.0],
            "region": ["North Atlantic", "Indian Ocean", "South China Sea", "Mediterranean Sea"],
            "pollution_type": ["Microplastics", "Oil Spill", "Heavy Metals", "Sewage"],
            "pollution_value": [45.2, 120.0, 15.5, 88.0],
            "temperature_celsius": [18.5, 26.0, 28.2, 16.0],
        })

    def test_load_sample_dataset(self):
        """Test loading bundled sample dataset."""
        df = load_sample_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 500, "Sample dataset should contain at least 500 records")
        self.assertIn("pollution_value", df.columns)

    def test_validate_dataset_valid(self):
        """Test validation on a valid dataset."""
        res = validate_dataset(self.valid_df)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("is_valid", False))
        self.assertEqual(len(res.get("errors", [])), 0)

    def test_validate_dataset_empty(self):
        """Test validation on an empty dataframe."""
        empty_df = pd.DataFrame()
        res = validate_dataset(empty_df)
        self.assertFalse(res.get("is_valid", True))
        self.assertGreater(len(res.get("errors", [])), 0)

    def test_get_dataset_info(self):
        """Test dataset information extraction."""
        info = get_dataset_info(self.valid_df)
        self.assertEqual(info["n_rows"], 4)
        self.assertEqual(info["n_cols"], 8)
        self.assertIn("memory_usage_human", info)
        self.assertIn("dtypes", info)

    def test_infer_column_roles(self):
        """Test column semantic role inference."""
        roles = infer_column_roles(self.valid_df)
        self.assertIn("observation_date", roles.get("date_cols", []))
        self.assertIn("latitude", roles.get("lat_cols", []))
        self.assertIn("longitude", roles.get("lon_cols", []))
        self.assertIsInstance(roles.get("numeric_cols", []), list)
        self.assertIsInstance(roles.get("categorical_cols", []), list)


if __name__ == "__main__":
    unittest.main()
