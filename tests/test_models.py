"""
Unit tests for MarineMine Pro — Machine Learning & Analytics Modules
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import run_kmeans, compute_elbow, compute_silhouette, get_cluster_stats
from src.classification import check_target_suitability, train_classifiers
from src.prediction import check_prediction_suitability, train_regressors
from src.association_rules import discretize_column, prepare_transactions, mine_frequent_itemsets


class TestModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        # Synthetic numeric dataset for ML tests
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(5, 1, n)
        target_num = 2.5 * x1 - 1.2 * x2 + np.random.normal(0, 0.5, n)
        target_cat = np.where(target_num > np.median(target_num), "High", "Low")

        self.df = pd.DataFrame({
            "feature1": x1,
            "feature2": x2,
            "feature3": np.random.uniform(10, 50, n),
            "target_num": target_num,
            "target_cat": target_cat,
            "category_A": np.random.choice(["Type1", "Type2", "Type3"], n),
            "category_B": np.random.choice(["Region1", "Region2"], n),
        })

    def test_kmeans_clustering(self):
        """Test K-Means execution and cluster assignment."""
        features = ["feature1", "feature2", "feature3"]
        df_clustered, labels = run_kmeans(self.df, features, n_clusters=3, random_state=42)
        self.assertIn("cluster", df_clustered.columns)
        self.assertEqual(len(set(labels)), 3)

        # Cluster stats
        stats = get_cluster_stats(df_clustered, features)
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertEqual(len(stats), 3)

    def test_elbow_and_silhouette(self):
        """Test elbow curve and silhouette score calculation."""
        features = ["feature1", "feature2"]
        elbow_dict = compute_elbow(self.df, features, k_range=range(2, 5))
        self.assertIn(2, elbow_dict)
        self.assertIn(3, elbow_dict)

        sil = compute_silhouette(self.df, features, n_clusters=3)
        self.assertIsInstance(sil, float)
        self.assertTrue(-1.0 <= sil <= 1.0)

    def test_classification_training(self):
        """Test multi-model classification pipeline."""
        suitability = check_target_suitability(self.df, "target_cat")
        self.assertTrue(suitability["is_suitable"])

        results = train_classifiers(
            self.df,
            feature_cols=["feature1", "feature2"],
            target_col="target_cat",
            test_size=0.2,
            random_state=42
        )
        self.assertIn("RandomForest", results)
        self.assertIn("LogisticRegression", results)
        rf_metrics = results["RandomForest"]
        self.assertTrue(0.0 <= rf_metrics["accuracy"] <= 1.0)
        self.assertIn("f1", rf_metrics)

    def test_regression_training(self):
        """Test regression models and evaluation metrics."""
        suitability = check_prediction_suitability(self.df, "target_num")
        self.assertTrue(suitability.get("suitable", False) or suitability.get("is_suitable", False))

        results = train_regressors(
            self.df,
            feature_cols=["feature1", "feature2"],
            target_col="target_num",
            test_size=0.2,
            random_state=42
        )
        self.assertIn("RandomForestRegressor", results)
        rf = results["RandomForestRegressor"]
        self.assertIn("mae", rf)
        self.assertIn("r2", rf)
        self.assertGreater(rf["r2"], 0.5)

    def test_association_rules(self):
        """Test transaction encoding and frequent itemset generation."""
        cols = ["category_A", "category_B"]
        tx_matrix = prepare_transactions(self.df, cols=cols)
        self.assertIsInstance(tx_matrix, pd.DataFrame)

        itemsets = mine_frequent_itemsets(tx_matrix, min_support=0.1)
        self.assertIsInstance(itemsets, pd.DataFrame)
        self.assertGreater(len(itemsets), 0)


if __name__ == "__main__":
    unittest.main()
