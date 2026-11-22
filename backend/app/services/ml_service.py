"""
MarineMine Pro — Machine Learning Service (Clustering, Classification, Association Rules)
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src import clustering, classification, association_rules


class MLService:
    @staticmethod
    def run_clustering(df: pd.DataFrame, features: List[str], k: int = 3) -> Dict[str, Any]:
        """Execute K-Means clustering and return silhouette, cluster profiles, and PCA 2D projections."""
        valid_features = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
        if len(valid_features) < 2:
            raise ValueError("Clustering requires at least 2 valid numeric features.")

        clean_sub = df[valid_features].dropna()
        if len(clean_sub) < k:
            raise ValueError(f"Insufficient non-null rows ({len(clean_sub)}) for {k} clusters.")

        result_df, labels = clustering.run_kmeans(clean_sub, valid_features, n_clusters=k)
        sil_score = clustering.compute_silhouette(clean_sub, valid_features, n_clusters=k)
        sil_val = round(float(sil_score), 4) if sil_score is not None else 0.0

        # Calculate cluster profiles
        profiles = []
        cluster_counts = {}
        total_pts = len(result_df)

        for c_id in sorted(list(set(labels))):
            c_mask = result_df["cluster"] == c_id
            c_sub = result_df[c_mask]
            c_size = len(c_sub)
            cluster_counts[f"Cluster {c_id}"] = c_size

            means_dict = {}
            for feat in valid_features:
                means_dict[feat] = round(float(c_sub[feat].mean()), 2)

            profiles.append({
                "cluster_id": int(c_id),
                "label": f"Cluster {c_id}",
                "size": c_size,
                "percentage": round((c_size / total_pts) * 100, 1),
                "means": means_dict,
            })

        # Calculate PCA 2D coordinates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean_sub[valid_features])
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        pca_points = []
        for idx in range(min(200, len(X_pca))):
            pca_points.append({
                "x": round(float(X_pca[idx, 0]), 3),
                "y": round(float(X_pca[idx, 1]), 3),
                "cluster": int(labels[idx]),
            })

        # Calculate inertia
        centers = []
        for c_id in sorted(list(set(labels))):
            c_pts = X_scaled[np.array(labels) == c_id]
            centers.append(c_pts.mean(axis=0) if len(c_pts) > 0 else np.zeros(X_scaled.shape[1]))
        centers_arr = np.array(centers)
        inertia_val = float(np.sum([np.sum((X_scaled[i] - centers_arr[labels[i]]) ** 2) for i in range(len(X_scaled))]))

        return {
            "k": k,
            "silhouette_score": sil_val,
            "inertia": round(inertia_val, 2),
            "cluster_counts": cluster_counts,
            "pca_data": pca_points,
            "cluster_profiles": profiles,
            "feature_importances": {feat: round(1.0 / len(valid_features), 3) for feat in valid_features},
        }

    @staticmethod
    def find_optimal_k(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Evaluate elbow inertia and silhouette curve across k in [2, 8]."""
        valid_features = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
        if len(valid_features) < 2:
            raise ValueError("Optimal k evaluation requires at least 2 numeric features.")

        clean_sub = df[valid_features].dropna()
        res = clustering.find_optimal_k(clean_sub, valid_features)

        scores = res.get("scores", [])
        k_range = [s["k"] for s in scores]
        inertias = [float(s["inertia"]) for s in scores]
        silhouettes = [float(s["silhouette"]) for s in scores]

        return {
            "k_range": k_range,
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "recommended_k": int(res.get("recommended_k", 3)),
            "best_silhouette": round(float(res.get("best_silhouette", 0.0)), 4),
        }

    @staticmethod
    def run_classification(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train and benchmark Decision Tree, Random Forest, and Logistic Regression."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        feats = feature_cols
        if not feats:
            feats = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feats:
                feats.remove(target_col)

        valid_features = [f for f in feats if f in df.columns and f != target_col]
        if not valid_features:
            raise ValueError("No valid input feature columns for classification.")

        t0 = time.time()
        results = classification.train_classifiers(
            df=df,
            feature_cols=valid_features,
            target_col=target_col,
            test_size=test_size,
        )
        elapsed = int((time.time() - t0) * 1000)

        # Select best model based on F1 score / accuracy
        best_name = max(results.keys(), key=lambda k: results[k].get("f1", 0.0))
        best_res = results[best_name]

        model_summaries = {}
        for m_name, m in results.items():
            model_summaries[m_name] = {
                "accuracy": round(float(m.get("accuracy", 0.0)), 4),
                "precision": round(float(m.get("precision", 0.0)), 4),
                "recall": round(float(m.get("recall", 0.0)), 4),
                "f1_score": round(float(m.get("f1", 0.0)), 4),
            }

        cm = best_res.get("cm")
        cm_list = cm.tolist() if isinstance(cm, np.ndarray) else [[0]]

        feat_imp_list = []
        try:
            imp_dict = classification.get_feature_importance(best_res["model"], valid_features)
            for feat, imp in imp_dict.items():
                feat_imp_list.append({"feature": feat, "importance": round(float(imp), 4)})
        except Exception:
            for f in valid_features:
                feat_imp_list.append({"feature": f, "importance": round(1.0 / len(valid_features), 4)})

        classes = [str(c) for c in df[target_col].dropna().unique()]

        return {
            "target_column": target_col,
            "classes": classes,
            "models": model_summaries,
            "best_model": best_name,
            "best_accuracy": round(float(best_res.get("accuracy", 0.0)), 4),
            "best_f1": round(float(best_res.get("f1", 0.0)), 4),
            "confusion_matrix": cm_list,
            "feature_importance": feat_imp_list,
            "elapsed_ms": elapsed,
        }

    @staticmethod
    def run_association_rules(
        df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        min_support: float = 0.05,
        min_confidence: float = 0.4,
        min_lift: float = 1.0,
    ) -> Dict[str, Any]:
        """Run Apriori frequent itemset and association rule mining."""
        target_cols = cat_cols
        if not target_cols:
            all_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
            target_cols = [c for c in all_cats if "id" not in c.lower() and "notes" not in c.lower()][:4]

        if not target_cols or len(target_cols) < 2:
            return {
                "rules_count": 0,
                "rules": [],
                "suitable": False,
                "explanation": "At least 2 categorical columns are required for association rule mining.",
            }

        try:
            basket_df = association_rules.create_transaction_basket(df, categorical_cols=target_cols)
            frequent_itemsets = association_rules.mine_frequent_itemsets(basket_df, min_support=min_support)
            rules_df = association_rules.generate_association_rules(
                frequent_itemsets,
                min_confidence=min_confidence,
                min_lift=min_lift,
            )
            rule_records = association_rules.rules_to_display_records(rules_df)
            return {
                "rules_count": len(rule_records),
                "rules": rule_records[:50],
                "suitable": True,
                "explanation": f"Generated {len(rule_records)} rules from {len(target_cols)} categorical fields.",
            }
        except Exception as exc:
            return {
                "rules_count": 0,
                "rules": [],
                "suitable": False,
                "explanation": f"Association mining unfeasible with current parameter thresholds: {exc}",
            }
