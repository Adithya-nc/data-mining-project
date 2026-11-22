"""
MarineMine Pro — Association Rules Mining Module
=================================================
MLxtend-based Apriori implementation for discovering hidden patterns
and co-occurrence relationships in marine pollution datasets.

Author : MarineMine Pro — Senior ML Engineering Team
Updated: 2026-11-22
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — give clear error messages if missing
# ---------------------------------------------------------------------------
try:
    from mlxtend.frequent_patterns import apriori, association_rules as mlxtend_assoc_rules
    from mlxtend.preprocessing import TransactionEncoder
    _MLXTEND_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLXTEND_AVAILABLE = False
    logger.warning(
        "mlxtend is not installed. Association rule mining will be unavailable. "
        "Install it with: pip install mlxtend"
    )


# ===========================================================================
# 1. DISCRETISATION
# ===========================================================================

def discretize_column(
    series: pd.Series,
    n_bins: int = 4,
    labels: Optional[List[str]] = None,
) -> pd.Series:
    """
    Bin a continuous numeric column into labelled categorical intervals.

    Parameters
    ----------
    series : pd.Series
        The numeric series to discretise.
    n_bins : int, default 4
        Number of equal-width bins.
    labels : list of str, optional
        Custom labels for each bin.  Length must equal ``n_bins``.
        When *None* the bins are auto-labelled as
        ``"<col_name>_low"``, ``"<col_name>_mid_1"``, …

    Returns
    -------
    pd.Series
        Categorical series with the same index, dtype ``object``.

    Raises
    ------
    ValueError
        If ``n_bins < 2`` or ``labels`` length does not match ``n_bins``.
    TypeError
        If ``series`` is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"discretize_column expects a numeric series, got '{series.dtype}' "
            f"for column '{series.name}'."
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")

    col_name = series.name or "feature"

    if labels is not None:
        if len(labels) != n_bins:
            raise ValueError(
                f"labels length ({len(labels)}) must equal n_bins ({n_bins})."
            )
    else:
        if n_bins == 2:
            labels = [f"{col_name}_low", f"{col_name}_high"]
        elif n_bins == 3:
            labels = [f"{col_name}_low", f"{col_name}_mid", f"{col_name}_high"]
        elif n_bins == 4:
            labels = [
                f"{col_name}_low",
                f"{col_name}_mid_low",
                f"{col_name}_mid_high",
                f"{col_name}_high",
            ]
        else:
            labels = [f"{col_name}_bin_{i+1}" for i in range(n_bins)]

    # Drop NaNs temporarily for cut, then restore
    non_null = series.dropna()
    if non_null.nunique() < 2:
        logger.warning(
            "Column '%s' has fewer than 2 unique non-null values — "
            "all rows will fall in one bin.",
            col_name,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        binned = pd.cut(
            series,
            bins=n_bins,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )

    return binned.astype(object)


# ===========================================================================
# 2. TRANSACTION MATRIX PREPARATION
# ===========================================================================

def prepare_transactions(
    df: pd.DataFrame,
    cols: List[str],
    num_cols_to_bin: Optional[List[str]] = None,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Build a boolean one-hot-encoded transaction matrix suitable for Apriori.

    Each row represents one observation (e.g. a water-quality sample);
    each column represents an *item* (e.g. ``"pH_high"`` or ``"Region_Coast"``).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset (may contain both numeric and categorical columns).
    cols : list of str
        Columns to include.  Numeric columns will be passed through
        ``discretize_column`` unless listed in ``num_cols_to_bin`` too.
    num_cols_to_bin : list of str, optional
        Subset of *cols* that are numeric and should be binned.
        If *None*, every numeric column in *cols* is auto-binned.
    n_bins : int, default 4
        Passed to :func:`discretize_column` for all numeric columns.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with shape ``(n_rows, n_items)``.

    Raises
    ------
    ValueError
        If *cols* is empty, no columns exist in *df*, or the resulting
        matrix has fewer than 2 non-null rows.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    if not cols:
        raise ValueError("cols must contain at least one column name.")

    work = df[cols].copy()

    # Decide which numeric columns to bin
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols_to_bin is None:
        to_bin = numeric_cols
    else:
        to_bin = [c for c in num_cols_to_bin if c in numeric_cols]

    for col in to_bin:
        try:
            work[col] = discretize_column(work[col], n_bins=n_bins)
        except Exception as exc:
            logger.warning("Could not bin column '%s': %s — skipping.", col, exc)
            work.drop(columns=[col], inplace=True)

    # Drop rows that are entirely NaN
    work.dropna(how="all", inplace=True)

    if len(work) < 2:
        raise ValueError(
            "After preprocessing, fewer than 2 rows remain. "
            "Cannot build a transaction matrix."
        )

    # One-hot encode — prefix each column with its source name
    ohe_parts = []
    for col in work.columns:
        dummies = pd.get_dummies(work[col], prefix=col, prefix_sep="=", dtype=bool)
        ohe_parts.append(dummies)

    if not ohe_parts:
        raise ValueError("No valid columns remained after preprocessing.")

    transaction_matrix = pd.concat(ohe_parts, axis=1).fillna(False)

    logger.info(
        "Transaction matrix built: %d rows × %d items.",
        len(transaction_matrix),
        len(transaction_matrix.columns),
    )
    return transaction_matrix


# ===========================================================================
# 3. FREQUENT ITEMSET MINING
# ===========================================================================

def mine_frequent_itemsets(
    transaction_matrix: pd.DataFrame,
    min_support: float = 0.05,
) -> pd.DataFrame:
    """
    Run the Apriori algorithm to find frequent itemsets.

    Parameters
    ----------
    transaction_matrix : pd.DataFrame
        Boolean DataFrame produced by :func:`prepare_transactions`.
    min_support : float, default 0.05
        Minimum support threshold (fraction of transactions, 0 < value ≤ 1).

    Returns
    -------
    pd.DataFrame
        Columns: ``support`` (float), ``itemsets`` (frozenset).
        Returns an *empty* DataFrame with these columns when no itemsets
        satisfy the threshold.

    Raises
    ------
    ImportError
        If mlxtend is not installed.
    ValueError
        If ``min_support`` is out of range or the input matrix is empty.
    """
    _require_mlxtend()

    if not (0 < min_support <= 1.0):
        raise ValueError(f"min_support must be in (0, 1], got {min_support}.")
    if transaction_matrix.empty:
        raise ValueError("transaction_matrix is empty.")

    logger.info("Running Apriori with min_support=%.4f …", min_support)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        itemsets_df = apriori(
            transaction_matrix,
            min_support=min_support,
            use_colnames=True,
            max_len=None,
            low_memory=False,
        )

    if itemsets_df.empty:
        logger.warning(
            "No frequent itemsets found at min_support=%.4f. "
            "Consider lowering the threshold.",
            min_support,
        )
        return _empty_itemsets_df()

    itemsets_df = itemsets_df.sort_values("support", ascending=False).reset_index(drop=True)
    logger.info("Found %d frequent itemset(s).", len(itemsets_df))
    return itemsets_df


# ===========================================================================
# 4. ASSOCIATION RULE GENERATION
# ===========================================================================

def generate_rules(
    itemsets_df: pd.DataFrame,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """
    Derive association rules from a set of frequent itemsets.

    Parameters
    ----------
    itemsets_df : pd.DataFrame
        Output of :func:`mine_frequent_itemsets`.
    min_confidence : float, default 0.5
        Minimum confidence threshold (0 < value ≤ 1).
    min_lift : float, default 1.0
        Minimum lift threshold (> 0).  Values > 1 indicate the antecedent
        makes the consequent *more* likely than by chance.

    Returns
    -------
    pd.DataFrame
        Standard mlxtend association-rules DataFrame, additionally filtered
        by *min_lift*, with columns sorted for readability.
        Returns an empty DataFrame (with the expected columns) when no rules
        are found.

    Raises
    ------
    ImportError
        If mlxtend is not installed.
    ValueError
        If thresholds are out of range.
    """
    _require_mlxtend()

    if not (0 < min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}.")
    if min_lift <= 0:
        raise ValueError(f"min_lift must be > 0, got {min_lift}.")

    if itemsets_df is None or itemsets_df.empty:
        logger.warning("itemsets_df is empty — no rules to generate.")
        return _empty_rules_df()

    # Need itemsets of length ≥ 2 to form rules with antecedent AND consequent
    if "itemsets" not in itemsets_df.columns:
        raise ValueError("itemsets_df must contain an 'itemsets' column.")

    has_pairs = itemsets_df["itemsets"].apply(len).ge(2).any()
    if not has_pairs:
        logger.warning(
            "All frequent itemsets have length 1 — cannot derive rules "
            "with both antecedent and consequent. "
            "Try lowering min_support or adding more columns."
        )
        return _empty_rules_df()

    logger.info(
        "Generating rules with min_confidence=%.2f, min_lift=%.2f …",
        min_confidence,
        min_lift,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rules_df = mlxtend_assoc_rules(
            itemsets_df,
            metric="confidence",
            min_threshold=min_confidence,
        )

    if rules_df.empty:
        logger.warning(
            "No rules met min_confidence=%.2f. "
            "Consider lowering the threshold.",
            min_confidence,
        )
        return _empty_rules_df()

    # Apply lift filter
    rules_df = rules_df[rules_df["lift"] >= min_lift].copy()

    if rules_df.empty:
        logger.warning(
            "No rules survived the lift filter (min_lift=%.2f).", min_lift
        )
        return _empty_rules_df()

    # Sort by lift desc, then confidence desc
    rules_df = rules_df.sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)

    logger.info("Generated %d association rule(s).", len(rules_df))
    return rules_df


# ===========================================================================
# 5. HUMAN-READABLE RULE EXPLANATION
# ===========================================================================

def explain_rules(
    rules_df: pd.DataFrame,
    top_n: int = 10,
) -> List[str]:
    """
    Convert the top-N rules into plain-English sentences for display or logging.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Output of :func:`generate_rules`.
    top_n : int, default 10
        Maximum number of rules to explain (ranked by lift, then confidence).

    Returns
    -------
    list of str
        Human-readable rule strings.  Returns a single-element list with
        a "no rules found" message when *rules_df* is empty.

    Example output
    --------------
    ::

        [
          "Rule 1: When [pH=high, Region=Coast] → [Turbidity=mid_high]  "
          "| Support: 0.123 (12.3% of samples)  "
          "| Confidence: 0.820 (82.0% of the time)  "
          "| Lift: 2.45 (2.45× more likely than by chance)",
          ...
        ]
    """
    if rules_df is None or rules_df.empty:
        return [
            "⚠️  No association rules were found with the current thresholds. "
            "Try lowering min_support or min_confidence, or add more columns "
            "to the transaction matrix."
        ]

    subset = rules_df.head(top_n).copy()
    explanations: List[str] = []

    for rank, (_, row) in enumerate(subset.iterrows(), start=1):
        ant = _frozenset_to_str(row["antecedents"])
        con = _frozenset_to_str(row["consequents"])
        sup = float(row["support"])
        conf = float(row["confidence"])
        lift = float(row["lift"])

        line = (
            f"Rule {rank}: When [{ant}] → [{con}]  "
            f"| Support: {sup:.3f} ({sup*100:.1f}% of samples)  "
            f"| Confidence: {conf:.3f} ({conf*100:.1f}% of the time)  "
            f"| Lift: {lift:.3f} ({lift:.2f}× more likely than by chance)"
        )
        explanations.append(line)

    return explanations


# ===========================================================================
# 6. STREAMLIT DISPLAY FORMATTER
# ===========================================================================

def format_rules_display(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the raw rules DataFrame into a clean, presentation-ready table
    suitable for ``st.dataframe()`` in a Streamlit app.

    Transformations applied
    -----------------------
    * Frozensets → comma-separated strings.
    * Numeric metrics rounded to 4 decimal places.
    * Columns renamed to title-case human labels.
    * A "Strength" categorical column added (Weak / Moderate / Strong / Very Strong)
      based on lift quartiles.
    * Columns reordered for logical reading flow.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Output of :func:`generate_rules`.

    Returns
    -------
    pd.DataFrame
        Display-ready DataFrame.  Returns a one-row placeholder DataFrame
        with a "Message" column when *rules_df* is empty.
    """
    if rules_df is None or rules_df.empty:
        return pd.DataFrame(
            {
                "Message": [
                    "No association rules found. "
                    "Adjust thresholds or select different columns."
                ]
            }
        )

    display = rules_df.copy()

    # Frozensets → readable strings
    display["antecedents"] = display["antecedents"].apply(_frozenset_to_str)
    display["consequents"] = display["consequents"].apply(_frozenset_to_str)

    # Round numeric columns
    numeric_cols = ["support", "confidence", "lift", "leverage", "conviction"]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = display[col].round(4)

    # Lift-based strength label
    if "lift" in display.columns:
        lift_vals = display["lift"]
        q33 = lift_vals.quantile(0.33)
        q66 = lift_vals.quantile(0.66)
        q90 = lift_vals.quantile(0.90)

        def _strength(lift_val: float) -> str:
            if lift_val >= q90:
                return "🔴 Very Strong"
            if lift_val >= q66:
                return "🟠 Strong"
            if lift_val >= q33:
                return "🟡 Moderate"
            return "🟢 Weak"

        display["Strength"] = display["lift"].apply(_strength)

    # Rename columns
    rename_map = {
        "antecedents": "If (Antecedent)",
        "consequents": "Then (Consequent)",
        "support": "Support",
        "confidence": "Confidence",
        "lift": "Lift",
        "leverage": "Leverage",
        "conviction": "Conviction",
    }
    display.rename(columns=rename_map, inplace=True)

    # Preferred column order (keep any extras at end)
    preferred_order = [
        "If (Antecedent)",
        "Then (Consequent)",
        "Support",
        "Confidence",
        "Lift",
        "Strength",
        "Leverage",
        "Conviction",
    ]
    ordered_cols = [c for c in preferred_order if c in display.columns]
    extra_cols = [c for c in display.columns if c not in preferred_order]
    display = display[ordered_cols + extra_cols].reset_index(drop=True)

    return display


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _require_mlxtend() -> None:
    """Raise ImportError with installation instructions if mlxtend is absent."""
    if not _MLXTEND_AVAILABLE:
        raise ImportError(
            "mlxtend is required for association rule mining but is not installed.\n"
            "Install it with:  pip install mlxtend"
        )


def _frozenset_to_str(fs: Union[frozenset, set]) -> str:
    """Convert a frozenset of item names to a readable comma-separated string."""
    return ", ".join(sorted(str(item) for item in fs))


def _empty_itemsets_df() -> pd.DataFrame:
    """Return a typed-but-empty itemsets DataFrame."""
    return pd.DataFrame(columns=["support", "itemsets"])


def _empty_rules_df() -> pd.DataFrame:
    """Return a typed-but-empty rules DataFrame matching mlxtend's schema."""
    return pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]
    )


# ===========================================================================
# QUICK SMOKE-TEST  (python src/association_rules.py)
# ===========================================================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # --- synthetic marine pollution dataset -----------------------------------
    rng = np.random.default_rng(42)
    n = 500

    sample_data = pd.DataFrame(
        {
            "pH": rng.normal(7.5, 0.8, n),
            "Turbidity_NTU": rng.exponential(5, n),
            "Dissolved_O2": rng.normal(6, 1.5, n),
            "Plastic_Index": rng.exponential(2, n),
            "Region": rng.choice(["Coastal", "Estuarine", "Offshore", "Riverine"], n),
            "Season": rng.choice(["Spring", "Summer", "Autumn", "Winter"], n),
        }
    )

    print("\n=== MarineMine Pro — Association Rules Smoke Test ===\n")

    # 1. Prepare transactions
    tx_matrix = prepare_transactions(
        sample_data,
        cols=["pH", "Turbidity_NTU", "Dissolved_O2", "Plastic_Index", "Region", "Season"],
        num_cols_to_bin=["pH", "Turbidity_NTU", "Dissolved_O2", "Plastic_Index"],
        n_bins=3,
    )
    print(f"Transaction matrix shape: {tx_matrix.shape}")
    print(f"Sample columns: {list(tx_matrix.columns[:6])} …\n")

    # 2. Mine frequent itemsets
    itemsets = mine_frequent_itemsets(tx_matrix, min_support=0.08)
    print(f"Frequent itemsets found: {len(itemsets)}")
    if not itemsets.empty:
        print(itemsets.head(5).to_string(index=False))

    print()

    # 3. Generate rules
    rules = generate_rules(itemsets, min_confidence=0.4, min_lift=1.0)
    print(f"Rules generated: {len(rules)}\n")

    # 4. Explain top rules
    print("=== Top Rules (Human-Readable) ===")
    for line in explain_rules(rules, top_n=5):
        print(" ", line)

    print()

    # 5. Format for display
    print("=== Display-Formatted Rules (first 3 rows) ===")
    formatted = format_rules_display(rules)
    print(formatted.head(3).to_string(index=False))

    print("\n✅  Smoke test complete.\n")
    sys.exit(0)
