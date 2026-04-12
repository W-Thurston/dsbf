# tests/generators/dsbf_test_dataset_generator.py
"""
Generates all DSBF test datasets and writes them to tests/datasets/.

Each generator method returns a DataFrame ready to profile.

Usage:
    python tests/generators/dsbf_test_dataset_generator.py
    python tests/generators/dsbf_test_dataset_generator.py --only clean tiny
"""

from __future__ import annotations

import argparse
import base64
import random
import string
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)


# ── 1. Comprehensive (existing) ────────────────────────────────────────────────


class DSBFTestDataGenerator:
    """
    Generates a comprehensive, issue-burdened dataset covering every DSBF
    task category. Retained from the original generator with no changes.
    """

    def __init__(self, n_rows: int = 5000, random_seed: int = 42):
        self.n_rows = n_rows
        _seed(random_seed)

    def generate(self) -> pd.DataFrame:
        data: dict[str, list[Any]] = {}
        data.update(self._generate_normal_columns())
        data.update(self._generate_constant_columns())
        data.update(self._generate_duplicate_columns())
        data.update(self._generate_partial_duplicate_columns())
        data.update(self._generate_high_cardinality_columns())
        data.update(self._generate_id_columns())
        data.update(self._generate_mixed_type_columns())
        data.update(self._generate_encoded_columns())
        data.update(self._generate_outlier_columns())
        data.update(self._generate_skewed_columns())
        data.update(self._generate_bimodal_columns())
        data.update(self._generate_zero_variance_columns())
        data.update(self._generate_sparse_columns())
        data.update(self._generate_datetime_columns())
        data.update(self._generate_text_columns())
        data.update(self._generate_boolean_columns())
        data.update(self._generate_ambiguous_boolean_column())
        data.update(self._generate_correlation_columns())
        data.update(self._generate_target_columns())
        data.update(self._generate_leakage_columns())
        data.update(self._generate_out_of_bounds_columns())
        data.update(self._generate_regex_violation_columns())
        data.update(self._generate_null_oddities())
        data.update(self._generate_unexpected_type_column())

        df = pd.DataFrame(data)
        # Add ~5% duplicate rows
        dup_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
        return df

    # ── internal generators (unchanged from original) ──────────────────────────

    def _generate_normal_columns(self):
        return {
            "normal_numeric_int": np.random.randint(1, 100, self.n_rows).tolist(),
            "normal_numeric_float": np.random.normal(50, 15, self.n_rows).tolist(),
            "normal_categorical": np.random.choice(
                ["A", "B", "C", "D"], self.n_rows
            ).tolist(),
            "normal_text": [f"Item_{i}" for i in range(self.n_rows)],
            "normal_boolean": np.random.choice([True, False], self.n_rows).tolist(),
            "normal_date": [
                datetime(2023, 1, 1) + timedelta(days=i % 365)
                for i in range(self.n_rows)
            ],
        }

    def _generate_constant_columns(self):
        return {
            "constant_numeric": [42] * self.n_rows,
            "constant_string": ["SAME_VALUE"] * self.n_rows,
            "constant_null": [None] * self.n_rows,
        }

    def _generate_duplicate_columns(self):
        base = np.random.randint(1, 50, self.n_rows)
        return {
            "duplicate_col_1": base.tolist(),
            "duplicate_col_2": base.copy().tolist(),
            "duplicate_col_3": base.copy().tolist(),
        }

    def _generate_partial_duplicate_columns(self):
        base = np.random.randint(0, 100, self.n_rows)
        noisy = base.copy().astype(float)
        idx = np.random.choice(self.n_rows, int(0.1 * self.n_rows), replace=False)
        noisy[idx] += np.random.normal(0, 1, len(idx))
        return {
            "partial_duplicate_1": base.tolist(),
            "partial_duplicate_2": noisy.tolist(),
        }

    def _generate_high_cardinality_columns(self):
        return {
            "high_cardinality_uuid": [str(uuid.uuid4()) for _ in range(self.n_rows)],
            "high_cardinality_random": [
                f"VAL_{random.randint(1, self.n_rows)}" for _ in range(self.n_rows)
            ],
        }

    def _generate_id_columns(self):
        return {
            "user_id": [f"USER_{i:06d}" for i in range(self.n_rows)],
            "transaction_id": [
                f"TXN_{uuid.uuid4().hex[:8].upper()}" for _ in range(self.n_rows)
            ],
            "sequential_id": list(range(1, self.n_rows + 1)),
        }

    def _generate_mixed_type_columns(self):
        vals = []
        for i in range(self.n_rows):
            if i % 10 == 0:
                vals.append(None)
            elif i % 7 == 0:
                vals.append(f"STRING_{i}")
            else:
                vals.append(random.uniform(1, 100))
        return {"mixed_type_column": vals}

    def _generate_encoded_columns(self):
        return {
            "base64_encoded": [
                base64.b64encode(f"data_{i}".encode()).decode()
                for i in range(self.n_rows)
            ],
            "hex_encoded": [f"{i:08x}" for i in range(self.n_rows)],
            "uuid_column": [str(uuid.uuid4()) for _ in range(self.n_rows)],
        }

    def _generate_outlier_columns(self):
        data = np.random.normal(50, 10, self.n_rows)
        idx = np.random.choice(self.n_rows, size=int(0.02 * self.n_rows), replace=False)
        data[idx] = np.random.choice([500, -500, 1000], len(idx))
        return {"outlier_column": data.tolist()}

    def _generate_skewed_columns(self):
        return {
            "right_skewed": np.random.exponential(2, self.n_rows).tolist(),
            "left_skewed": (-np.random.exponential(2, self.n_rows)).tolist(),
            "extreme_skew": np.random.pareto(0.1, self.n_rows).tolist(),
        }

    def _generate_bimodal_columns(self):
        m1 = np.random.normal(20, 5, self.n_rows // 2)
        m2 = np.random.normal(80, 5, self.n_rows - self.n_rows // 2)
        data = np.concatenate([m1, m2])
        np.random.shuffle(data)
        return {"bimodal_column": data.tolist()}

    def _generate_zero_variance_columns(self):
        base = 100.0
        noise = np.random.normal(0, 0.0001, self.n_rows)
        return {
            "near_zero_variance": [base] * (self.n_rows - 10) + list(base + noise[:10])
        }

    def _generate_sparse_columns(self):
        sparse = np.zeros(self.n_rows)
        idx = np.random.choice(self.n_rows, size=int(0.05 * self.n_rows), replace=False)
        sparse[idx] = np.random.normal(10, 3, len(idx))
        return {
            "sparse_column": sparse.tolist(),
            "mostly_zeros": [0] * int(0.9 * self.n_rows) + [1] * int(0.1 * self.n_rows),
        }

    def _generate_datetime_columns(self):
        return {
            "datetime_with_gaps": [
                (
                    (
                        datetime(2023, 1, 1) + timedelta(days=i * random.randint(1, 30))
                    ).isoformat()
                    if i % 50 != 0
                    else None
                )
                for i in range(self.n_rows)
            ],
            "inconsistent_datetime": [
                random.choice(
                    [
                        "2023-01-15",
                        "15/01/2023",
                        "Jan 15 2023",
                        "20230115",
                        None,
                    ]
                )
                for _ in range(self.n_rows)
            ],
        }

    def _generate_text_columns(self):
        ctrl = [
            "text\x00with\x1bcontrols" if i % 50 == 0 else "clean text"
            for i in range(self.n_rows)
        ]
        long = [
            "A" * random.randint(200, 2000) if i % 100 == 0 else "short text"
            for i in range(self.n_rows)
        ]
        return {
            "short_text": [
                random.choice(["A", "B", "C", ""]) for _ in range(self.n_rows)
            ],
            "long_text": [
                " ".join(
                    random.choices(
                        ["lorem", "ipsum", "dolor", "sit", "amet"],
                        k=random.randint(10, 100),
                    )
                )
                for _ in range(self.n_rows)
            ],
            "variable_length_text": [
                "".join(random.choices(string.ascii_letters, k=random.randint(1, 200)))
                for _ in range(self.n_rows)
            ],
            "extremely_long_strings": long,
            "control_character_text": ctrl,
        }

    def _generate_boolean_columns(self):
        return {
            "imbalanced_boolean": (
                [True] * int(0.95 * self.n_rows) + [False] * int(0.05 * self.n_rows)
            ),
            "mixed_boolean_types": [
                random.choice([True, False, 1, 0, "True", "False", "Y", "N"])
                for _ in range(self.n_rows)
            ],
        }

    def _generate_ambiguous_boolean_column(self):
        return {
            "ambiguous_booleans": [
                random.choice(["yes", "no", "1", "0", True, False])
                for _ in range(self.n_rows)
            ]
        }

    def _generate_correlation_columns(self):
        base = np.random.normal(50, 15, self.n_rows)
        return {
            "corr_base": base.tolist(),
            "highly_correlated": (
                base * 2 + np.random.normal(0, 1, self.n_rows)
            ).tolist(),
            "moderately_correlated": (
                base + np.random.normal(0, 10, self.n_rows)
            ).tolist(),
            "negatively_correlated": (
                -base + np.random.normal(0, 5, self.n_rows)
            ).tolist(),
        }

    def _generate_target_columns(self):
        return {
            "balanced_target": np.random.choice(
                [0, 1], self.n_rows, p=[0.5, 0.5]
            ).tolist(),
            "imbalanced_target": np.random.choice(
                [0, 1], self.n_rows, p=[0.95, 0.05]
            ).tolist(),
            "multiclass_target": np.random.choice(
                [0, 1, 2, 3, 4], self.n_rows, p=[0.6, 0.2, 0.1, 0.08, 0.02]
            ).tolist(),
        }

    def _generate_leakage_columns(self):
        target = np.random.choice([0, 1], self.n_rows, p=[0.7, 0.3])
        return {
            "target_for_leakage": target.tolist(),
            "perfect_leakage": target.copy().tolist(),
            "future_leakage": [f"outcome_{t}_{random.randint(1, 10)}" for t in target],
            "subtle_leakage": (
                target * 100 + np.random.normal(0, 5, self.n_rows)
            ).tolist(),
        }

    def _generate_out_of_bounds_columns(self):
        ages = np.random.randint(18, 80, self.n_rows)
        ages[np.random.choice(self.n_rows, 50, replace=False)] = np.random.choice(
            [-5, 150, 999], 50
        )
        return {
            "age_with_bounds_issues": ages.tolist(),
            "percentage_out_of_bounds": np.random.uniform(
                -50, 150, self.n_rows
            ).tolist(),
            "negative_counts": np.random.randint(-10, 50, self.n_rows).tolist(),
        }

    def _generate_regex_violation_columns(self):
        emails = [
            f"invalid_email_{i}" if i % 20 == 0 else f"user{i}@example.com"
            for i in range(self.n_rows)
        ]
        phones = [
            (
                f"NOT_A_PHONE_{i}"
                if i % 15 == 0
                else f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            )
            for i in range(self.n_rows)
        ]
        return {"email_with_violations": emails, "phone_with_violations": phones}

    def _generate_null_oddities(self):
        vals = [
            None if i % 3 == 0 else np.nan if i % 3 == 1 else pd.NA
            for i in range(self.n_rows)
        ]
        return {
            "mixed_null_types": vals,
            "empty_vs_nan": [""] * int(self.n_rows * 0.9)
            + [np.nan] * int(self.n_rows * 0.1),
        }

    def _generate_unexpected_type_column(self):
        vals = [
            {"key": i} if i % 3 == 0 else [i, i + 1] if i % 3 == 1 else "normal"
            for i in range(self.n_rows)
        ]
        return {"unexpected_object_column": vals}


# ── 2. Clean dataset ───────────────────────────────────────────────────────────


def generate_clean_dataset(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    A perfectly well-behaved dataset with no error-level findings expected anywhere.

    Tests:
    - All Quality dimensions show green / nothing flagged
    - No error-level ML Readiness findings
    - Trust banner = 'Looking Good'
    - Empty states render correctly in every section

    Design principles:
    - Beta(5,5) for continuous columns: symmetric, bell-shaped, naturally bounded,
      analytically guaranteed to have near-zero IQR outliers (< 0.1%), near-zero
      skewness, and a clear single mode. Does not trigger bimodal detection.
      Uniform distributions were tried but caused false-positive bimodal flags since
      a flat distribution has no clear peak for the GMM test to identify.
    - No datetime column: datetime triggers high-cardinality encoding findings
      (1095 unique daily values = high cardinality) even when typed as datetime.
      Datetime handling is covered by the time_series dataset instead.
    - Categorical columns balanced to avoid dominant-value and high-cardinality flags.
    - No inter-column correlation to avoid leakage flags.

    Known scorer calibration issue (tracked, not a test failure):
    - The ML readiness gate maps any 'red' dimension → 'not_ready' regardless of
      finding severity. Encoding recommendations at 'good' level for categorical
      columns count toward the affected-column proportion and push encoding to 'red'.
      The assertion checks for error-level findings directly rather than trusting
      the gate value until the scorer calibration is fixed.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # Beta(5,5): symmetric bell-shape, bounded [0,1] by construction.
    # IQR outlier rate ≈ 0.05-0.1% (well below flag_threshold=1%).
    # Z-score outlier rate ≈ 0%. Skewness ≈ 0. Clearly unimodal.
    return pd.DataFrame(
        {
            "age": (rng.beta(5, 5, n) * 40 + 25).round().astype(int).tolist(),
            "income": (rng.beta(5, 5, n) * 80000 + 30000).round(2).tolist(),
            "score": (rng.beta(5, 5, n) * 0.6 + 0.2).round(4).tolist(),
            "tenure_years": (rng.beta(4, 5, n) * 12 + 1).round(1).tolist(),
            # Categorical — low cardinality, balanced
            "region": rng.choice(
                ["North", "South", "East", "West"], n, p=[0.25, 0.25, 0.25, 0.25]
            ).tolist(),
            "plan_type": rng.choice(
                ["Basic", "Standard", "Premium"], n, p=[0.33, 0.34, 0.33]
            ).tolist(),
            "department": rng.choice(
                ["Engineering", "Sales", "Support", "Marketing"],
                n,
                p=[0.28, 0.26, 0.23, 0.23],
            ).tolist(),
            # Boolean — balanced
            "is_active": rng.choice([True, False], n, p=[0.55, 0.45]).tolist(),
            "has_upgrade": rng.choice([True, False], n, p=[0.48, 0.52]).tolist(),
        }
    )


# ── 3. Tiny dataset ────────────────────────────────────────────────────────────


def generate_tiny_dataset(n_rows: int = 25, seed: int = 42) -> pd.DataFrame:
    """
    A very small dataset (25 rows). Tests graceful degradation on tasks
    that assume sufficient sample size.

    Tasks expected to degrade gracefully (not error):
    - normality_tests (low power, result may be unreliable)
    - detect_bimodal_distribution (BIC fit may be skipped)
    - compute_pairwise_associations (few pairs)
    - isolation_forest (may skip if n < threshold)
    - vif computation (may be unstable)
    - fuzzy_duplicate_detection (small population)

    Tests:
    - No task status == "error" (degradation != failure)
    - Sample size adequacy warning appears in Overview
    - Percentile table renders with sparse data
    - Guidance is still coherent despite limited evidence
    """
    _seed(seed)
    n = n_rows
    return pd.DataFrame(
        {
            "x": np.random.normal(10, 3, n).round(2).tolist(),
            "y": np.random.normal(20, 5, n).round(2).tolist(),
            "z": (np.random.normal(10, 3, n) * 2 + np.random.normal(0, 1, n))
            .round(2)
            .tolist(),
            "category": np.random.choice(["A", "B", "C"], n).tolist(),
            "flag": np.random.choice([True, False], n).tolist(),
            "signup": [datetime(2023, 6, 1) + timedelta(days=i) for i in range(n)],
        }
    )


# ── 4. Near-clean dataset ("realistic production quality") ─────────────────────


def generate_near_clean_dataset(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Mostly well-behaved data with a precise set of deliberate, realistic issues.

    Each issue targets a specific DSBF finding and is documented below so
    assertions can be written against known expected outputs.

    Deliberate issues (and their expected findings):
    - ``income``: 7% nulls → Completeness amber (warn)
    - ``purchase_amount``: exponential distribution, skew ≈ 2.2 →
      log-transform recommended (Transformations warn)
    - ``plan_type``: "Standard" = 71% of values → single dominant value
      (Usability warn). Still below the error threshold (≥ 95%) so it
      should not block the ML Readiness gate.
    - ``region``, ``plan_type``, ``segment``: raw object dtype →
      Encoding Required warn (sklearn cannot ingest without encoding)

    Intentionally absent (to test "all clear" states):
    - No ID columns (customer_id uses low-cardinality hashed tokens, not
      sequential integers — avoids triggering detect_id_columns)
    - No leakage (no near-perfectly correlated column pairs)
    - No constants, no zero-variance columns
    - No out-of-bounds values
    - No duplicate columns
    - No datetime column (avoids high-cardinality encoding noise)

    Expected scoring:
    - Quality: Completeness amber, Usability amber, all others green
    - ML Readiness: gate = needs_work (warn findings, no errors)
    - Trust banner: "A Few Things to Note" (amber, not red)

    Edge cases probed:
    - Auto-open logic in Quality tab: only Completeness and Usability
      sections should open; the other three stay collapsed
    - The ML Readiness gate distinguishes needs_work from not_ready
      (no error-level findings present)
    - Scoring thresholds: 7% null is above the 5% amber threshold but
      only one column is affected — tests proportional amber calibration
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ── Continuous columns ─────────────────────────────────────────────────────

    # Clean bell-shaped columns — Beta(5,5) guarantees skew ≈ 0, no outliers
    age = (rng.beta(5, 5, n) * 40 + 25).round().astype(int).tolist()
    tenure_months = (rng.beta(3, 4, n) * 58 + 1).round().astype(int).tolist()
    satisfaction = (rng.beta(5, 5, n) * 8 + 1).round(1).tolist()

    # Deliberately skewed — exponential produces skew ≈ 2.2, well above
    # the log-transform threshold of 1.0
    purchase_amount = rng.exponential(scale=50, size=n).round(2).tolist()

    # 7% missingness — above the 5% amber threshold for completeness
    income_raw = rng.normal(60000, 15000, n)
    income_raw = np.clip(income_raw, 15000, 200_000).round(2)
    null_mask = rng.random(n) < 0.07
    income: list = [None if null_mask[i] else float(income_raw[i]) for i in range(n)]

    # ── Categorical columns ────────────────────────────────────────────────────

    # Balanced low-cardinality — clean, only advisory encoding notes
    region = rng.choice(
        ["North", "South", "East", "West"],
        n,
        p=[0.25, 0.25, 0.25, 0.25],
    ).tolist()

    # Deliberately dominant — "Standard" ≈ 71%
    # Above the warn threshold (≥ 70%) but below error (≥ 95%)
    plan_type = rng.choice(
        ["Basic", "Standard", "Premium"],
        n,
        p=[0.145, 0.710, 0.145],
    ).tolist()

    # A third categorical to give encoding three columns to flag
    segment = rng.choice(
        ["Enterprise", "SMB", "Consumer"],
        n,
        p=[0.30, 0.40, 0.30],
    ).tolist()

    # Boolean — clean, balanced
    is_churned = rng.choice([True, False], n, p=[0.15, 0.85]).tolist()

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "purchase_amount": purchase_amount,
            "tenure_months": tenure_months,
            "satisfaction": satisfaction,
            "region": region,
            "plan_type": plan_type,
            "segment": segment,
            "is_churned": is_churned,
        }
    )


# ── 5. All-categorical dataset ─────────────────────────────────────────────────


def generate_all_categorical_dataset(
    n_rows: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """
    A dataset with zero continuous columns — only categorical, boolean, and
    one high-cardinality string column.

    Primary purpose: verify that every continuous-only task degrades cleanly
    (status='success' with empty output) rather than erroring when it finds
    no columns to process.

    Deliberate column design:
    - ``color``       4 balanced values — clean low-cardinality categorical
    - ``size``        3 balanced values — clean low-cardinality categorical
    - ``material``    5 balanced values — clean low-cardinality categorical
    - ``region``      4 balanced values — clean low-cardinality categorical
    - ``category``    6 balanced values — borderline for one-hot (still ≤ 10)
    - ``tag``         ~500 unique free-text tags — triggers detect_high_cardinality
                      (warn) and frequency encoding suggestion
    - ``dominant``    "Standard" = 96% of rows — triggers detect_single_dominant_value
                      at error level (≥ 95% threshold)
    - ``is_active``   boolean — exercises summarize_boolean_fields path
    - ``is_premium``  boolean — second boolean for association table coverage

    Expected findings:
    - Usability ``error``: dominant column (96% single value)
    - Encoding ``warn``:   tag high-cardinality + raw string encoding required
    - All continuous-only tasks: status='success', empty/skipped output
    - Outlier, normality, skewness, VIF, bimodal sections: nothing to show
    - Association table: Cramér's V only — no Pearson r, no eta squared

    Intentionally absent:
    - No numeric columns of any kind (no int, float, continuous)
    - No datetime columns
    - No nulls (tests that empty-state rendering is not caused by missingness)
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # Balanced low-cardinality categoricals — clean baseline
    color = rng.choice(
        ["Red", "Blue", "Green", "Yellow"],
        n,
        p=[0.25, 0.25, 0.25, 0.25],
    ).tolist()
    size = rng.choice(
        ["Small", "Medium", "Large"],
        n,
        p=[0.33, 0.34, 0.33],
    ).tolist()
    material = rng.choice(
        ["Wood", "Metal", "Plastic", "Glass", "Fabric"],
        n,
        p=[0.20, 0.20, 0.20, 0.20, 0.20],
    ).tolist()
    region = rng.choice(
        ["North", "South", "East", "West"],
        n,
        p=[0.25, 0.25, 0.25, 0.25],
    ).tolist()
    category = rng.choice(
        ["A", "B", "C", "D", "E", "F"],
        n,
        p=[0.17, 0.17, 0.17, 0.17, 0.16, 0.16],
    ).tolist()

    # High-cardinality string column — ~500 unique tags from a pool of 600.
    # cardinality > default high_cardinality_threshold (50) triggers warn-level
    # encoding finding and a frequency encoding suggestion.
    tag_pool = [f"tag_{i:03d}" for i in range(600)]
    tag = rng.choice(tag_pool, n).tolist()

    # Dominant column — 96% "Standard", well above the 95% error threshold.
    # This is the only column expected to produce an error-level finding.
    dominant = rng.choice(["Standard", "Other"], n, p=[0.96, 0.04]).tolist()

    # Boolean columns — exercises the boolean encoding and stats paths
    is_active = rng.choice([True, False], n, p=[0.60, 0.40]).tolist()
    is_premium = rng.choice([True, False], n, p=[0.25, 0.75]).tolist()

    return pd.DataFrame(
        {
            "color": color,
            "size": size,
            "material": material,
            "region": region,
            "category": category,
            "tag": tag,
            "dominant": dominant,
            "is_active": is_active,
            "is_premium": is_premium,
        }
    )


# ── 6. High-missingness dataset ────────────────────────────────────────────────


def generate_high_missingness_dataset(
    n_rows: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """
    A dataset with extreme and structured missing data across multiple columns.

    Each missing-data pattern is deliberately different so that the
    missingness mechanism analysis can detect structural patterns, and
    so that imputation suggestions are exercised across a range of null rates.

    Null pattern inventory:
    ┌─────────────────┬──────────┬──────────────────────────────────────────┐
    │ Column          │ Null %   │ Pattern / reason                         │
    ├─────────────────┼──────────┼──────────────────────────────────────────┤
    │ age             │ ~8%      │ MCAR — random dropout, above amber (5%)  │
    │ income          │ ~25%     │ MCAR — moderate random dropout           │
    │ device_type     │ ~60%     │ MAR — null when channel == "web"         │
    │ notes           │ ~75%     │ High sparse — tests drop-vs-impute logic │
    │ premium_score   │ ~50%     │ MAR — null when plan_type == "Basic"     │
    │ region          │ 0%       │ Clean baseline categorical               │
    │ channel         │ 0%       │ Clean — drives device_type missingness   │
    │ plan_type       │ 0%       │ Clean — drives premium_score missingness │
    │ is_churned      │ 0%       │ Clean boolean baseline                   │
    └─────────────────┴──────────┴──────────────────────────────────────────┘

    The MAR patterns (device_type, premium_score) are deliberately structured
    so that missingness mechanism analysis can detect the correlation:
    - P(device_type = null | channel = "web") ≈ 0.95
    - P(premium_score = null | plan_type = "Basic") ≈ 0.90

    Expected findings:
    - Completeness: red (device_type 60%, notes 75%, income 25% all above
      the amber/red thresholds)
    - ML Readiness Missingness: red or amber (high null rates affect model
      training directly — rows with nulls are dropped by most sklearn estimators)
    - Imputation suggestions: "drop or use indicator" for notes (75%);
      "impute with median/mean" for income (25%)
    - Missingness mechanism: MAR signals for device_type and premium_score
    - Reliability warnings on summarize_numeric for high-null columns
      (statistics computed on 20–40% of data)
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ── Clean baseline columns (0% null) ──────────────────────────────────────

    channel = rng.choice(
        ["web", "mobile", "email", "direct"],
        n,
        p=[0.40, 0.30, 0.20, 0.10],
    ).tolist()

    plan_type = rng.choice(
        ["Basic", "Standard", "Premium"],
        n,
        p=[0.40, 0.35, 0.25],
    ).tolist()

    region = rng.choice(
        ["North", "South", "East", "West"],
        n,
        p=[0.25, 0.25, 0.25, 0.25],
    ).tolist()

    is_churned = rng.choice([True, False], n, p=[0.20, 0.80]).tolist()

    # ── MCAR columns ──────────────────────────────────────────────────────────

    # age: Beta(5,5) distribution, ~8% MCAR null — just above amber threshold
    age_raw = (rng.beta(5, 5, n) * 40 + 25).round().astype(int)
    age_mask = rng.random(n) < 0.08
    age: list = [None if age_mask[i] else int(age_raw[i]) for i in range(n)]

    # income: ~25% MCAR null — moderate dropout
    income_raw = rng.normal(60000, 20000, n)
    income_raw = np.clip(income_raw, 15000, 200_000).round(2)
    income_mask = rng.random(n) < 0.25
    income: list = [None if income_mask[i] else float(income_raw[i]) for i in range(n)]

    # ── MAR columns — missingness correlated with other columns ───────────────

    # device_type: null with P≈0.95 when channel=="web", P≈0.05 otherwise.
    # Web sessions don't always report device; other channels always do.
    device_pool = ["Desktop", "Mobile", "Tablet"]
    device_type: list = []
    for i in range(n):
        if channel[i] == "web":
            device_type.append(None if rng.random() < 0.95 else rng.choice(device_pool))
        else:
            device_type.append(None if rng.random() < 0.05 else rng.choice(device_pool))

    # premium_score: null with P≈0.90 when plan_type=="Basic".
    # Basic plan users don't have a premium score computed.
    premium_score_raw = rng.beta(3, 2, n) * 100
    premium_score: list = []
    for i in range(n):
        if plan_type[i] == "Basic":
            premium_score.append(
                None if rng.random() < 0.90 else round(float(premium_score_raw[i]), 2)
            )
        else:
            premium_score.append(
                None if rng.random() < 0.05 else round(float(premium_score_raw[i]), 2)
            )

    # ── Highly sparse column ──────────────────────────────────────────────────

    # notes: ~75% null — tests imputation logic at extreme sparsity.
    # The 25% present values are short free-text strings.
    note_options = [
        "Follow up needed",
        "VIP customer",
        "Billing issue",
        "Product feedback",
        "Support escalation",
        "No issues",
    ]
    notes_mask = rng.random(n) < 0.75
    notes: list = [
        None if notes_mask[i] else str(rng.choice(note_options)) for i in range(n)
    ]

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "device_type": device_type,
            "premium_score": premium_score,
            "notes": notes,
            "region": region,
            "channel": channel,
            "plan_type": plan_type,
            "is_churned": is_churned,
        }
    )


# ── 7. Severe multicollinearity dataset ────────────────────────────────────────


def generate_severe_multicollinearity_dataset(
    n_rows: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """
    All-continuous dataset targeting four distinct VIF failure modes.

    Each column block exercises a different way VIF computation can go
    wrong or produce misleading results in practice.

    ── Block A: Classic severe collinearity ─────────────────────────────────
    Four columns that form an obvious linear cluster.  These test the
    happy path — VIF should correctly flag all four.

      base        N(0,1) anchor
      copy_a      r≈0.99 with base  → VIF ~84 (correctly flagged)
      copy_b      r≈0.96 with base  → VIF ~16 (correctly flagged)
      combo       0.7*copy_a + 0.3*copy_b + noise → VIF ~45 (flagged)

    ── Block B: Scale-mismatch (exposes the add_constant bug) ───────────────
    Two genuinely independent columns at extreme different scales.
    Under the broken code (no add_constant), their large nonzero means
    inflate VIF to 6–8, producing spurious warn-level findings.
    Under the fixed code, both score VIF ≈ 1.0.

      revenue     N(500_000, 200_000) — mean 500k, independent
      click_rate  N(0.035, 0.012)     — mean 0.035, independent

    ── Block C: All-collinear trio (no independent column) ──────────────────
    Three columns all derived from the same latent factor.  VIF is high
    for all three — tests that the task correctly handles the case where
    there is no "clean" reference column in the subgroup.

      factor_x    latent + noise
      factor_y    0.95*latent + noise
      factor_z    1.05*latent + noise

    ── Block D: Sparse continuous column (dropna reduces sample size) ────────
    One column with 65% NaN.  After dropna(), VIF is computed on only
    ~35% of rows (≈700 rows).  Among those observed rows, signal and
    sparse_c are correlated — VIF correctly reflects this.  The task
    should emit a low_row_count reliability warning.

      signal      N(50, 10) — fully observed
      sparse_c    1.1*signal + noise, 65% NaN

    Expected findings (with the add_constant fix applied):
    - Block A: base, copy_a, copy_b, combo all flagged (VIF > 10)
    - Block B: revenue and click_rate NOT flagged (VIF ≈ 1.0)
    - Block C: factor_x, factor_y, factor_z all flagged (VIF > 10)
    - Block D: signal and sparse_c flagged among observed rows (VIF ~36);
               reliability warning emitted for reduced sample size
    - Redundancy dimension: red

    Under the broken code (no add_constant, select_dtypes):
    - Block B: revenue VIF ≈ 6, click_rate VIF ≈ 8 (spurious warn findings)
    - This is the live bug this dataset is designed to detect
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ── Block A: Classic severe collinearity ──────────────────────────────────
    base = rng.normal(0, 1, n)
    copy_a = 0.99 * base + rng.normal(0, 0.14, n)
    copy_b = 0.96 * base + rng.normal(0, 0.28, n)
    combo = 0.7 * copy_a + 0.3 * copy_b + rng.normal(0, 0.15, n)

    # ── Block B: Scale-mismatch — tests add_constant fix ─────────────────────
    # Independent columns at very different scales and nonzero means.
    # Without add_constant the intercept-less VIF regression treats their
    # large means as shared variance → spurious VIF inflation.
    revenue = rng.normal(500_000, 200_000, n)
    click_rate = rng.normal(0.035, 0.012, n)

    # ── Block C: All-collinear trio ───────────────────────────────────────────
    latent = rng.normal(0, 1, n)
    factor_x = latent + rng.normal(0, 0.12, n)
    factor_y = 0.95 * latent + rng.normal(0, 0.20, n)
    factor_z = 1.05 * latent + rng.normal(0, 0.15, n)

    # ── Block D: Sparse continuous — dropna reduces to ~35% of rows ──────────
    signal = rng.normal(50, 10, n)
    sparse_c = 1.1 * signal + rng.normal(0, 2, n)
    nan_mask = rng.random(n) < 0.65
    sparse_c = np.where(nan_mask, np.nan, sparse_c)

    return pd.DataFrame(
        {
            # Block A
            "base": np.round(base, 6).tolist(),
            "copy_a": np.round(copy_a, 6).tolist(),
            "copy_b": np.round(copy_b, 6).tolist(),
            "combo": np.round(combo, 6).tolist(),
            # Block B
            "revenue": np.round(revenue, 2).tolist(),
            "click_rate": np.round(click_rate, 6).tolist(),
            # Block C
            "factor_x": np.round(factor_x, 6).tolist(),
            "factor_y": np.round(factor_y, 6).tolist(),
            "factor_z": np.round(factor_z, 6).tolist(),
            # Block D
            "signal": np.round(signal, 4).tolist(),
            "sparse_c": [None if np.isnan(v) else round(float(v), 4) for v in sparse_c],
        }
    )


# ── 8. Single-column dataset ──────────────────────────────────────────────────


def generate_single_column_dataset(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    A dataset with exactly one column — a continuous numeric feature.

    Primary purpose: verify that every task which requires two or more
    columns degrades cleanly (empty output, status='success') rather than
    raising an unhandled exception.

    500 rows is chosen deliberately:
    - Large enough for per-column tasks (outliers, skewness, normality)
      to run meaningfully and produce real findings
    - Small enough to keep profiling fast
    - Well above the tiny-dataset low-row-count threshold (N < 30)

    The single column (``value``) is a mildly right-skewed log-normal
    distribution.  This ensures:
    - detect_skewness fires with a real finding (skew > 1.0)
    - detect_outliers has extreme values to flag
    - normality_tests can run (not just empty)
    - summarize_numeric produces real stats (mean, std, percentiles)

    Tasks expected to produce empty/minimal output (correct behavior):
    - detect_collinear_features: "Not enough numeric features" (< 2 cols)
    - compute_pairwise_associations: 0 pairs computed
    - detect_duplicate_columns: 0 pairs to compare
    - generate_dataset_summary_plots: correlation matrix skipped (< 2 cols)
    - compute_mutual_information: needs categorical column, skips
    - one_way_anova / kruskal_wallis: needs categorical grouping column
    - detect_data_leakage: no pairs to check

    None of these should error — they should all return status='success'
    with empty data structures.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # Log-normal: naturally right-skewed (skew > 1.0 triggers log-transform warn)
    # Using exp(normal) gives skew ≈ 6 for sigma=1, but we want skew ≈ 1.5-2.5
    # so use sigma=0.6 which gives skew ≈ 2.0
    value = rng.lognormal(mean=3.5, sigma=0.6, size=n).round(2)

    return pd.DataFrame({"value": value.tolist()})


# ── Entry point ────────────────────────────────────────────────────────────────

GENERATORS = {
    "comprehensive": lambda: DSBFTestDataGenerator(n_rows=5000).generate(),
    "clean": generate_clean_dataset,
    "tiny": generate_tiny_dataset,
    "near_clean": generate_near_clean_dataset,
    "all_categorical": generate_all_categorical_dataset,
    "high_missingness": generate_high_missingness_dataset,
    "severe_multicollinearity": generate_severe_multicollinearity_dataset,
    "single_column": generate_single_column_dataset,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSBF test datasets")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(GENERATORS),
        default=list(GENERATORS),
        help="Which datasets to generate (default: all)",
    )
    args = parser.parse_args()

    for name in args.only:
        print(f"Generating {name}…")
        df = GENERATORS[name]()
        out = DATASETS_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  → {out}  ({len(df):,} rows × {len(df.columns)} columns)")

    print("\nDone. Load any dataset with:")
    print("  dsbf profile tests/datasets/<name>.csv")


if __name__ == "__main__":
    main()
