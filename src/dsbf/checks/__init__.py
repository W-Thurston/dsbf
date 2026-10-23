# dsbf/checks/__init__.py
"""The ``core`` pack: the checks DSBF ships.

A check is declared with ``@check`` at the top level of its own module here, so a
worker process can import it by name, and ``core_pack`` lists it. Nothing registers
on import: ``dsbf.profile()`` and ``dsbf plan`` call ``Registry.register_pack``.

Conventions every core check follows:

- A check reads the columns ``applies_to`` selects, through ``ctx.columns``, the
  profile and ``ctx.values``, and never picks columns by dtype itself.
- A check never calls ``warnings.catch_warnings()``. The engine records a warning a
  check raises as a ``python_warning`` note with a count; the block would also hide
  the warnings of the other checks running beside it.
- A check gives scipy numeric arrays, never object dtype: scipy checks an object
  array for NaN one element at a time. It fits scikit-learn models on float64:
  `GaussianMixture` mis-scores integer input, differently in 1.7 and 1.8.
- A check that computes with extreme values skips a column whose arithmetic
  overflows, saying why, and never puts a NaN or an infinity in its output.
- A check states ``n_used`` and the method on every finding's evidence, declares the
  bounds of each parameter, and leaves recommendations to the tips attached after it
  returns.
"""

from dsbf.checks.associations import associations
from dsbf.checks.autocorrelation import autocorrelation
from dsbf.checks.bimodality import bimodality
from dsbf.checks.class_imbalance import class_imbalance
from dsbf.checks.collinearity import collinearity
from dsbf.checks.constant_columns import constant_columns
from dsbf.checks.contingency import contingency
from dsbf.checks.datetime_consistency import datetime_consistency
from dsbf.checks.datetime_features import datetime_features
from dsbf.checks.dominant_value import dominant_value
from dsbf.checks.dtype_optimizations import dtype_optimizations
from dsbf.checks.duplicate_columns import duplicate_columns
from dsbf.checks.duplicate_rows import duplicate_rows
from dsbf.checks.encoded_columns import encoded_columns
from dsbf.checks.entropy import entropy
from dsbf.checks.format_violations import format_violations
from dsbf.checks.fuzzy_duplicates import fuzzy_duplicates
from dsbf.checks.group_differences import group_differences
from dsbf.checks.high_cardinality import high_cardinality
from dsbf.checks.id_columns import id_columns
from dsbf.checks.kurtosis import kurtosis
from dsbf.checks.missingness_mechanism import missingness_mechanism
from dsbf.checks.mixed_types import mixed_types
from dsbf.checks.multivariate_outliers import multivariate_outliers
from dsbf.checks.near_zero_variance import near_zero_variance
from dsbf.checks.normality import normality
from dsbf.checks.null_rate import null_rate
from dsbf.checks.out_of_bounds import out_of_bounds
from dsbf.checks.outliers import outliers
from dsbf.checks.sample_size import sample_size
from dsbf.checks.schema import schema
from dsbf.checks.seasonal_decomposition import seasonal_decomposition
from dsbf.checks.skewness import skewness
from dsbf.checks.stationarity import stationarity
from dsbf.checks.string_anomalies import string_anomalies
from dsbf.checks.temporal_gaps import temporal_gaps
from dsbf.checks.text_encoding import text_encoding
from dsbf.checks.zeros import zeros
from dsbf.engine import Pack

CORE_PACK = "core"

__all__ = ["CORE_PACK", "core_pack"]


def core_pack() -> Pack:
    """The core checks, as a pack to register explicitly."""
    return Pack(
        CORE_PACK,
        (
            zeros,
            null_rate,
            constant_columns,
            high_cardinality,
            id_columns,
            dominant_value,
            near_zero_variance,
            duplicate_rows,
            duplicate_columns,
            text_encoding,
            skewness,
            kurtosis,
            entropy,
            dtype_optimizations,
            sample_size,
            class_imbalance,
            schema,
            outliers,
            multivariate_outliers,
            normality,
            bimodality,
            out_of_bounds,
            associations,
            contingency,
            group_differences,
            collinearity,
            string_anomalies,
            encoded_columns,
            mixed_types,
            format_violations,
            missingness_mechanism,
            fuzzy_duplicates,
            stationarity,
            autocorrelation,
            seasonal_decomposition,
            datetime_consistency,
            temporal_gaps,
            datetime_features,
        ),
    )
