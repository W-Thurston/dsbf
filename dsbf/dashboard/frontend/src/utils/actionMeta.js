/**
 * dsbf/dashboard/frontend/src/utils/actionMeta.js
 *
 * Central knowledge base for ML Readiness action metadata.
 * Imported by MlReadinessTab and TransformationPreviewCard.
 *
 * TRANSFORM_META  — per-transform metadata for TransformationPreviewCard
 * SKIP_CONTEXT    — enriched skip-reason explanations
 * ACTION_META     — per-action metadata for suggested action rows
 * actionMeta()    — lookup helper
 * normalizeMethod() — string normalisation for lookup keys
 */

// ── Helpers ────────────────────────────────────────────────────────────────────

export function normalizeMethod(s) {
  return (s ?? '').toLowerCase()
    .replace(/-/g, '_')
    .replace(/\./g, '_')
    .replace(/\s+/g, '_')
}

// ── Per-transform metadata (for TransformationPreviewCard) ─────────────────────

export const TRANSFORM_META = {
  log1p: {
    what:     'Takes the natural log of (1 + x). Adding 1 before the log preserves zero values and compresses right-skewed tails without distorting the distribution shape.',
    tradeoff: 'Very effective at reducing right skew. Requires all values to be ≥ 0 — negative values cannot be transformed. If the column has negatives, use Yeo-Johnson instead.',
    after:    'Values are on a log scale. A one-unit difference represents a multiplicative change in the original. The column range will be much narrower.',
    ref:      { label: 'numpy.log1p', url: 'https://numpy.org/doc/stable/reference/generated/numpy.log1p.html' },
  },
  sqrt: {
    what:     'Takes the square root of each value. A milder compression than log — useful when log1p over-corrects or when the column has many zeros.',
    tradeoff: 'Less aggressive skew reduction than log1p. Also requires all values to be ≥ 0. A good middle ground for count data or lightly right-skewed columns.',
    after:    'Values are on a square root scale. Units become the square root of the originals, which are less intuitive to interpret directly.',
    ref:      { label: 'numpy.sqrt', url: 'https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html' },
  },
  square: {
    what:     'Squares each value (x²). Amplifies large positive values and compresses values near zero. Designed for left-skewed distributions.',
    tradeoff: 'Can fix left skew but dramatically amplifies outliers and changes units to the square of the originals. Rarely the right choice when right skew is the concern.',
    after:    'Values are on a squared scale. The column range and variance will be much larger than before.',
    ref:      null,
  },
  yeo_johnson: {
    what:     'A generalised power transformation that works on any distribution, including those with negative values. The λ parameter is estimated automatically from the data using maximum likelihood.',
    tradeoff: 'The most flexible option — handles positive, zero, and negative values. The transformation is data-specific and the output is harder to interpret directly without knowing the estimated λ.',
    after:    'Values are on a power-transformed scale. The specific transformation depends on the fitted λ — save it in your preprocessing pipeline to invert the transform later.',
    ref:      { label: 'sklearn PowerTransformer', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html' },
  },
  box_cox: {
    what:     'A power transformation similar to Yeo-Johnson but requiring all values to be strictly positive (> 0). The λ parameter is estimated automatically to maximise normality.',
    tradeoff: 'Often achieves better normality than Yeo-Johnson when applicable, since it has one fewer constraint. Fails on zero or negative values — use Yeo-Johnson for those cases.',
    after:    'Values are on a power-transformed scale determined by the estimated λ. The transformation is reversible if you save the fitted λ value.',
    ref:      { label: 'scipy.stats.boxcox', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html' },
  },
}

// ── Enriched skip-reason explanations ─────────────────────────────────────────

export const SKIP_CONTEXT = {
  'requires all values >= 0':
    'This transform requires all values to be ≥ 0, but this column contains negative values. ' +
    'Consider Yeo-Johnson, which handles negative values natively.',
  'requires all values > 0':
    'This transform requires all values to be strictly positive (> 0), but this column contains ' +
    'zeros or negative values. Consider Yeo-Johnson as a drop-in replacement.',
  'high_skewness':
    'Skipped because the column skewness is too extreme for a reliable fit.',
}

export function enrichSkipReason(skipReason) {
  return SKIP_CONTEXT[skipReason] ?? skipReason ?? 'This transform could not be applied to this column.'
}

// ── Per-action metadata (for suggested action rows) ────────────────────────────

export const ACTION_META = {

  // ── Transformations ──────────────────────────────────────────────────────────

  log1p: {
    what:     'Takes the natural log of (1 + x), compressing large values and right-skewed tails while preserving zero values.',
    tradeoff: 'Effectively reduces right skew and outlier influence. Requires all values ≥ 0; negative values cannot be transformed.',
    after:    'Values are on a log scale. Differences between values represent multiplicative changes in the original.',
    ref:      { label: 'numpy.log1p', url: 'https://numpy.org/doc/stable/reference/generated/numpy.log1p.html' },
  },
  sqrt: {
    what:     'Takes the square root of each value. A milder compression than log, suitable for moderate right skew.',
    tradeoff: 'Less aggressive than log1p; useful when log over-corrects. Requires all values ≥ 0.',
    after:    'Values are on a square root scale. Units become the square root of the originals, which is less intuitive to interpret.',
    ref:      { label: 'numpy.sqrt', url: 'https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html' },
  },
  square: {
    what:     'Squares each value (x²). Amplifies large positive values and compresses values near zero.',
    tradeoff: 'Useful for mild left skew. Dramatically amplifies outliers and changes units to the square of the original.',
    after:    'Values are on a squared scale. The column range and variance will be much larger than before.',
    ref:      null,
  },
  yeo_johnson: {
    what:     'A generalised power transformation that handles negative values. The λ parameter is estimated automatically from the data.',
    tradeoff: 'Works on any distribution including those with negative values. The transformation is data-specific, making the output harder to interpret directly.',
    after:    'Values are on a power-transformed scale. The specific transformation depends on the estimated λ — check the fitted value.',
    ref:      { label: 'sklearn PowerTransformer', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html' },
  },
  box_cox: {
    what:     'A power transformation similar to Yeo-Johnson but requiring strictly positive values. The λ parameter is estimated automatically.',
    tradeoff: 'Often achieves better normality than Yeo-Johnson when applicable. Fails on zero or negative values.',
    after:    'Values are on a power-transformed scale determined by the estimated λ parameter.',
    ref:      { label: 'scipy.stats.boxcox', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html' },
  },
  winsorize: {
    what:     'Clips extreme values at specified percentile bounds (e.g., 1st/99th). Extreme values are replaced with the boundary value, not removed.',
    tradeoff: 'Reduces outlier influence while keeping all rows. Does not reshape the distribution — only truncates the tails. Column retains its original scale.',
    after:    'The distribution shape is mostly preserved but extreme values are capped. The column range becomes bounded by the chosen percentiles.',
    ref:      { label: 'scipy.stats.mstats.winsorize', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html' },
  },
  robustscaler: {
    what:     'Scales each value by subtracting the median and dividing by the interquartile range (IQR), rather than using mean and standard deviation.',
    tradeoff: 'More robust to outliers than standard scaling (StandardScaler) because it uses median and IQR instead of mean and std. Does not bound the output range — extreme outliers still produce extreme scaled values.',
    after:    'Values are centred around 0 (median becomes 0) with the IQR spanning roughly -0.5 to +0.5. Outliers remain but are less dominant.',
    ref:      { label: 'sklearn RobustScaler', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html' },
  },
  reflect_then_log: {
    what:     'Reflects a left-skewed distribution by subtracting each value from the maximum, then applies log1p to the reflected values.',
    tradeoff: 'Allows log transformation on left-skewed data where direct log would worsen skew. The reflection reverses the direction of the variable, which changes interpretation.',
    after:    'Values represent the log of the distance from the original maximum. The direction of the variable is reversed — higher original values become lower transformed values.',
    ref:      null,
  },
  pd_qcut: {
    what:     'Divides the column into equal-frequency (quantile) bins, assigning each row to a rank-based bucket rather than a value range.',
    tradeoff: 'Eliminates outlier influence entirely by converting to ranks. Loses the original numeric information — the model sees bins, not values. Bin labels are arbitrary.',
    after:    'Column becomes categorical with a fixed number of buckets. Apply an encoding step afterward.',
    ref:      { label: 'pandas.qcut', url: 'https://pandas.pydata.org/docs/reference/api/pandas.qcut.html' },
  },
  gmm_cluster_indicator: {
    what:     'Fits a Gaussian Mixture Model to the column and adds binary indicators for each detected component — essentially flagging which sub-population each row belongs to.',
    tradeoff: 'Useful for bimodal or multimodal columns where the mixture structure carries signal. Adds multiple columns to the dataset. The number of components requires tuning.',
    after:    'Adds N binary columns (one per component). The original column may be retained or dropped depending on context.',
    ref:      { label: 'sklearn GaussianMixture', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html' },
  },
  monitor: {
    what:     'Keep the column but flag it for active monitoring — track its distribution over time and set up drift alerts rather than transforming it now.',
    tradeoff: 'Preserves all information while acknowledging the issue. Requires infrastructure to monitor distributions in production. Does not address the underlying skew or variance issue.',
    after:    'No immediate change to the column. Set up distribution tracking using a monitoring tool and define alert thresholds.',
    ref:      null,
  },
  segment: {
    what:     'Analyse or model different segments of the data separately, rather than treating the full column as a single distribution.',
    tradeoff: 'Can reveal segment-specific patterns that a single model would average over. Requires enough data in each segment to model reliably. Increases complexity.',
    after:    'Dataset is split by segment for separate analysis or model training.',
    ref:      null,
  },
  use_tree_model: {
    what:     'Use a tree-based model (e.g. Random Forest, XGBoost, LightGBM) that is invariant to monotonic transformations of features.',
    tradeoff: 'Eliminates the need to transform the column entirely — tree splits are order-based, not magnitude-based. Trades model flexibility for skew robustness. Linear models, SVMs, and neural nets still benefit from transformation.',
    after:    'No column change required. The model handles skew natively through its split-finding mechanism.',
    ref:      { label: 'sklearn RandomForest', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html' },
  },

  // ── Encoding ────────────────────────────────────────────────────────────────

  one_hot: {
    what:     'Creates a separate binary (0/1) indicator column for each unique category value.',
    tradeoff: 'Simple and universally compatible. Best for low-cardinality columns (< ~15 categories). High-cardinality columns produce too many columns and can cause the curse of dimensionality.',
    after:    'Each original category becomes its own column. The original column is replaced by N binary columns where N is the number of unique values (minus one if drop_first is used).',
    ref:      { label: 'sklearn OneHotEncoder', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html' },
  },
  target_encoding: {
    what:     'Replaces each category with the mean of the target variable for rows in that category.',
    tradeoff: 'Very effective for high-cardinality columns. Must be applied within cross-validation folds to prevent target leakage — applying it to the full dataset before splitting inflates model performance.',
    after:    'Column becomes numeric. Similar categories by target rate will have similar encoded values. Rare categories may have noisy estimates.',
    ref:      { label: 'sklearn TargetEncoder', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html' },
  },
  frequency_encoding: {
    what:     'Replaces each category with the proportion of rows it appears in across the dataset.',
    tradeoff: 'No target variable needed. Captures relative frequency as a signal. Does not encode the relationship to the target — common and rare categories are distinguishable, but ordering by frequency may not reflect predictive importance.',
    after:    'Column becomes numeric (0–1 scale). Frequent categories get large values; rare ones get small values. Categories with equal frequency become indistinguishable.',
    ref:      null,
  },
  hash_encoding: {
    what:     'Maps category values to a fixed number of binary dimensions using a hash function.',
    tradeoff: 'Handles arbitrary cardinality at fixed memory cost and works on unseen categories at inference time. Some distinct categories may collide and share the same encoding.',
    after:    'Column expands to N binary columns (N is configurable). Some information loss from hash collisions is possible.',
    ref:      { label: 'category_encoders HashingEncoder', url: 'https://contrib.scikit-learn.org/category_encoders/hashing.html' },
  },
  drop_first: {
    what:     'Drops the first dummy variable when one-hot encoding to avoid perfect multicollinearity (the "dummy variable trap").',
    tradeoff: 'Required for linear models that cannot handle perfectly collinear features. For tree-based models, keeping all dummies is harmless and sometimes slightly better. The dropped category becomes the implicit reference group.',
    after:    'N−1 binary columns instead of N. The dropped category is encoded as all zeros.',
    ref:      { label: 'pandas.get_dummies drop_first', url: 'https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html' },
  },
  bucket_rare_values: {
    what:     'Groups infrequent category values below a count or proportion threshold into a single "other" bucket before encoding.',
    tradeoff: 'Reduces effective cardinality and prevents rare categories from producing unreliable target encoding estimates. Some information about rare categories is lost by merging them.',
    after:    'The column still needs a subsequent encoding step. Rare categories are now a single "other" group, which may mix unrelated values.',
    ref:      null,
  },
  group_rare: {
    what:     'Similar to bucket_rare_values — merges low-frequency categories into a single group to reduce cardinality.',
    tradeoff: 'Reduces noise from rare categories at the cost of losing their individual identity. The threshold for "rare" needs to be defined.',
    after:    'Apply encoding after grouping. The "rare" bucket becomes a single category.',
    ref:      null,
  },
  class_weight: {
    what:     'Pass class weights to the model so minority classes are penalised more heavily during training, compensating for imbalance.',
    tradeoff: 'Simple to implement (most sklearn estimators support class_weight="balanced"). Does not change the data — only changes the loss function. May not fully correct severe imbalance.',
    after:    'No change to the column. The model applies higher penalty for misclassifying the minority class.',
    ref:      { label: 'sklearn class_weight', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html' },
  },
  smote: {
    what:     'Synthetic Minority Over-sampling Technique. Generates synthetic samples for the minority class by interpolating between existing minority samples.',
    tradeoff: 'More effective than random oversampling at avoiding exact duplicates. Must be applied only to the training fold — never to validation or test data. Can introduce noisy samples near class boundaries.',
    after:    'Training set gains synthetic minority class rows. Class distribution becomes more balanced. Apply within cross-validation folds.',
    ref:      { label: 'imbalanced-learn SMOTE', url: 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html' },
  },
  stratified_split: {
    what:     'Use stratified train/test splitting to ensure class proportions are preserved in both the training and test sets.',
    tradeoff: 'Simple and always recommended for imbalanced targets. Does not change the data or the model — only the splitting strategy. Essential to avoid train/test distribution mismatch.',
    after:    'Both train and test sets reflect the original class distribution. Use stratify=y in sklearn train_test_split.',
    ref:      { label: 'sklearn train_test_split stratify', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html' },
  },

  // ── Missingness ──────────────────────────────────────────────────────────────

  drop: {
    what:     'Removes the column entirely from the dataset.',
    tradeoff: 'Simple and always valid. Appropriate when missingness is too high to impute reliably, or when the column carries no signal. Permanently loses any information the column contained.',
    after:    'The column is absent from the dataset. Any downstream steps that reference it by name will need updating.',
    ref:      null,
  },
  add_indicator: {
    what:     'Adds a binary column (0/1) indicating which rows had a missing value, alongside imputation of the original.',
    tradeoff: 'Preserves the information that a value was missing, which can itself be a predictive signal. Adds a column to the dataset. Must be combined with another imputation strategy for the original column.',
    after:    'A new binary column captures the missingness pattern. The original column is imputed separately. The model can learn from both the imputed value and whether the value was originally present.',
    ref:      { label: 'sklearn MissingIndicator', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html' },
  },
  mean_or_median: {
    what:     'Replaces missing values with either the mean (symmetric distributions) or median (skewed distributions) of the non-missing values.',
    tradeoff: 'Fast and simple. Mean is sensitive to outliers; median is more robust. Neither preserves variance — imputed rows all receive the same value.',
    after:    'Missing rows all receive the same central value. The distribution becomes slightly more concentrated around the centre.',
    ref:      { label: 'sklearn SimpleImputer', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html' },
  },
  median: {
    what:     'Replaces missing values with the median of the non-missing values.',
    tradeoff: 'More robust to outliers and skewed distributions than mean imputation. Does not preserve variance.',
    after:    'Missing rows all receive the same value (the median). Works well when the distribution is skewed or has outliers.',
    ref:      { label: 'sklearn SimpleImputer', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html' },
  },
  forward_fill: {
    what:     'Propagates the last known value forward to fill subsequent missing entries. Used for time-ordered or sequential data.',
    tradeoff: 'Appropriate when data has temporal structure and adjacent values are likely similar. Inappropriate for non-sequential data or when gaps represent genuinely missing information rather than a continuation.',
    after:    'Missing rows receive the value of the most recent non-missing row. Assumes temporal continuity within each series.',
    ref:      { label: 'pandas.DataFrame.ffill', url: 'https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html' },
  },
  model_based: {
    what:     'Trains a predictive model on the non-missing rows to predict the missing values from other columns.',
    tradeoff: 'Produces the most statistically plausible imputations by leveraging relationships between columns. Computationally expensive and requires careful cross-validation to avoid leakage.',
    after:    'Missing values are replaced with model-predicted estimates. Imputed values reflect patterns learned from the full observed dataset.',
    ref:      { label: 'sklearn IterativeImputer', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html' },
  },
  see_task: {
    what:     'The specific imputation strategy depends on the column type and context — refer to the task details for the recommended approach.',
    tradeoff: 'N/A — see the specific recommendation.',
    after:    'Depends on the approach selected.',
    ref:      null,
  },

  // ── Leakage / Unusable ───────────────────────────────────────────────────────

  investigate: {
    what:     'Manually examine the column origin, data pipeline, and semantics before taking action.',
    tradeoff: 'Required before dropping or retaining a flagged column — automated detection cannot determine intent. Takes time but prevents incorrect decisions.',
    after:    'Understanding the data provenance determines whether the column is safe to use, should be dropped, or needs re-engineering.',
    ref:      null,
  },
  replace_with_nan: {
    what:     'Replaces sentinel or placeholder values (e.g., -999, 0, "N/A") with actual null values so they are treated as missing rather than as a data point.',
    tradeoff: 'Essential when placeholder values are present — leaving them in treats them as real data. Increases the apparent missingness rate, which may require follow-up imputation.',
    after:    'Placeholder values are now null. Apply a missingness strategy afterward.',
    ref:      null,
  },
  bin: {
    what:     'Converts a continuous column into discrete buckets (e.g., deciles or domain-defined ranges).',
    tradeoff: 'Can extract signal from near-constant columns where crossing a threshold is meaningful, even if the raw value has near-zero variance. Loses granularity and introduces an arbitrary binning choice.',
    after:    'Column becomes categorical with a small number of levels. Apply an encoding step afterward.',
    ref:      { label: 'pandas.cut', url: 'https://pandas.pydata.org/docs/reference/api/pandas.cut.html' },
  },

}

// ── Lookup helper ──────────────────────────────────────────────────────────────

/**
 * Returns ACTION_META for an action object.
 * Tries act.method first (e.g. "log1p", "target_encoding"),
 * then falls back to act.action (e.g. "add_indicator", "drop").
 */
export function actionMeta(act) {
  if (act.method) {
    const byMethod = ACTION_META[normalizeMethod(act.method)]
    if (byMethod) return byMethod
  }
  return ACTION_META[normalizeMethod(act.action ?? '')] ?? null
}
