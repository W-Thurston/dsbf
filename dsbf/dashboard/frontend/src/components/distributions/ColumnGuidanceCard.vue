<template>
  <div class="card guidance-card">
    <div class="card-title">Guidance</div>

    <div v-if="!insights.length" class="no-guidance">
      No notable characteristics detected for this column.
    </div>

    <div v-else class="insights">
      <div
        v-for="(insight, i) in insights"
        :key="i"
        class="insight-item"
        :class="`insight-${insight.level}`"
      >
        <div class="insight-header">
          <span class="insight-icon">{{ insight.icon }}</span>
          <span class="insight-title">{{ insight.title }}</span>
        </div>
        <p class="insight-body">{{ insight.body }}</p>
        <div v-if="insight.actions?.length" class="insight-actions">
          <span
            v-for="action in insight.actions"
            :key="action"
            class="action-chip"
          >{{ action }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

const insights = computed(() => {
  const col    = props.column
  const intent = props.tasks.infer_types?.data?.[col]?.analysis_intent_dtype ?? 'unknown'
  const dtype  = props.tasks.infer_types?.data?.[col]?.inferred_dtype        ?? ''
  const nm     = props.tasks.summarize_numeric?.data?.[col]                  ?? {}
  const skew   = props.tasks.detect_skewness?.data?.[col]                    ?? null
  const nullPct = props.tasks.summarize_nulls?.data?.null_percentages?.[col] ?? null
  const unique  = props.tasks.summarize_unique?.data?.[col]                  ?? null
  const dom     = props.tasks.detect_single_dominant_value?.data?.[col]      ?? {}
  const rowCount = props.tasks.summarize_dataset_shape?.data?.num_rows       ?? null
  const isConstant = (props.tasks.detect_constant_columns?.data?.constant_columns ?? []).includes(col)
  const isId       = col in (props.tasks.detect_id_columns?.data ?? {})
  const hiCard     = props.tasks.detect_high_cardinality?.data?.[col]        ?? null
  const encoding   = props.tasks.suggest_categorical_encoding?.data?.[col]   ?? null
  const bounds     = props.tasks.detect_out_of_bounds?.data?.[col]           ?? null
  const zeroPct    = props.tasks.detect_zeros?.data?.zero_percentages?.[col]  ?? null

  const out = []

  // ── CONSTANT ────────────────────────────────────────────────────────────────
  if (isConstant) {
    out.push({
      level: 'error', icon: '🚫', title: 'Constant Column',
      body: 'This column has only one unique value across all rows. It contains no information and will cause issues in most ML models — particularly ones that normalise features.',
      actions: ['Drop before modeling'],
    })
    return out  // nothing else relevant
  }

  // ── NULL PERCENTAGE ──────────────────────────────────────────────────────────
  if (nullPct != null && nullPct >= 0.5) {
    out.push({
      level: 'error', icon: '⚠️', title: `High Missingness (${(nullPct * 100).toFixed(1)}%)`,
      body: 'More than half of values are missing. This column will likely hurt model performance unless the missingness itself is informative. Consider dropping or imputing with extreme care.',
      actions: ['Consider dropping', 'Add missingness indicator', 'Investigate source'],
    })
  } else if (nullPct != null && nullPct >= 0.2) {
    out.push({
      level: 'warn', icon: '⚠️', title: `Significant Missingness (${(nullPct * 100).toFixed(1)}%)`,
      body: 'Substantial missing data. Simple mean/mode imputation will introduce bias. Consider median imputation for skewed distributions, or model-based imputation if missing-not-at-random.',
      actions: ['Median imputation', 'Model-based imputation', 'Add missingness indicator'],
    })
  } else if (nullPct != null && nullPct >= 0.05) {
    out.push({
      level: 'info', icon: 'ℹ️', title: `Some Missingness (${(nullPct * 100).toFixed(1)}%)`,
      body: 'A small fraction of values are missing. For tree-based models this is often handled natively. For linear models, impute before fitting.',
      actions: ['Mean/median imputation', 'Forward fill if time series'],
    })
  }

  if (intent === 'continuous') {
    // ── LIKELY ID ──────────────────────────────────────────────────────────────
    if (isId) {
      out.push({
        level: 'warn', icon: '🔑', title: 'Likely ID Column',
        body: 'Unique value count equals or nearly equals row count, suggesting this is an identifier column. ID columns cause severe overfitting and provide no generalizable signal — a model that learns IDs cannot generalise to new data.',
        actions: ['Drop before modeling'],
      })
      return out
    }

    // ── SKEWNESS ────────────────────────────────────────────────────────────────
    if (skew != null && Math.abs(skew) > 2) {
      const dir = skew > 0 ? 'right (positive)' : 'left (negative)'
      const tailDir = skew > 0 ? 'high' : 'low'
      out.push({
        level: 'warn', icon: '📐', title: `Heavily Skewed (skew = ${skew.toFixed(2)})`,
        body: `Distribution is strongly ${dir}-skewed — the majority of values cluster at one end with a long tail toward ${tailDir} values. Effects: linear models (linear/logistic regression, SVM) will over-weight the tail; distance-based models (KNN) will be distorted by scale; tree-based models (Random Forest, XGBoost) are largely unaffected.`,
        actions: [
          skew > 0 ? 'log1p transform (if values ≥ 0)' : 'Reflect then log transform',
          'Box-Cox transform (flexible)',
          'Yeo-Johnson transform (handles negatives)',
          'Winsorize outliers',
        ],
      })
    } else if (skew != null && Math.abs(skew) > 1) {
      out.push({
        level: 'info', icon: '📐', title: `Moderate Skew (skew = ${skew.toFixed(2)})`,
        body: `Distribution is moderately skewed. Linear models may benefit from a transform; tree-based models are robust to this. Monitor residual patterns after fitting.`,
        actions: ['Consider log or sqrt transform', 'Check after model fitting'],
      })
    }

    // ── MEAN / MEDIAN DIVERGENCE ─────────────────────────────────────────────────
    const mean   = nm.mean   ?? null
    const median = nm['50%'] ?? null
    const std    = nm.std    ?? null
    if (mean != null && median != null && std != null && std > 0) {
      const divergence = Math.abs(mean - median) / std
      if (divergence > 0.5 && (skew == null || Math.abs(skew) <= 1)) {
        // Only show if skew didn't already catch it
        out.push({
          level: 'info', icon: '⚖️', title: `Mean/Median Divergence (${divergence.toFixed(2)}σ)`,
          body: `Mean (${mean.toFixed(3)}) and median (${median.toFixed(3)}) differ by ${divergence.toFixed(2)} standard deviations. This suggests the distribution is not symmetric — outliers or a skewed tail are pulling the mean away from the center of mass.`,
          actions: ['Check histogram for outlier clusters', 'Consider median over mean for summary'],
        })
      }
    }

    // ── NEAR ZERO VARIANCE ───────────────────────────────────────────────────────
    if (nm.near_zero_variance === true) {
      out.push({
        level: 'warn', icon: '📉', title: 'Near-Zero Variance',
        body: 'This column has very little variation. Low-variance features contribute minimal signal and can cause numerical instability in some algorithms. Regularisation-based models handle this poorly.',
        actions: ['Consider dropping', 'Apply variance threshold filter'],
      })
    }

    // ── OUT OF BOUNDS ─────────────────────────────────────────────────────────────
    if (bounds != null) {
      out.push({
        level: 'warn', icon: '🚧', title: 'Out-of-Bounds Values Detected',
        body: `Some values fall outside expected domain boundaries: ${JSON.stringify(bounds)}. These may represent data entry errors, unit mismatches, or genuine edge cases worth investigating.`,
        actions: ['Investigate source', 'Winsorize or cap values', 'Validate domain rules'],
      })
    }

    // ── EXCESS ZEROS ───────────────────────────────────────────────────────────
    if (zeroPct != null && zeroPct > 0.3) {
      out.push({
        level: 'info', icon: '0️⃣', title: `High Zero Rate (${(zeroPct * 100).toFixed(1)}%)`,
        body: 'A large proportion of values are zero. This may indicate a zero-inflated distribution. Consider whether zeros represent "none" (structural) or "unknown" (measurement gap), as the treatment differs.',
        actions: ['Zero-inflated model', 'Log1p transform', 'Add binary "is_zero" indicator'],
      })
    }
  }

  if (intent === 'categorical' || intent === 'text') {
    const cardRatio = (unique != null && rowCount != null) ? unique / rowCount : null

    // ── HIGH CARDINALITY / LIKELY ID ────────────────────────────────────────────
    if (isId || (cardRatio != null && cardRatio > 0.9)) {
      out.push({
        level: 'warn', icon: '🔑', title: 'Likely ID or Near-Unique Column',
        body: `${unique?.toLocaleString()} unique values across ${rowCount?.toLocaleString()} rows. This column is likely an identifier. One-hot encoding this would create ${unique?.toLocaleString()} sparse features — extremely high-dimensional and prone to overfitting.`,
        actions: ['Drop before modeling', 'If needed: hash encoding or target encoding'],
      })
    } else if (hiCard != null) {
      out.push({
        level: 'warn', icon: '🔢', title: `High Cardinality (${unique?.toLocaleString()} unique values)`,
        body: 'One-hot encoding this column will produce many sparse features. For tree-based models, target encoding or ordinal encoding is more efficient. For linear models, consider grouping rare categories.',
        actions: [
          encoding ? `Suggested: ${encoding}` : 'Target encoding',
          'Frequency encoding',
          'Group rare categories into "Other"',
        ],
      })
    } else if (unique != null && unique === 2) {
      out.push({
        level: 'info', icon: '✌️', title: 'Binary Category',
        body: 'This column has exactly two values. It behaves like a boolean and can be label-encoded (0/1) directly without one-hot encoding.',
        actions: ['Label encode (0/1)', 'Verify values are consistent'],
      })
    } else if (unique != null && unique < 10) {
      out.push({
        level: 'info', icon: '🏷️', title: `Low Cardinality (${unique} unique values)`,
        body: 'Small number of distinct values — safe for one-hot encoding without dimensionality concerns.',
        actions: encoding ? [`Suggested: ${encoding}`] : ['One-hot encoding'],
      })
    }

    // ── DOMINANT VALUE ──────────────────────────────────────────────────────────
    if (dom.mode_proportion != null && dom.mode_proportion >= 0.9) {
      out.push({
        level: 'warn', icon: '📊', title: `Near-Constant ("${dom.mode}" = ${(dom.mode_proportion * 100).toFixed(1)}%)`,
        body: 'One value dominates almost all rows. This provides minimal discriminative power and may cause class imbalance issues if used as a target variable.',
        actions: ['Consider dropping', 'Investigate if data quality issue'],
      })
    } else if (dom.mode_proportion != null && dom.mode_proportion >= 0.7) {
      out.push({
        level: 'info', icon: '📊', title: `Dominant Value ("${dom.mode}" = ${(dom.mode_proportion * 100).toFixed(1)}%)`,
        body: 'One value is significantly more common. If this is a target variable, your model will be biased toward this class — consider stratified sampling or class weight adjustments.',
        actions: ['Check if target variable', 'Consider stratified splitting', 'Class weight balancing'],
      })
    }
  }

  if (intent === 'boolean') {
    const vc    = props.tasks.summarize_value_counts?.data?.[col] ?? {}
    const vals  = Object.entries(vc).sort((a, b) => b[1] - a[1])
    const total = vals.reduce((s, [, n]) => s + n, 0)
    const topProp = total > 0 && vals[0] ? vals[0][1] / total : null
    if (topProp != null && topProp > 0.9) {
      out.push({
        level: 'warn', icon: '⚖️', title: `Severely Imbalanced (${(topProp * 100).toFixed(1)}% "${vals[0][0]}")`,
        body: 'One class dominates. If used as a target variable this will cause a biased model that predicts the majority class. If a feature, it carries little discriminative information.',
        actions: ['SMOTE / oversampling if target', 'Class weight parameter', 'Consider dropping if feature'],
      })
    } else if (topProp != null && topProp > 0.75) {
      out.push({
        level: 'info', icon: '⚖️', title: `Moderately Imbalanced (${(topProp * 100).toFixed(1)}% "${vals[0][0]}")`,
        body: 'Classes are not balanced. For classification targets, use stratified train/test splits and consider class weight adjustments.',
        actions: ['Stratified splitting', 'class_weight="balanced"'],
      })
    }
  }

  // If nothing triggered, provide a positive note
  if (out.length === 0) {
    out.push({
      level: 'good', icon: '✅', title: 'No Issues Detected',
      body: 'This column looks clean with no significant skewness, missing data, or cardinality concerns. It should be model-ready with standard preprocessing.',
      actions: [],
    })
  }

  // Depth note for continuous columns — bimodal detection requires full depth
  if (intent === 'continuous') {
    const depth = props.tasks.summarize_dataset_shape?.data?.profiling_depth ?? null
    const bimodalRan = 'detect_bimodal_distribution' in (props.tasks ?? {})
    if (!bimodalRan) {
      out.push({
        level: 'info', icon: '🔬', title: 'Bimodal Detection Not Run',
        body: 'Bimodal distribution detection requires full profiling depth. If you suspect this column may have two distinct clusters, re-run the profiler with --depth full.',
        actions: ['poetry run dsbf profile <file> --depth full'],
      })
    }
  }

  return out
})
</script>

<style scoped>
.guidance-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
}

.no-guidance {
  color: #475569;
  font-size: 13px;
  padding: 16px 0;
}

.insights {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.insight-item {
  border-radius: 6px;
  padding: 10px 12px;
  border-left: 3px solid;
}
.insight-error { background: #3d0f0f; border-color: #f87171; }
.insight-warn  { background: #3d2510; border-color: #fb923c; }
.insight-info  { background: #1e3a5f; border-color: #60a5fa; }
.insight-good  { background: #14291f; border-color: #4ade80; }

.insight-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.insight-icon  { font-size: 14px; flex-shrink: 0; }
.insight-title {
  font-size: 12px;
  font-weight: 600;
  color: #f1f5f9;
}
.insight-error .insight-title { color: #fca5a5; }
.insight-warn  .insight-title { color: #fed7aa; }
.insight-info  .insight-title { color: #bfdbfe; }
.insight-good  .insight-title { color: #bbf7d0; }

.insight-body {
  font-size: 12px;
  line-height: 1.5;
  color: #94a3b8;
  margin: 0 0 8px;
}

.insight-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.action-chip {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  background: #0f172a;
  color: #64748b;
  border: 1px solid #334155;
  white-space: nowrap;
}
</style>
