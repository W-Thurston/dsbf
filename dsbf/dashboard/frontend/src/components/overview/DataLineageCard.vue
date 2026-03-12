<template>
  <div class="card lineage-card">
    <div class="card-title">
      Dataset Lineage
      <TooltipIcon text="Source file information - where the data came from and its on-disk properties at the time of this run." align="right" />
    </div>
    <div v-if="!run?.source_path && !isBuiltin" class="lineage-note">
      ⓘ Source path not recorded for this run. Re-run the profiler to capture full lineage.
    </div>
    <div class="lineage-grid">
      <div class="lineage-item" v-for="item in lineageItems" :key="item.label">
        <span class="lineage-label">
          {{ item.label }}
          <TooltipIcon :text="item.tooltip" align="center" />
        </span>
        <span class="lineage-value" :title="item.full ?? item.value">{{ item.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  run: { type: Object, default: null },
})

function formatBytes(bytes) {
  if (bytes == null) return '-'
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 ** 2)   return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3)   return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function formatModified(ts) {
  if (ts == null) return '-'
  return new Date(ts * 1000).toLocaleString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function fileName(path) {
  if (!path) return '-'
  return path.split('/').pop().split('\\').pop()
}

const isBuiltin = computed(() => {
  const src = props.run?.dataset_source
  return src === 'seaborn' || src === 'sklearn' || src === 'openml'
})

const lineageItems = computed(() => {
  if (!props.run) return []
  const sp    = props.run.source_path ?? null
  const fname = fileName(sp)
  const hasPath = !!sp

  if (isBuiltin.value) {
    // Built-in dataset — show source library info instead of file fields
    const src = props.run.dataset_source ?? 'built-in'
    const srcLabels = { seaborn: 'Seaborn', sklearn: 'Scikit-learn', openml: 'OpenML' }
    return [
      {
        label:   'Dataset',
        tooltip: 'The logical dataset name this run belongs to in DSBF.',
        value:   props.run.dataset_name ?? '—',
      },
      {
        label:   'File Name',
        tooltip: 'Not applicable — this dataset is loaded from a Python library, not a file on disk.',
        value:   'N/A (built-in)',
      },
      {
        label:   'File Path',
        tooltip: 'Not applicable — this dataset is loaded from a Python library, not a file on disk.',
        value:   'N/A (built-in)',
      },
      {
        label:   'File Size',
        tooltip: 'Not applicable — no file on disk.',
        value:   'N/A (built-in)',
      },
      {
        label:   'Last Modified',
        tooltip: 'Not applicable — no file on disk.',
        value:   'N/A (built-in)',
      },
    ]
  }

  return [
    {
      label:   'Dataset',
      tooltip: 'The logical dataset name this run belongs to in DSBF.',
      value:   props.run.dataset_name ?? '—',
    },
    {
      label:   'File Name',
      tooltip: 'The name of the source file that was profiled.',
      value:   hasPath ? (fname.length > 32 ? fname.slice(0, 32) + '…' : fname) : '-',
      full:    hasPath ? fname : null,
    },
    {
      label:   'File Path',
      tooltip: 'Full path to the source file on disk at the time this run was executed.',
      value:   sp ? (sp.length > 44 ? '…' + sp.slice(-44) : sp) : '-',
      full:    sp,
    },
    {
      label:   'File Size',
      tooltip: 'Size of the source file on disk. Compare with Memory in the strip above - a large ratio indicates the file format is verbose (e.g. CSV with long strings). Only available when the source path is recorded.',
      value:   props.run.source_file_size_bytes != null ? formatBytes(props.run.source_file_size_bytes) : '-',
    },
    {
      label:   'Last Modified',
      tooltip: 'When the source file was last written to on disk. Only available when the source path is recorded and the file still exists.',
      value:   props.run.source_last_modified != null ? formatModified(props.run.source_last_modified) : '-',
    },
  ]
})
</script>

<style scoped>
.lineage-card { height: 100%; }

.lineage-note {
  font-size: 12px;
  color: #475569;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 12px;
  line-height: 1.4;
}

.lineage-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.lineage-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #0f172a;
}
.lineage-item:last-child { border-bottom: none; padding-bottom: 0; }

.lineage-label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  display: flex;
  align-items: center;
  gap: 2px;
  white-space: nowrap;
  flex-shrink: 0;
}

.lineage-value {
  font-size: 13px;
  color: #e2e8f0;
  font-family: monospace;
  text-align: right;
  word-break: break-all;
  cursor: default;
}
</style>
