<template>
  <span class="tip-wrap">
    <span class="tip-icon">ⓘ</span>
    <span class="tip-box" :class="[alignClass, directionClass]">{{ text }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  text:      { type: String, required: true },
  /** 'left' | 'center' | 'right' - horizontal anchor */
  align:     { type: String, default: 'center' },
  /** 'up' | 'down' - whether the box appears above or below the icon */
  direction: { type: String, default: 'up' },
})

const alignClass     = computed(() => `tip-${props.align}`)
const directionClass = computed(() => `tip-${props.direction}`)
</script>

<style scoped>
.tip-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
  margin-left: 4px;
  vertical-align: middle;
}

.tip-icon {
  font-size: 11px;
  color: #64748b;
  cursor: default;
  line-height: 1;
  transition: color 0.15s;
  user-select: none;
}
.tip-wrap:hover .tip-icon { color: #60a5fa; }

/* ── Base tooltip box ───────────────────────────────────────────────────── */
.tip-box {
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  position: absolute;
  width: 220px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
  font-weight: 400;
  color: #cbd5e1;
  line-height: 1.45;
  z-index: 9999;
  transition: opacity 0.15s, visibility 0.15s;
  white-space: normal;
  text-transform: none;
  letter-spacing: 0;
}

.tip-wrap:hover .tip-box { visibility: visible; opacity: 1; }

/* ── Direction: up (default) ────────────────────────────────────────────── */
.tip-up { bottom: calc(100% + 6px); }

.tip-up::after {
  content: '';
  position: absolute;
  top: 100%;
  border: 5px solid transparent;
  border-top-color: #334155;
}

/* ── Direction: down (for use inside overflow containers like table headers) */
.tip-down { top: calc(100% + 6px); }

.tip-down::after {
  content: '';
  position: absolute;
  bottom: 100%;
  border: 5px solid transparent;
  border-bottom-color: #334155;
}

/* ── Horizontal alignment ───────────────────────────────────────────────── */
.tip-center { left: 50%; transform: translateX(-50%); }
.tip-center.tip-up::after   { left: 50%; transform: translateX(-50%); }
.tip-center.tip-down::after { left: 50%; transform: translateX(-50%); }

.tip-left { left: 0; }
.tip-left.tip-up::after   { left: 8px; }
.tip-left.tip-down::after { left: 8px; }

.tip-right { right: 0; }
.tip-right.tip-up::after   { right: 8px; }
.tip-right.tip-down::after { right: 8px; }
</style>
