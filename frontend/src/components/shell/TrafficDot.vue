<template>
  <span
    :class="['dot', `dot-${mark}`]"
    :data-mark="mark"
    role="img"
    :aria-label="LABEL[mark]"
    :style="{ '--dot-size': `${size}px` }"
  />
</template>

<script setup lang="ts">
import type { Mark } from './health'

withDefaults(defineProps<{ mark: Mark; size?: number }>(), { size: 10 })

// Colour never carries the state alone: the label is read aloud and the text beside a
// dot says the same thing.
const LABEL: Record<Mark, string> = {
  green: 'all clear',
  blue: 'minor',
  amber: 'warning',
  red: 'serious',
  none: 'not assessed',
}
</script>

<style scoped>
.dot {
  display: inline-block;
  width: var(--dot-size);
  height: var(--dot-size);
  border-radius: 50%;
  flex: none;
}
.dot-green {
  background: var(--level-green);
}
.dot-blue {
  background: var(--level-blue);
}
.dot-amber {
  background: var(--level-amber);
}
.dot-red {
  background: var(--level-red);
}
/* An unassessed dimension is an empty ring, so it cannot pass for a filled green. */
.dot-none {
  background: transparent;
  box-shadow: inset 0 0 0 2px var(--level-none);
}
</style>
