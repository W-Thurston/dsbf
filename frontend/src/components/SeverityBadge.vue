<template>
  <span :class="['badge', `badge-${severity ?? 'ok'}`]">{{ severity ?? 'no findings' }}</span>
</template>

<script setup lang="ts">
import type { SeverityName } from '../generated/ranks'

defineProps<{ severity?: SeverityName | null }>()
</script>
