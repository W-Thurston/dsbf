<template>
  <nav v-if="total > limit" class="pager" aria-label="Pages">
    <button type="button" :disabled="offset === 0" @click="emit('update:offset', previous)">
      Previous
    </button>
    <span class="muted">{{ offset + 1 }}–{{ Math.min(offset + limit, total) }} of {{ total }}</span>
    <button
      type="button"
      :disabled="offset + limit >= total"
      @click="emit('update:offset', offset + limit)"
    >
      Next
    </button>
  </nav>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ offset: number; limit: number; total: number }>()
const emit = defineEmits<{ 'update:offset': [offset: number] }>()

const previous = computed(() => Math.max(0, props.offset - props.limit))
</script>

<style scoped>
.pager {
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: flex-end;
  margin-top: 12px;
}
</style>
