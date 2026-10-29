<template>
  <div class="settings-fields">
    <template v-for="field in fields" :key="field.name">
      <fieldset v-if="field.kind === 'group'" class="settings-group">
        <legend>{{ label(field.name) }}</legend>
        <p v-if="field.description" class="muted">{{ field.description }}</p>
        <SettingsFields
          :fields="field.fields"
          :model-value="(modelValue[field.name] as Overlay | undefined) ?? {}"
          @update:model-value="set(field.name, $event)"
        />
      </fieldset>
      <label v-else :title="field.description">
        {{ label(field.name) }}
        <select
          v-if="field.kind === 'enum' || field.kind === 'boolean'"
          :data-field="field.name"
          :value="String(modelValue[field.name] ?? '')"
          @change="set(field.name, choice(field, ($event.target as HTMLSelectElement).value))"
        >
          <option value="">default ({{ shown(field.default) }})</option>
          <option v-for="option in optionsOf(field)" :key="option" :value="option">
            {{ option }}
          </option>
        </select>
        <input
          v-else-if="field.kind === 'number' || field.kind === 'integer'"
          type="number"
          :step="field.kind === 'integer' ? 1 : 'any'"
          :data-field="field.name"
          :value="(modelValue[field.name] as number | undefined) ?? ''"
          :placeholder="shown(field.default)"
          @input="set(field.name, number(($event.target as HTMLInputElement).value))"
        />
        <input
          v-else-if="field.kind === 'list'"
          type="text"
          :data-field="field.name"
          :value="listText(modelValue[field.name])"
          :placeholder="shown(field.default)"
          @input="set(field.name, parseList(($event.target as HTMLInputElement).value))"
        />
        <textarea
          v-else-if="field.kind === 'json'"
          rows="3"
          spellcheck="false"
          :data-field="field.name"
          :value="jsonText(modelValue[field.name])"
          :placeholder="shown(field.default)"
          @input="setJson(field.name, ($event.target as HTMLTextAreaElement).value)"
        ></textarea>
        <input
          v-else
          type="text"
          :data-field="field.name"
          :value="(modelValue[field.name] as string | undefined) ?? ''"
          :placeholder="shown(field.default)"
          @input="set(field.name, ($event.target as HTMLInputElement).value)"
        />
        <span v-if="jsonError[field.name]" class="error">{{ jsonError[field.name] }}</span>
      </label>
    </template>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { listText, parseList, type Field, type Overlay } from '../../composables/schemaForm'

const props = defineProps<{ fields: Field[]; modelValue: Overlay }>()
const emit = defineEmits<{ 'update:modelValue': [Overlay] }>()

const jsonError = reactive<Record<string, string>>({})

const label = (name: string) => name.replaceAll('_', ' ')
const optionsOf = (field: Field) => (field.kind === 'boolean' ? ['true', 'false'] : field.options)

function shown(value: unknown): string {
  if (value === null || value === undefined) return 'unset'
  if (Array.isArray(value)) return value.join(', ') || 'none'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function choice(field: Field, text: string): unknown {
  if (text === '') return undefined
  return field.kind === 'boolean' ? text === 'true' : text
}

function number(text: string): number | undefined {
  return text === '' ? undefined : Number(text)
}

const jsonText = (value: unknown) => (value === undefined ? '' : JSON.stringify(value, null, 2))

function set(name: string, value: unknown) {
  const next = { ...props.modelValue }
  if (value === undefined) delete next[name]
  else next[name] = value
  emit('update:modelValue', next)
}

function setJson(name: string, text: string) {
  if (text.trim() === '') {
    jsonError[name] = ''
    set(name, undefined)
    return
  }
  try {
    const value: unknown = JSON.parse(text)
    jsonError[name] = ''
    set(name, value)
  } catch {
    jsonError[name] = 'not valid JSON'
  }
}
</script>
