import { onMounted, ref, watch, type Ref } from 'vue'

interface Page<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

/** A paged API list: the current page, its loading state and an error to show. */
export function usePage<T>(load: (limit: number, offset: number) => Promise<Page<T>>, size = 25) {
  const page: Ref<Page<T> | null> = ref(null)
  const offset = ref(0)
  const error = ref('')

  async function refresh() {
    error.value = ''
    try {
      page.value = await load(size, offset.value)
    } catch (caught) {
      error.value = caught instanceof Error ? caught.message : String(caught)
    }
  }

  onMounted(refresh)
  watch(offset, refresh)

  return { page, offset, error, size }
}
