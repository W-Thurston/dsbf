import createClient from 'openapi-fetch'
import type { paths } from '../generated/api'

/** The service answered with an error; `problem` is its RFC 9457 body when it sent one. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// The dashboard is served by `dsbf dashboard`, so the API is on the same origin; Vite's
// dev server proxies /v2 there.
export const api = createClient<paths>({ baseUrl: import.meta.env.VITE_API_BASE ?? '' })

interface Settled<T> {
  data?: T
  error?: unknown
  response: Response
}

function problemMessage(error: unknown, response: Response): string {
  if (typeof error === 'object' && error !== null) {
    const { title, detail } = error as { title?: unknown; detail?: unknown }
    if (typeof detail === 'string') return detail
    if (typeof title === 'string') return title
  }
  return `${response.status} ${response.statusText}`.trim()
}

/** Return a call's data, or throw an `ApiError` carrying the service's own explanation. */
export async function unwrap<T>(call: Promise<Settled<T>>): Promise<T> {
  const { data, error, response } = await call
  if (data === undefined) throw new ApiError(response.status, problemMessage(error, response))
  return data
}
