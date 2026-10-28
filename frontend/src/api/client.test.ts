import { afterEach, describe, expect, it, vi } from 'vitest'
import createClient from 'openapi-fetch'
import type { paths } from '../generated/api'
import { ApiError, unwrap } from './client'

function clientAnswering(body: unknown, status: number, contentType: string) {
  const fetch = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { 'content-type': contentType },
      }),
  )
  return { client: createClient<paths>({ baseUrl: 'http://dsbf.test', fetch }), fetch }
}

afterEach(() => vi.restoreAllMocks())

describe('the generated client', () => {
  it('returns a page of datasets from /v2/datasets', async () => {
    const page = { items: [{ key: 'titanic', name: 'titanic' }], total: 1, limit: 50, offset: 0 }
    const { client, fetch } = clientAnswering(page, 200, 'application/json')

    const result = await unwrap(client.GET('/v2/datasets'))

    expect(result.total).toBe(1)
    expect(result.items[0]?.name).toBe('titanic')
    const request = (fetch.mock.calls[0] as unknown as [Request])[0]
    expect(new URL(request.url).pathname).toBe('/v2/datasets')
  })

  it('encodes a path parameter', async () => {
    const { client, fetch } = clientAnswering({}, 404, 'application/problem+json')

    await client.GET('/v2/datasets/{key}/runs', { params: { path: { key: 'a/b#c' } } })

    const request = (fetch.mock.calls[0] as unknown as [Request])[0]
    expect(request.url).toContain('/v2/datasets/a%2Fb%23c/runs')
  })

  it('throws the service problem detail as an ApiError', async () => {
    const problem = {
      title: 'Not Found',
      status: 404,
      detail: 'No run r1',
      instance: '/v2/runs/r1',
    }
    const { client } = clientAnswering(problem, 404, 'application/problem+json')

    const failure = unwrap(client.GET('/v2/runs/{run_id}', { params: { path: { run_id: 'r1' } } }))

    await expect(failure).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
      message: 'No run r1',
    })
    await expect(failure).rejects.toBeInstanceOf(ApiError)
  })
})
