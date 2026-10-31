// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import type { components } from '../../generated/api'
import RerunProgress from './RerunProgress.vue'

type Job = components['schemas']['Job']
type JobEvent = components['schemas']['JobEvent']

const job = (state: Job['state']) => ({ state, error: null }) as unknown as Job
const event = { kind: 'check_started', check_id: 'outliers', position: 2, total: 9 } as JobEvent

describe('rerun progress', () => {
  it('says it is starting until the first event arrives', () => {
    const wrapper = mount(RerunProgress, { props: { job: job('running'), events: [] } })
    expect(wrapper.find('[data-role="starting"]').exists()).toBe(true)
  })

  it('names the check in progress once an event arrives', () => {
    const wrapper = mount(RerunProgress, { props: { job: job('running'), events: [event] } })
    expect(wrapper.find('[data-role="starting"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('outliers (2 of 9)')
  })

  it('does not say starting for a finished job', () => {
    const wrapper = mount(RerunProgress, { props: { job: job('failed'), events: [] } })
    expect(wrapper.find('[data-role="starting"]').exists()).toBe(false)
  })
})
