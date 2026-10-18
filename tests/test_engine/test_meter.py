"""Each result records wall time, process CPU time and a sampled memory peak."""

import resource
import time

import numpy as np
import pytest

from dsbf.engine.meter import Meter
from tests.test_engine.toys import declare, null_share, run_plan

pytestmark = pytest.mark.unit

MEGABYTE = 1024 * 1024


def busy(seconds):
    def spin(ctx, params):
        end = time.process_time() + seconds
        while time.process_time() < end:
            pass

    return spin


def test_every_result_carries_wall_cpu_and_memory():
    run = run_plan([null_share])

    timing = run.results["null_share"].timing
    assert timing.wall_seconds > 0
    assert timing.cpu_seconds is not None and timing.cpu_seconds >= 0
    assert timing.peak_memory_bytes and timing.peak_memory_bytes > 10 * MEGABYTE
    assert timing.started_at is not None


def test_a_check_that_burns_cpu_shows_it():
    run = run_plan([declare("spin", function=busy(0.2))])

    timing = run.results["spin"].timing
    assert timing.cpu_seconds is not None and 0.2 <= timing.cpu_seconds < 1.0
    assert timing.wall_seconds > 0.15  # the CPU and wall clocks differ slightly


def test_a_check_that_sleeps_uses_wall_time_but_little_cpu():
    run = run_plan([declare("nap", function=lambda ctx, params: time.sleep(0.2))])

    timing = run.results["nap"].timing
    assert timing.wall_seconds >= 0.2
    assert timing.cpu_seconds is not None and timing.cpu_seconds < 0.1


def test_the_memory_peak_sees_an_allocation_the_check_released():
    size = 128 * MEGABYTE

    def allocate(ctx, params):
        block = np.ones(size // 8)  # touched, so resident
        time.sleep(0.1)
        del block
        time.sleep(0.05)

    run = run_plan(
        [declare("allocate", function=allocate), declare("after", after=["allocate"])]
    )

    peak = run.results["allocate"].timing.peak_memory_bytes
    later = run.results["after"].timing.peak_memory_bytes
    assert peak and later
    assert peak - later > 0.75 * size
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak <= 1.01 * high_water  # the kernel's counter lags the sampled RSS


def test_memory_sampling_can_be_turned_off():
    run = run_plan([declare("spin", function=busy(0.05))], memory_sample_seconds=None)

    timing = run.results["spin"].timing
    assert timing.peak_memory_bytes is None
    assert timing.cpu_seconds is not None and timing.cpu_seconds > 0


def test_a_sample_interval_must_be_positive():
    with pytest.raises(ValueError, match="memory_sample_seconds"):
        Meter(0)


def test_overlapping_windows_each_see_the_peak_while_they_were_open():
    size = 96 * MEGABYTE
    with Meter(0.005) as meter:
        with meter.window() as outer:
            with meter.window() as inner:
                block = np.ones(size // 8)
                time.sleep(0.1)
                del block
            time.sleep(0.05)

    assert inner.peak_bytes and outer.peak_bytes
    assert outer.peak_bytes >= inner.peak_bytes > size
