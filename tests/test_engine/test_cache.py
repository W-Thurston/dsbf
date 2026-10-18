"""Results are kept between runs, and only what a change touches is run again."""

import dataclasses
import json
import os
import stat
import time

import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import (
    CacheKey,
    CheckFinished,
    CheckStarted,
    DiskCache,
    Limits,
    NoteRecorded,
    Registry,
    Requires,
    Selection,
    SemanticTypes,
    execute,
    make_plan,
)
from dsbf.engine.cache import CacheSession, result_digest
from dsbf.model import CheckResult, CheckStatus
from dsbf.paths import dsbf_home
from dsbf.profile import profile_dataset
from tests.test_engine.toys import (
    build_frame,
    comparable,
    declare,
    make_dataset,
    null_share,
    numeric_spread,
    quality_summary,
)

pytestmark = pytest.mark.unit


class MemoryCache:
    """A cache that is a dict, and remembers what it was asked."""

    def __init__(self):
        self.entries: dict[str, CheckResult] = {}
        self.asked: list[CacheKey] = []

    def get(self, key):
        self.asked.append(key)
        return self.entries.get(key.digest)

    def put(self, key, result):
        self.entries[key.digest] = result


class BrokenCache:
    def get(self, key):
        raise OSError("disk gone")

    def put(self, key, result):
        raise OSError("disk gone")


def counted(toy, calls):
    """``toy`` with a note of each time it is actually called."""

    def run(ctx, params):
        calls.append(toy.id)
        return toy.function(ctx, params)

    return dataclasses.replace(toy, function=run)


def with_version(toy, version):
    return dataclasses.replace(toy, spec=dataclasses.replace(toy.spec, version=version))


def go(checks, cache, dataset=None, params=None, **options):
    dataset = dataset or make_dataset()
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(checks),
        Selection(entry_points=False),
        params=params,
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="0.0.0", cache=cache, **options)


ALL = ["null_share", "numeric_spread", "quality_summary"]


def toys(calls):
    return [counted(t, calls) for t in (null_share, numeric_spread, quality_summary)]


class TestReuse:
    def test_an_unchanged_run_calls_no_check_and_gives_the_same_run(self):
        cache, calls = MemoryCache(), []
        first = go(toys(calls), cache)
        calls.clear()

        second = go(toys(calls), cache)

        assert calls == []
        assert comparable(second) == comparable(first)
        assert [r.timing.cached for r in second.results.values()] == [True] * 3
        assert [r.timing.cached for r in first.results.values()] == [False] * 3

    def test_without_a_cache_every_run_calls_every_check(self):
        calls = []
        go(toys(calls), None)
        go(toys(calls), None)

        assert sorted(calls) == sorted(ALL * 2)

    def test_a_cached_run_is_the_same_for_any_worker_count(self):
        cache = MemoryCache()
        fresh = go(toys([]), cache, workers=1)

        assert comparable(go(toys([]), cache, workers=1)) == comparable(fresh)
        assert comparable(go(toys([]), cache, workers=4)) == comparable(fresh)

    def test_a_hit_is_timed_as_a_lookup(self):
        cache = MemoryCache()
        go(toys([]), cache)

        timing = go(toys([]), cache).results["numeric_spread"].timing

        assert timing.cached is True
        assert timing.cpu_seconds is None and timing.peak_memory_bytes is None
        assert timing.shared_with == 0
        assert timing.started_at is not None


class TestInvalidation:
    @pytest.fixture
    def warm(self):
        cache, calls = MemoryCache(), []
        go(toys(calls), cache)
        calls.clear()
        return cache, calls

    def test_changing_one_parameter_runs_that_check_and_its_dependents(self, warm):
        cache, calls = warm

        go(toys(calls), cache, params={"numeric_spread": {"min_std": 5.0}})

        assert sorted(calls) == ["numeric_spread", "quality_summary"]

    def test_the_unchanged_checks_are_served_from_the_cache(self, warm):
        cache, calls = warm

        run = go(toys(calls), cache, params={"numeric_spread": {"min_std": 5.0}})

        cached = {i: r.timing.cached for i, r in run.results.items()}
        assert cached == {
            "null_share": True,
            "numeric_spread": False,
            "quality_summary": False,
        }

    def test_a_new_version_runs_that_check_and_its_dependents(self, warm):
        cache, calls = warm
        bumped = [
            counted(with_version(null_share, "1.0.1"), calls),
            counted(numeric_spread, calls),
            counted(quality_summary, calls),
        ]

        run = go(bumped, cache)

        assert sorted(calls) == ["null_share", "quality_summary"]
        assert run.results["null_share"].check_version == "1.0.1"

    def test_a_dependent_whose_upstream_reran_to_the_same_result_is_served(self, warm):
        cache, calls = warm
        (spread,) = (
            digest
            for digest, result in cache.entries.items()
            if result.check_id == "numeric_spread"
        )
        del cache.entries[spread]

        go(toys(calls), cache)

        assert calls == ["numeric_spread"]

    def test_other_data_runs_everything(self, warm):
        cache, calls = warm
        frame = build_frame().with_columns(pl.col("amount") * 2)

        go(toys(calls), cache, dataset=make_dataset(frame))

        assert sorted(calls) == ALL

    def test_a_sample_of_the_same_data_runs_everything(self, warm):
        cache, calls = warm

        go(toys(calls), cache, dataset=make_dataset(rows=50, strategy="head"))

        assert sorted(calls) == ALL

    def test_another_seed_is_another_sample(self):
        cache, calls = MemoryCache(), []
        go(toys(calls), cache, dataset=make_dataset(rows=50, strategy="random", seed=1))
        calls.clear()

        go(toys(calls), cache, dataset=make_dataset(rows=50, strategy="random", seed=1))
        assert calls == []
        go(toys(calls), cache, dataset=make_dataset(rows=50, strategy="random", seed=2))
        assert sorted(calls) == ALL

    def test_a_soft_upstream_that_failed_last_time_is_not_the_same_input(self):
        def explode(ctx, params):
            raise RuntimeError("boom")

        cache, calls = MemoryCache(), []
        broken = dataclasses.replace(numeric_spread, function=explode)
        first = go([null_share, broken, counted(quality_summary, calls)], cache)
        assert first.results["numeric_spread"].status is CheckStatus.ERROR
        calls.clear()

        go(toys(calls), cache)

        assert sorted(calls) == ["numeric_spread", "quality_summary"]


class TestKeys:
    def asked(self, **changes):
        cache = MemoryCache()
        go([null_share, numeric_spread], cache, **changes)
        return {key.check_id: key for key in cache.asked}

    def test_equal_inputs_give_equal_keys(self):
        assert self.asked() == self.asked()

    def test_the_key_names_what_it_covers(self):
        components = self.asked()["numeric_spread"].components

        assert components["check"] == "numeric_spread"
        assert components["version"] == "1.2.0"
        assert components["params"] == {"min_std": 0.5}
        assert components["fingerprint"] == make_dataset().fingerprint
        assert components["sampling"] is None
        assert components["upstream"] == []

    def test_a_dependent_key_holds_the_digest_of_what_it_reads(self):
        cache = MemoryCache()
        run = go(
            [null_share, numeric_spread, quality_summary],
            cache,
        )

        keys = {key.check_id: key for key in cache.asked}

        assert keys["quality_summary"].components["upstream"] == [
            ["null_share", result_digest(run.results["null_share"])],
            ["numeric_spread", result_digest(run.results["numeric_spread"])],
        ]

    def test_the_profile_is_part_of_the_key(self):
        dataset = make_dataset()
        filled = build_frame().with_columns(pl.col("amount").fill_null(0.0))
        other = make_dataset(filled)

        one = self.session(dataset, profile_dataset(dataset))
        two = self.session(dataset, profile_dataset(other))

        assert one._scope["profile"] != two._scope["profile"]

    def test_profiling_the_same_data_twice_gives_the_same_digest(self):
        one = self.session(make_dataset(), profile_dataset(make_dataset()))
        two = self.session(make_dataset(), profile_dataset(make_dataset()))

        assert one._scope == two._scope

    @staticmethod
    def session(dataset, profile):
        return CacheSession(MemoryCache(), dataset, profile)

    def test_the_digest_of_a_result_ignores_its_timing(self):
        run = go([null_share], None)
        result = run.results["null_share"]
        later = result.model_copy(
            update={"timing": result.timing.model_copy(update={"wall_seconds": 99.0})}
        )

        assert result_digest(later) == result_digest(result)


class TestWhatIsStored:
    def test_only_ok_results_are_stored(self):
        def explode(ctx, params):
            raise RuntimeError("boom")

        def sleep(ctx, params):
            time.sleep(5)

        cache = MemoryCache()
        checks = [
            declare("fine"),
            declare("broken", function=explode),
            declare("slow", function=sleep),
            declare("blocked", requires=Requires(checks=["broken"])),
        ]

        run = go(checks, cache, limits=Limits(check_timeout_seconds=0.2))

        assert {r.status for r in run.results.values()} == {
            CheckStatus.OK,
            CheckStatus.ERROR,
            CheckStatus.TIMEOUT,
            CheckStatus.SKIPPED,
        }
        assert [r.check_id for r in cache.entries.values()] == ["fine"]

    def test_a_check_that_ended_badly_runs_again_next_time(self):
        calls = []

        def flaky(ctx, params):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("first time")

        cache = MemoryCache()
        checks = [declare("flaky", function=flaky)]

        assert go(checks, cache).results["flaky"].status is CheckStatus.ERROR
        assert go(checks, cache).results["flaky"].status is CheckStatus.OK
        assert go(checks, cache).results["flaky"].timing.cached is True
        assert len(calls) == 2

    def test_a_not_applicable_result_is_not_stored(self):
        cache = MemoryCache()
        only_dates = declare("dates_only", applies_to=SemanticTypes("datetime"))
        data = load(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))

        run = go([only_dates], cache, dataset=data)

        assert run.results["dates_only"].status is CheckStatus.NOT_APPLICABLE
        assert cache.entries == {}


class TestEvents:
    def test_a_hit_has_no_start_but_its_notes_and_its_finish(self):
        def noted(ctx, params):
            ctx.note(code="heads_up", message="Look at this.")

        cache = MemoryCache()
        checks = [declare("noted", function=noted)]
        go(checks, cache)
        events = []

        go(checks, cache, hooks=[events.append])

        assert [type(e) for e in events] == [NoteRecorded, CheckFinished]
        assert events[0].note.code == "heads_up"
        assert events[1].result.timing.cached is True
        assert not any(isinstance(e, CheckStarted) for e in events)

    def test_a_hit_is_served_after_the_run_budget_is_used_up(self):
        cache = MemoryCache()
        checks = [declare("quick")]
        go(checks, cache)

        run = go(checks, cache, limits=Limits(run_budget_seconds=1e-9))

        assert run.results["quick"].status is CheckStatus.OK
        assert run.results["quick"].timing.cached is True


class TestFailingCache:
    def test_a_cache_that_raises_does_not_stop_the_run(self, caplog):
        calls = []

        with caplog.at_level("WARNING", logger="dsbf.engine.cache"):
            run = go(toys(calls), BrokenCache())

        assert {r.status for r in run.results.values()} == {CheckStatus.OK}
        assert comparable(run) == comparable(go(toys([]), None))
        warnings = [r for r in caplog.records if r.name == "dsbf.engine.cache"]
        assert len(warnings) == 1
        assert "disk gone" in warnings[0].getMessage()


class TestDiskCache:
    def key(self, digest="a" * 64, check_id="null_share"):
        return CacheKey(check_id, digest, {"check": check_id})

    def result(self):
        return go([null_share], None).results["null_share"]

    def test_a_result_comes_back_equal(self, tmp_path):
        cache = DiskCache(tmp_path)
        result = self.result()

        cache.put(self.key(), result)

        assert cache.get(self.key()) == result

    def test_a_result_with_output_and_findings_comes_back_equal(self, tmp_path):
        cache = DiskCache(tmp_path)
        run = go([null_share, numeric_spread, quality_summary], None)

        for check_id, result in run.results.items():
            cache.put(self.key(check_id=check_id), result)

        for check_id, result in run.results.items():
            assert cache.get(self.key(check_id=check_id)) == result
        assert run.results["numeric_spread"].output
        assert run.results["null_share"].findings

    def test_an_unknown_key_is_a_miss(self, tmp_path):
        assert DiskCache(tmp_path).get(self.key()) is None

    def test_one_file_per_entry_under_the_check(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.put(self.key(), self.result())

        found = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")]
        assert found == [f"null_share/{'a' * 64}.json"]
        assert list(tmp_path.rglob("*.tmp")) == []

    @pytest.mark.parametrize(
        "damage",
        [
            lambda text: text[: len(text) // 2],
            lambda text: "",
            lambda text: json.dumps({"format": 99}),
            lambda text: text.replace('"format": 1', '"format": 2'),
            lambda text: text.replace("a" * 64 + '",', "b" * 64 + '",'),
            lambda text: json.dumps({**json.loads(text), "result": {"check_id": "x"}}),
        ],
        ids=["truncated", "empty", "other-format", "format-2", "other-key", "invalid"],
    )
    def test_an_entry_that_cannot_be_used_is_a_miss(self, tmp_path, damage):
        cache = DiskCache(tmp_path)
        cache.put(self.key(), self.result())
        path = cache.path(self.key())
        path.write_text(damage(path.read_text()))

        assert cache.get(self.key()) is None

    def test_a_damaged_entry_is_replaced_by_the_next_run(self, tmp_path):
        cache = DiskCache(tmp_path)
        calls = []
        checks = toys(calls)
        go(checks, cache)
        for path in tmp_path.rglob("*.json"):
            path.write_text("{")
        calls.clear()

        go(checks, cache)
        assert len(calls) == 3
        calls.clear()
        go(checks, cache)

        assert calls == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
    def test_a_root_that_cannot_be_written_is_a_miss_and_does_not_fail_the_run(
        self, tmp_path
    ):
        tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            run = go(toys([]), DiskCache(tmp_path / "cache"))
        finally:
            tmp_path.chmod(stat.S_IRWXU)

        assert {r.status for r in run.results.values()} == {CheckStatus.OK}

    def test_the_default_follows_the_dsbf_home(self, monkeypatch, tmp_path):
        assert DiskCache.default().root == dsbf_home() / "cache" / "results"

        monkeypatch.setenv("DSBF_HOME", str(tmp_path))

        assert DiskCache.default().root == tmp_path.resolve() / "cache" / "results"

    def test_nothing_is_created_until_a_result_is_stored(self, tmp_path):
        root = tmp_path / "not" / "yet"
        cache = DiskCache(root)

        assert cache.get(self.key()) is None
        assert not root.exists()

    def test_the_whole_engine_run_is_reproduced_from_disk(self, tmp_path):
        cache = DiskCache(tmp_path)
        calls = []
        first = go(toys(calls), cache)
        calls.clear()

        second = go(toys(calls), DiskCache(tmp_path))

        assert calls == []
        assert comparable(second) == comparable(first)
