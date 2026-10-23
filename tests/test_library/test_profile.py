"""``dsbf.profile()`` end to end: one call in, a complete ``Run`` out."""

import textwrap
import warnings

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dsbf
from dsbf.checks import core_pack
from dsbf.config import LimitsConfig
from dsbf.engine import CheckFinished, DiskCache, PlanError
from dsbf.model import CheckStatus, Run
from dsbf.utils.versioning import get_dsbf_version
from tests.test_legacy.support import build_frame

pytestmark = pytest.mark.integration

FILE_WRITERS = {"generate_dataset_summary_plots", "generate_univariate_plots"}


@pytest.fixture(scope="module", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return build_frame()


@pytest.fixture(scope="module")
def quick(frame) -> Run:
    return dsbf.profile(frame, name="demo", depth="quick")


def tree(root) -> set[str]:
    return {str(path) for path in root.rglob("*")}


class TestResult:
    def test_the_run_is_complete_and_names_the_data(self, quick, frame):
        assert isinstance(quick, Run)
        assert set(quick.results) == set(quick.plan.check_ids)
        assert quick.dataset.name == "demo"
        assert (quick.dataset.n_rows, quick.dataset.n_columns) == frame.shape
        assert quick.health.counts.total == len(quick.results)
        assert quick.health.counts.ok > 0

    def test_a_default_run_has_no_errors_and_no_file_writing_checks(self, quick):
        assert quick.health.counts.error == 0
        assert quick.health.counts.timeout == 0
        assert not FILE_WRITERS & set(quick.results)
        assert quick.notes == ()

    def test_the_version_comes_from_the_installed_package(self, quick):
        assert quick.plan.dsbf_version == get_dsbf_version() == dsbf.__version__
        assert quick.plan.dsbf_version != "dev"

    def test_the_health_text_has_the_run_header_and_a_counts_line(self, quick):
        lines = quick.render_health().splitlines()

        assert lines[0].startswith("DSBF run ")
        assert "(240 × 8, full scan)" in lines[0]
        assert lines[0].endswith(f"{len(quick.results)} checks")
        assert lines[1].startswith(f"  ok {quick.health.counts.ok}")

    def test_findings_come_with_the_run(self, quick):
        assert quick.findings
        assert all(f.check_id in quick.results for f in quick.findings)

    def test_the_run_survives_a_json_round_trip(self, quick):
        assert Run.model_validate_json(quick.model_dump_json()).run_id == quick.run_id


class TestNoWrites:
    def test_a_run_creates_no_file_anywhere_it_could(
        self, frame, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.setenv("DSBF_HOME", str(home))
        monkeypatch.chdir(work)

        dsbf.profile(frame, depth="quick")

        assert tree(work) == set()
        assert not home.exists()

    def test_an_opt_in_cache_writes_under_the_dsbf_home(
        self, frame, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))

        dsbf.profile(frame, depth="quick", cache=True)

        assert (tmp_path / "home" / "cache" / "results").is_dir()


@pytest.fixture(scope="module")
def sparse() -> pl.DataFrame:
    """A numeric column that is 60% zeros, beside one that has none."""
    values = [0.0] * 60 + [float(i) + 0.5 for i in range(40)]
    return pl.DataFrame({"sparse": values, "dense": [float(i) for i in range(100)]})


class TestMigratedChecks:
    """A core check replaces its legacy task, which keeps feeding the scorers."""

    def test_the_default_selection_runs_the_core_check_not_the_task(self, sparse):
        run = dsbf.profile(sparse, depth="quick")

        assert run.results["zeros"].status is CheckStatus.OK
        zero_findings = [f for f in run.findings if f.subject.columns == ("sparse",)]
        assert {f.check_id for f in zero_findings if "zero" in f.code} == {"zeros"}

    def test_a_scorer_still_gets_the_task_as_a_requirement_without_its_findings(
        self, sparse
    ):
        run = dsbf.profile(sparse, depth="quick")

        task = run.results["detect_zeros"]
        assert task.status is CheckStatus.OK
        assert task.output is not None
        assert task.findings == ()
        assert [n.code for n in task.notes] == ["findings_superseded"]
        assert run.results["ml_readiness_scorer"].status is CheckStatus.OK

    def test_without_the_core_pack_the_task_reports_its_findings(self, sparse):
        run = dsbf.profile(sparse, depth="quick", packs=["legacy"])

        assert "zeros" not in run.results
        assert run.results["detect_zeros"].findings
        assert not run.results["detect_zeros"].notes

    def test_naming_the_task_asks_for_it_as_it_was(self, sparse):
        run = dsbf.profile(sparse, depth="quick", checks=["+detect_zeros"])

        assert run.results["detect_zeros"].findings
        assert run.results["zeros"].findings

    def test_the_target_setting_drives_class_imbalance(self, frame):
        run = dsbf.profile(frame, depth="quick", target="flag")

        planned = {c.check_id: c for c in run.plan.checks}["class_imbalance"]
        assert planned.params["target"] == "flag"
        output = run.results["class_imbalance"].output
        assert output is not None
        assert output["target"] == "flag"

    def test_without_a_target_class_imbalance_is_not_applicable(self, frame):
        run = dsbf.profile(frame, depth="quick")

        assert run.results["class_imbalance"].status is CheckStatus.NOT_APPLICABLE
        assert run.results["schema"].status is CheckStatus.NOT_APPLICABLE

    def test_a_retired_task_runs_only_for_the_scorer_that_reads_it(self, frame):
        run = dsbf.profile(frame, depth="quick")

        for task in ("summarize_unique", "sample_head", "compute_entropy"):
            assert task not in run.results or run.results[task].findings == ()
        assert run.results["infer_types"].findings == ()

    def test_a_check_in_its_own_pack_can_be_run_alone(self, sparse):
        # `fuzzy_duplicates` is a heavy check, which only the deep depth runs.
        run = dsbf.profile(sparse, packs=["core"], depth="deep")

        assert set(run.results) == {check.id for check in core_pack().checks}
        assert not [r for r in run.results if r.startswith(("detect_", "summarize_"))]


class TestSelection:
    def test_a_minus_removes_a_check_nothing_else_needs(self, frame):
        run = dsbf.profile(frame, depth="quick", checks=["-detect_duplicate_columns"])

        assert "detect_duplicate_columns" not in run.results
        assert "detect_constant_columns" in run.results

    def test_removing_a_check_another_needs_is_a_plan_error(self, frame):
        with pytest.raises(PlanError, match="exclude 'ml_readiness_scorer' too"):
            dsbf.profile(frame, depth="quick", checks=["-detect_outliers"])

    def test_an_unknown_pack_is_a_plan_error(self, frame):
        with pytest.raises(PlanError, match="nope"):
            dsbf.profile(frame, packs=["nope"])

    def test_a_deeper_depth_plans_more(self, frame, quick):
        standard = dsbf.profile(
            frame, depth="standard", checks=["-fuzzy_duplicate_detection"]
        )

        assert set(quick.plan.check_ids) < set(standard.plan.check_ids)
        assert quick.plan.depth == "quick"
        assert standard.plan.depth == "standard"

    def test_task_settings_reach_the_check_and_its_recorded_parameters(self, frame):
        settings = {"detect_outliers": {"flag_threshold": 0.4}}

        run = dsbf.profile(
            frame,
            tasks=settings,
            depth="standard",
            checks=["-fuzzy_duplicate_detection"],
        )

        planned = {c.check_id: c for c in run.plan.checks}["detect_outliers"]
        assert planned.params["task"]["flag_threshold"] == 0.4
        assert settings == {"detect_outliers": {"flag_threshold": 0.4}}

    def test_target_and_time_index_reach_the_tasks_that_read_them(self, frame):
        run = dsbf.profile(
            frame,
            depth="standard",
            target="flag",
            time_index="when",
            checks=["-fuzzy_duplicate_detection"],
        )

        params = {c.check_id: c.params for c in run.plan.checks}
        assert params["compute_mutual_information"]["task"]["target_column"] == "flag"
        assert params["class_imbalance"]["target"] == "flag"
        assert params["compute_mutual_information"]["shared"]["time_series"] == {
            **params["compute_mutual_information"]["shared"]["time_series"],
            "enabled": True,
            "datetime_index_column": "when",
        }


class TestSampling:
    def test_a_sample_is_profiled_and_its_provenance_recorded(self, frame):
        run = dsbf.profile(
            frame, depth="quick", sample=dsbf.Sample(rows=60, strategy="random", seed=3)
        )

        assert run.dataset.n_rows == 240
        assert run.dataset.sampling is not None
        assert run.dataset.sampling.strategy == "random"
        assert (run.dataset.sampling.sampled_rows, run.dataset.sampling.seed) == (60, 3)
        assert run.profile.n_rows == 60
        assert "sample of 60 rows" in run.render_health()

    def test_a_number_is_the_first_rows(self, frame):
        run = dsbf.profile(frame, depth="quick", sample=50)

        assert run.dataset.sampling is not None
        assert run.dataset.sampling.strategy == "head"
        assert run.profile.n_rows == 50

    def test_the_same_seed_gives_the_same_run(self, frame):
        sample = dsbf.Sample(rows=60, strategy="random", seed=3)

        first = dsbf.profile(frame, depth="quick", sample=sample)
        again = dsbf.profile(frame, depth="quick", sample=sample)

        assert first.dataset == again.dataset
        assert first.profile == again.profile


class TestSources:
    def test_pandas_polars_and_arrow_give_the_same_data_identity(self, frame):
        runs = [
            dsbf.profile(source, depth="quick")
            for source in (frame, frame.to_pandas(), frame.to_arrow())
        ]

        assert len({run.dataset.fingerprint for run in runs}) == 1
        assert len({run.health.counts.ok for run in runs}) == 1

    def test_a_file_names_the_run_after_its_stem(self, frame, tmp_path):
        path = tmp_path / "events.csv"
        frame.write_csv(path)

        run = dsbf.profile(path, depth="quick")

        assert run.dataset.name == "events"
        assert run.dataset.source.kind == "file"
        assert not [p for p in tmp_path.iterdir() if p != path]

    def test_a_parquet_file_with_a_datetime_column_profiles(self, tmp_path):
        path = tmp_path / "stamped.parquet"
        pd.DataFrame(
            {
                "stamp": pd.date_range("2024-01-01", periods=40, freq="D"),
                "value": range(40),
            }
        ).to_parquet(path)

        run = dsbf.profile(path, depth="quick", time_index="stamp")

        assert run.health.counts.error == 0
        assert run.model_dump_json()  # serialises: no Timestamp left unconverted


class TestRunsToTheEnd:
    @pytest.fixture
    def plugin(self, tmp_path):
        folder = tmp_path / "plugin"
        folder.mkdir()
        (folder / "boom.py").write_text(
            textwrap.dedent(
                '''
                from dsbf.engine import SemanticTypes, check


                @check(
                    id="boom",
                    version="1.0.0",
                    kind="rule",
                    applies_to=SemanticTypes("numeric"),
                    dimensions=["validity"],
                )
                def boom(ctx, params):
                    """Always fails."""
                    raise RuntimeError("forced failure")
                '''
            )
        )
        (folder / "broken.py").write_text("raise ImportError('cannot load')\n")
        return folder

    def test_a_check_that_raises_still_gives_a_complete_run_with_a_note(
        self, frame, plugin
    ):
        run = dsbf.profile(frame, depth="quick", plugins=[plugin], checks=["+boom"])

        assert set(run.results) == set(run.plan.check_ids)
        assert run.results["boom"].status is CheckStatus.ERROR
        assert "forced failure" in run.results["boom"].notes[0].message
        assert run.results["detect_constant_columns"].status is CheckStatus.OK
        issues = {i.check_id: i.status for i in run.health.issues}
        assert issues["boom"] is CheckStatus.ERROR
        assert "boom" in run.render_health()

    def test_a_plugin_that_cannot_load_is_a_run_note(self, frame, plugin):
        run = dsbf.profile(frame, depth="quick", plugins=[plugin])

        assert "cannot load" in "".join(n.message + (n.detail or "") for n in run.notes)
        assert run.health.counts.error == 0

    def test_a_run_budget_that_is_used_up_skips_the_rest(self, frame):
        run = dsbf.profile(
            frame,
            depth="quick",
            limits=LimitsConfig(run_budget_seconds=1e-6),
            workers=1,
        )

        assert set(run.results) == set(run.plan.check_ids)
        assert run.health.counts.skipped + run.health.counts.timeout > 0

    def test_hooks_see_every_check_finish(self, frame):
        events = []

        run = dsbf.profile(frame, depth="quick", hooks=[events.append])

        finished = [e.check_id for e in events if isinstance(e, CheckFinished)]
        assert sorted(finished) == sorted(run.results)


class TestCache:
    def test_changing_one_task_setting_runs_only_that_check_and_its_readers(
        self, frame, tmp_path
    ):
        cache = DiskCache(tmp_path / "results")
        options = {"depth": "quick", "cache": cache}

        first = dsbf.profile(frame, **options)
        same = dsbf.profile(frame, **options)
        changed = dsbf.profile(
            frame, tasks={"detect_zeros": {"flag_threshold": 0.9}}, **options
        )

        assert not any(r.timing.cached for r in first.results.values())
        assert all(
            r.timing.cached for r in same.results.values() if r.status is CheckStatus.OK
        )
        reran = {
            check_id
            for check_id, result in changed.results.items()
            if result.status is CheckStatus.OK and not result.timing.cached
        }
        readers = {
            check_id
            for check_id, planned in {
                c.check_id: c for c in changed.plan.checks
            }.items()
            if "detect_zeros" in planned.depends_on
        }
        # A reader re-runs only if the changed result is different.
        assert "detect_zeros" in reran
        assert reran <= {"detect_zeros"} | readers
        assert len(reran) < changed.health.counts.ok / 2

    def test_a_cache_that_fails_is_a_run_note_not_an_error(self, frame):
        class Broken:
            def get(self, key):
                raise OSError("disk gone")

            def put(self, key, result):
                raise OSError("disk gone")

        run = dsbf.profile(frame, depth="quick", cache=Broken())

        assert [n.code for n in run.notes] == ["cache_unavailable"]
        assert run.health.counts.error == 0


class TestPackage:
    def test_profile_is_a_function_and_the_profiling_pass_is_a_module(self):
        import dsbf.profiling

        assert callable(dsbf.profile)
        assert dsbf.profiling.profile_dataset
        assert dsbf.Sample is dsbf.library.Sample

    def test_an_unknown_name_is_an_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'nothing'"):
            dsbf.nothing  # noqa: B018

    def test_arrow_tables_are_accepted(self, frame):
        table = pa.Table.from_pandas(frame.to_pandas())

        assert dsbf.profile(table, depth="quick").dataset.n_columns == 8
