"""The check an adapted task declares, and how it runs."""

import pytest

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY, TaskSpec
from dsbf.eda.task_result import TaskResult
from dsbf.engine import DiskCache, Registry, Selection, make_plan
from dsbf.legacy import (
    EXCLUDED_TASKS,
    LEGACY_PACK,
    LegacyParams,
    adapt_task,
    legacy_checks,
)
from dsbf.model import CheckStatus, Cost, Dimension
from tests.test_legacy.support import (
    make_config,
    make_session,
    output_of,
    params_of,
    run_adapted,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checks(tmp_path_factory):
    session = make_session(tmp_path_factory.mktemp("figures"))
    return {check.id: check for check in legacy_checks(session)}


def test_every_registered_task_becomes_a_check_but_the_timing_reports(checks):
    load_all_tasks()

    assert set(checks) == set(TASK_REGISTRY) - EXCLUDED_TASKS
    assert EXCLUDED_TASKS == {"identify_bottleneck_tasks", "log_resource_usage"}
    assert len(checks) > 60


def test_building_the_checks_scans_the_native_libraries_before_any_check_runs(
    monkeypatch, tmp_path
):
    # Scanning them while checks run in parallel can deadlock against a check that
    # imports a compiled module (see ``_scan_native_libraries``).
    from sklearn.utils import parallel

    monkeypatch.setattr(parallel, "_threadpool_controller", None)

    legacy_checks(make_session(tmp_path))

    assert getattr(parallel, "_threadpool_controller") is not None


def test_the_checks_form_a_valid_plan(tmp_path):
    session = make_session(tmp_path)

    plan = make_plan(
        Registry(legacy_checks(session)),
        Selection(depth="deep"),
        profile=session.profile,
    )

    assert {entry.id for entry in plan.entries} == set(TASK_REGISTRY) - EXCLUDED_TASKS


def test_an_adapted_check_runs_on_a_thread_in_the_legacy_pack(checks):
    for check in checks.values():
        assert check.spec.runs_in == "thread"
        assert check.spec.packs == (LEGACY_PACK,)
        assert check.spec.kind == "analysis"
        assert check.spec.output is not None


def test_depends_on_becomes_hard_requirements_and_undeclared_reads_soft(checks):
    for name in checks:
        spec = TASK_REGISTRY[name]
        assert checks[name].spec.requires.checks == tuple(spec.depends_on or ())
    scorer = checks["data_quality_scorer"].spec
    assert "summarize_numeric" in scorer.after
    assert not set(scorer.after) & set(scorer.requires.checks)


def test_expected_types_become_the_profile_types_they_map_to(checks):
    assert checks["detect_outliers"].spec.applies_to.names == ("numeric",)
    assert checks["summarize_nulls"].spec.applies_to.names == ()
    assert set(checks["compute_entropy"].spec.applies_to.names) <= {
        "categorical",
        "boolean",
        "numeric",
        "text",
    }
    expected = TASK_REGISTRY["extract_datetime_features"].expected_semantic_types
    assert expected == ["datetime"]
    assert checks["extract_datetime_features"].spec.applies_to.names == ("datetime",)


def test_cost_and_dimension_come_from_the_registration(checks):
    slow = [n for n, s in TASK_REGISTRY.items() if s.runtime_estimate == "slow"]
    assert slow
    assert all(checks[name].spec.cost is Cost.HEAVY for name in slow)
    assert checks["summarize_nulls"].spec.dimensions == (Dimension.COMPLETENESS,)
    assert checks["detect_data_leakage"].spec.dimensions == (Dimension.LEAKAGE,)


class TestParams:
    def test_they_carry_the_task_config_and_the_shared_config(self, checks):
        config = make_config()
        params = params_of(checks["detect_outliers"])

        assert params.task == (config["tasks"].get("detect_outliers") or {})
        assert params.shared["time_series"]["datetime_index_column"] == "when"
        assert "tasks" not in params.shared

    def test_only_a_task_that_writes_files_names_the_output_dir(self, checks, tmp_path):
        assert params_of(checks["generate_univariate_plots"]).output_dir
        assert params_of(checks["detect_constant_columns"]).output_dir is None

    def test_a_task_that_reads_the_reference_frame_names_its_digest(self, tmp_path):
        import pandas as pd

        base = make_session(tmp_path)
        session = type(base)(
            base.dataset,
            base.profile,
            config=base.config,
            output_dir=tmp_path,
            reference_data=pd.DataFrame({"age": [1.0, 2.0]}),
        )
        by_id = {check.id: check for check in legacy_checks(session)}

        assert params_of(by_id["compare_with_reference_dataset"]).reference
        assert params_of(by_id["detect_constant_columns"]).reference is None

    def test_the_source_digest_is_stable_and_each_task_has_its_own(self, checks):
        first = params_of(checks["detect_outliers"]).source
        again = params_of(checks["detect_outliers"]).source
        other = params_of(checks["detect_zeros"]).source

        assert first == again and first != other
        assert isinstance(params_of(checks["detect_outliers"]), LegacyParams)


# Tasks defined for the tests, since the built-in ones do not misbehave on cue.
class ReturnsFailure(BaseTask):
    def run(self) -> None:
        self.output = TaskResult(
            name=self.name,
            status="failed",
            summary={"message": "Could not fit the model."},
            error_metadata={
                "error_type": "LinAlgError",
                "trace_summary": "singular matrix",
                "suggested_action": "Drop collinear columns.",
            },
        )


class ReturnsSkip(BaseTask):
    def run(self) -> None:
        self.output = TaskResult(
            name=self.name, status="skipped", summary={"message": "Nothing to compare."}
        )


class Raises(BaseTask):
    def run(self) -> None:
        raise RuntimeError("boom")


class NeedsFigures(BaseTask):
    def run(self) -> None:
        self.get_output_path("x.png")
        self.output = TaskResult(name=self.name)


class DropsAColumn(BaseTask):
    """Edits its input in place, as legacy tasks may."""

    def run(self) -> None:
        self.input_data.drop(columns=["age"], inplace=True)
        self.output = TaskResult(
            name=self.name, summary={"n": self.input_data.shape[1]}
        )


class CountsColumns(BaseTask):
    def run(self) -> None:
        self.output = TaskResult(
            name=self.name, summary={"n": self.input_data.shape[1]}
        )


def adapted(cls, session, **overrides):
    return adapt_task(
        TaskSpec(name=cls.__name__.lower(), cls=cls, **overrides), session
    )


def run_one(session, *tasks):
    from dsbf.engine import execute

    plan = make_plan(Registry(tasks), Selection(depth="deep"), profile=session.profile)
    return execute(plan, session.dataset, session.profile, dsbf_version="test")


class TestStatuses:
    def test_a_task_that_returns_failed_ends_error_with_its_error_metadata(
        self, tmp_path
    ):
        session = make_session(tmp_path)

        run = run_one(session, adapted(ReturnsFailure, session))

        result = run.results["returnsfailure"]
        assert result.status is CheckStatus.ERROR
        (note,) = result.notes
        assert note.code == "task_failed"
        assert note.message == "Could not fit the model."
        assert (note.exception_type, note.detail, note.hint) == (
            "LinAlgError",
            "singular matrix",
            "Drop collinear columns.",
        )

    def test_a_task_that_returns_skipped_ends_skipped(self, tmp_path):
        session = make_session(tmp_path)

        run = run_one(session, adapted(ReturnsSkip, session))

        result = run.results["returnsskip"]
        assert result.status is CheckStatus.SKIPPED
        assert result.notes[0].message == "Nothing to compare."

    def test_a_task_that_raises_ends_error(self, tmp_path):
        session = make_session(tmp_path)

        run = run_one(session, adapted(Raises, session))

        result = run.results["raises"]
        assert result.status is CheckStatus.ERROR
        assert result.notes[0].code == "check_failed"
        assert "boom" in (result.notes[0].detail or "")

    def test_a_task_that_needs_figures_without_a_directory_ends_error(self, tmp_path):
        base = make_session(tmp_path)
        session = type(base)(base.dataset, base.profile, config=base.config)

        run = run_one(session, adapted(NeedsFigures, session))

        result = run.results["needsfigures"]
        assert result.status is CheckStatus.ERROR
        assert result.notes[0].code == "output_dir_missing"

    def test_a_dependent_of_a_failed_task_is_skipped(self, tmp_path):
        session = make_session(tmp_path)
        first = adapted(ReturnsFailure, session)
        second = adapted(CountsColumns, session, depends_on=["returnsfailure"])

        run = run_one(session, first, second)

        assert run.results["countscolumns"].status is CheckStatus.SKIPPED
        assert run.results["countscolumns"].notes[0].blocked_by == ("returnsfailure",)


def test_a_task_that_edits_its_input_does_not_change_what_another_sees(tmp_path):
    config = make_config()
    config["engine"]["backend"] = "pandas"
    session = make_session(tmp_path, config=config)
    width = session.dataset.n_columns

    run = run_one(
        session, adapted(DropsAColumn, session), adapted(CountsColumns, session)
    )

    assert output_of(run, "dropsacolumn")["result"]["summary"]["n"] == width - 1
    assert output_of(run, "countscolumns")["result"]["summary"]["n"] == width


class TestCache:
    def test_a_second_run_is_served_from_the_cache(self, tmp_path):
        session = make_session(tmp_path / "figures")
        cache = DiskCache(tmp_path / "cache")

        first = run_adapted(session, "detect_constant_columns", cache=cache)
        second = run_adapted(session, "detect_constant_columns", cache=cache)

        assert not first.results["detect_constant_columns"].timing.cached
        assert second.results["detect_constant_columns"].timing.cached
        assert (
            second.results["detect_constant_columns"].output
            == first.results["detect_constant_columns"].output
        )

    def test_changing_the_task_config_runs_it_again(self, tmp_path):
        cache = DiskCache(tmp_path / "cache")
        config = make_config()
        run_adapted(
            make_session(tmp_path / "f", config=config),
            "detect_constant_columns",
            cache=cache,
        )
        changed = make_config()
        changed.setdefault("tasks", {})["detect_constant_columns"] = {"tolerance": 0.5}

        run = run_adapted(
            make_session(tmp_path / "f", config=changed),
            "detect_constant_columns",
            cache=cache,
        )

        assert not run.results["detect_constant_columns"].timing.cached
