"""A plan adds what selected checks require and orders everything by dependency."""

import textwrap

import pytest

from dsbf.engine import (
    Check,
    PlanError,
    Registry,
    Requires,
    Selection,
    check,
    make_plan,
    plugins,
)
from dsbf.engine.check import SemanticTypes
from dsbf.model import Cost, Dimension, Severity

pytestmark = pytest.mark.unit


def make_check(
    check_id: str,
    *,
    requires: tuple[str, ...] = (),
    after: tuple[str, ...] = (),
    cost: Cost = Cost.CHEAP,
    packs: tuple[str, ...] = ("core",),
) -> Check:
    @check(
        id=check_id,
        version="1.0.0",
        kind="rule",
        applies_to=SemanticTypes(),
        dimensions=[Dimension.VALIDITY],
        requires=Requires(checks=requires),
        after=after,
        cost=cost,
        packs=packs,
    )
    def declared(ctx, params):
        """A check for plan tests."""

    return declared


def plan_of(checks: list[Check], **selection):
    selection.setdefault("entry_points", False)
    return make_plan(Registry(checks), Selection(**selection))


class TestRequirements:
    def test_a_required_check_is_pulled_in_and_marked(self):
        plan = plan_of(
            [
                make_check("scorer", requires=("null_rate",)),
                make_check("null_rate", packs=("basics",)),
            ],
            packs="core",
        )

        assert plan.ids == ("null_rate", "scorer")
        assert plan.get("scorer").selected
        assert plan.get("scorer").required_by == ()
        assert not plan.get("null_rate").selected
        assert plan.get("null_rate").required_by == ("scorer",)
        assert plan.get("scorer").requires == ("null_rate",)

    def test_requirements_are_followed_through_the_chain_and_bypass_depth(self):
        plan = plan_of(
            [
                make_check("top", requires=("middle",)),
                make_check("middle", requires=("base",), cost=Cost.MODERATE),
                make_check("base", cost=Cost.HEAVY, packs=()),
            ],
            include="top",
            depth="quick",
        )

        assert plan.ids == ("base", "middle", "top")
        assert [e.selected for e in plan] == [False, False, True]
        assert plan.get("base").required_by == ("middle",)

    def test_a_check_both_selected_and_required_keeps_both_facts(self):
        plan = plan_of(
            [
                make_check("scorer", requires=("null_rate",)),
                make_check("summary", requires=("null_rate",)),
                make_check("null_rate"),
            ]
        )

        entry = plan.get("null_rate")
        assert entry.selected
        assert entry.required_by == ("scorer", "summary")

    def test_a_required_check_the_registry_lacks_is_a_plan_error(self):
        with pytest.raises(PlanError, match="'scorer' requires 'ghost', which is not"):
            plan_of([make_check("scorer", requires=("ghost",))])

    def test_a_required_check_the_user_excluded_is_a_plan_error_naming_both(self):
        registry = [
            make_check("scorer", requires=("null_rate",)),
            make_check("null_rate"),
        ]

        with pytest.raises(PlanError) as raised:
            plan_of(registry, exclude="null_rate")

        assert "'scorer' requires 'null_rate', which was excluded" in str(raised.value)

    def test_excluding_the_dependent_too_resolves_it(self):
        registry = [
            make_check("scorer", requires=("null_rate",)),
            make_check("null_rate"),
        ]

        assert plan_of(registry, exclude=["scorer", "null_rate"]).ids == ()

    def test_every_missing_requirement_is_reported(self):
        with pytest.raises(PlanError) as raised:
            plan_of(
                [
                    make_check("a", requires=("x",)),
                    make_check("b", requires=("y",)),
                ]
            )

        assert len(raised.value.problems) == 2


class TestSoftOrdering:
    def test_a_soft_dependency_orders_but_does_not_pull_in(self):
        checks = [
            make_check("scorer", after=("collinear",)),
            make_check("collinear", packs=("extra",)),
        ]

        alone = plan_of(checks, packs="core")
        both = plan_of(checks)

        assert alone.ids == ("scorer",)
        assert alone.get("scorer").after == ()
        assert both.ids == ("collinear", "scorer")
        assert both.get("scorer").after == ("collinear",)
        assert both.get("scorer").requires == ()

    def test_a_soft_dependency_on_an_unregistered_check_is_dropped(self):
        plan = plan_of([make_check("scorer", after=("ghost",))])

        assert plan.get("scorer").after == ()

    def test_a_soft_dependency_is_ordered_whatever_level_the_checks_sit_at(self):
        """data_quality_scorer read two checks it only reached by luck of level."""
        plan = plan_of(
            [
                make_check("scorer", after=("collinear", "out_of_bounds")),
                make_check("collinear", requires=("prep",)),
                make_check("out_of_bounds"),
                make_check("prep", requires=("base",)),
                make_check("base"),
            ]
        )

        levels = {e.id: e.level for e in plan}
        assert levels == {
            "base": 0,
            "out_of_bounds": 0,
            "prep": 1,
            "collinear": 2,
            "scorer": 3,
        }
        assert plan.get("scorer").requires == ()
        assert plan.get("scorer").after == ("collinear", "out_of_bounds")

    def test_a_soft_edge_stays_apart_from_hard_edges(self):
        plan = plan_of(
            [
                make_check("a"),
                make_check("b"),
                make_check("c", requires=("a",), after=("b",)),
            ]
        )

        entry = plan.get("c")
        assert (entry.requires, entry.after) == (("a",), ("b",))


class TestOrdering:
    def test_entries_run_by_level_then_id_whatever_the_registration_order(self):
        checks = [
            make_check("c", requires=("a",)),
            make_check("b"),
            make_check("a"),
            make_check("d", after=("c",)),
        ]

        forward = plan_of(checks)
        backward = plan_of(list(reversed(checks)))

        assert forward.ids == backward.ids == ("a", "b", "c", "d")
        assert [[e.id for e in level] for level in forward.levels] == [
            ["a", "b"],
            ["c"],
            ["d"],
        ]

    def test_an_entry_is_always_after_everything_it_depends_on(self):
        plan = plan_of(
            [
                make_check("e", requires=("d",), after=("a",)),
                make_check("d", requires=("c",)),
                make_check("c", after=("b",)),
                make_check("b", requires=("a",)),
                make_check("a"),
            ]
        )

        position = {check_id: i for i, check_id in enumerate(plan.ids)}
        for entry in plan:
            for earlier in (*entry.requires, *entry.after):
                assert position[earlier] < position[entry.id]
                assert plan.get(earlier).level < entry.level

    def test_independent_checks_share_a_level(self):
        plan = plan_of([make_check("a"), make_check("b"), make_check("c")])

        assert len(plan.levels) == 1
        assert len(plan.levels[0]) == 3


class TestCycles:
    def test_a_hard_cycle_is_a_plan_error_with_its_path(self):
        with pytest.raises(PlanError) as raised:
            plan_of(
                [
                    make_check("a", requires=("b",)),
                    make_check("b", requires=("c",)),
                    make_check("c", requires=("a",)),
                ]
            )

        assert (
            str(raised.value)
            == "dependency cycle: a requires b, b requires c, c requires a"
        )

    def test_a_cycle_through_a_soft_edge_is_still_a_cycle(self):
        with pytest.raises(PlanError) as raised:
            plan_of([make_check("a", requires=("b",)), make_check("b", after=("a",))])

        assert str(raised.value) == "dependency cycle: a requires b, b after a"

    def test_the_path_leaves_out_checks_that_only_wait_on_the_cycle(self):
        with pytest.raises(PlanError) as raised:
            plan_of(
                [
                    make_check("a", requires=("b",)),
                    make_check("b", requires=("a",)),
                    make_check("waits", requires=("a",)),
                ]
            )

        assert str(raised.value) == "dependency cycle: a requires b, b requires a"

    def test_a_cycle_among_checks_that_are_not_planned_is_not_an_error(self):
        checks = [
            make_check("a", requires=("b",), packs=("loop",)),
            make_check("b", requires=("a",), packs=("loop",)),
            make_check("fine"),
        ]

        assert plan_of(checks, packs="core").ids == ("fine",)


class TestEmptyPlan:
    def test_an_empty_selection_gives_an_empty_plan_with_a_warning(self):
        plan = plan_of([make_check("a")], packs="core", exclude="a")

        assert plan.ids == ()
        assert len(plan) == 0
        assert plan.levels == ()
        assert [(n.code, n.severity) for n in plan.notes] == [
            ("plan_empty", Severity.WARN)
        ]

    def test_a_plan_with_checks_has_no_notes(self):
        assert plan_of([make_check("a")]).notes == ()

    def test_planning_keeps_the_selection_that_produced_it(self):
        selection = Selection(packs="core", entry_points=False)

        plan = make_plan(Registry([make_check("a")]), selection)

        assert plan.selection is selection
        assert "a" in plan
        assert "b" not in plan


class TestPlugins:
    PLUGIN = textwrap.dedent(
        '''
        from dsbf.engine import SemanticTypes, check


        @check(
            id="{id}",
            version="1.0.0",
            kind="rule",
            applies_to=SemanticTypes(),
            dimensions=["validity"],
            packs=["mine"],
        )
        def {id}(ctx, params):
            """A plugin check."""
        '''
    )

    def test_plugin_directories_in_the_selection_are_loaded_before_selecting(
        self, tmp_path
    ):
        folder = tmp_path / "mine"
        folder.mkdir()
        (folder / "extra.py").write_text(self.PLUGIN.format(id="extra"))
        registry = Registry([make_check("null_rate")])

        plan = make_plan(
            registry, Selection(packs="mine", plugins=folder, entry_points=False)
        )
        again = make_plan(
            registry, Selection(packs="mine", plugins=folder, entry_points=False)
        )

        assert plan.ids == again.ids == ("extra",)
        assert registry.notes == ()

    def test_a_broken_plugin_leaves_a_note_and_the_plan_still_forms(self, tmp_path):
        folder = tmp_path / "broken"
        folder.mkdir()
        (folder / "bad.py").write_text("raise RuntimeError('boom')\n")
        registry = Registry([make_check("null_rate")])

        plan = make_plan(registry, Selection(plugins=folder, entry_points=False))

        assert plan.ids == ("null_rate",)
        assert [n.code for n in registry.notes] == ["plugin_load_failed"]

    def test_entry_points_load_only_when_the_selection_asks(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plugins, "entry_points", lambda group: calls.append(group) or ()
        )

        make_plan(Registry(), Selection(entry_points=False))
        assert calls == []

        make_plan(Registry(), Selection())
        assert calls == [plugins.CHECKS_GROUP, plugins.PACKS_GROUP]

    def test_a_plugin_check_that_requires_a_direct_one_is_resolved(self, tmp_path):
        folder = tmp_path / "mine"
        folder.mkdir()
        source = (
            self.PLUGIN.format(id="extra")
            .replace(
                'packs=["mine"],',
                'packs=["mine"], requires=Requires(checks=["null_rate"]),',
            )
            .replace("SemanticTypes, check", "Requires, SemanticTypes, check")
        )
        (folder / "extra.py").write_text(source)

        plan = make_plan(
            Registry([make_check("null_rate", packs=("core",))]),
            Selection(packs="mine", plugins=folder, entry_points=False),
        )

        assert plan.ids == ("null_rate", "extra")
        assert plan.get("null_rate").required_by == ("extra",)
