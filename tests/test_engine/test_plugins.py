"""Plugins: local directories and installed entry points."""

import json
import subprocess
import sys
import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from dsbf.engine import Registry, plugins

pytestmark = pytest.mark.unit

CHECK_SOURCE = '''
from dsbf.engine import SemanticTypes, check


@check(
    id="{id}",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=["validity"],
    packs={packs!r},
)
def {id}(ctx, params):
    """A plugin check."""
'''


def check_source(check_id: str, packs: tuple[str, ...] = ()) -> str:
    return CHECK_SOURCE.format(id=check_id, packs=packs)


def write(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(textwrap.dedent(source))
    return path


def codes(registry: Registry) -> list[str]:
    return [note.code for note in registry.notes]


class TestLocalDirectories:
    def test_checks_in_a_directory_are_registered(self, tmp_path):
        write(tmp_path / "mine", "one.py", check_source("one", ("mine",)))
        write(tmp_path / "mine", "two.py", check_source("two"))
        registry = Registry()

        registry.load_directory(tmp_path / "mine")

        assert [c.id for c in registry] == ["one", "two"]
        assert registry.packs == {"mine": ("one",)}
        assert registry.source_of("one") == f"directory {(tmp_path / 'mine').resolve()}"
        assert registry.notes == ()

    def test_two_directories_each_with_a_utils_module_coexist(self, tmp_path):
        for name, value in (("alpha", 1), ("beta", 2)):
            folder = tmp_path / name
            write(folder, "utils.py", f"VALUE = {value}\n")
            write(
                folder,
                f"{name}_check.py",
                check_source(f"{name}_check")
                + "\nfrom .utils import VALUE\nSEEN = VALUE\n",
            )
        registry = Registry()

        registry.load_directory(tmp_path / "alpha")
        registry.load_directory(tmp_path / "beta")

        assert [c.id for c in registry] == ["alpha_check", "beta_check"]
        modules = {
            name: sys.modules[check.function.__module__]
            for name, check in (
                ("alpha", registry.get("alpha_check")),
                ("beta", registry.get("beta_check")),
            )
        }
        assert modules["alpha"].SEEN == 1
        assert modules["beta"].SEEN == 2
        assert "utils" not in sys.modules
        assert registry.notes == ()

    def test_files_starting_with_an_underscore_are_skipped(self, tmp_path):
        write(tmp_path, "_helper.py", "raise RuntimeError('must not run')\n")
        write(tmp_path, "visible.py", check_source("visible"))
        registry = Registry()

        registry.load_directory(tmp_path)

        assert [c.id for c in registry] == ["visible"]
        assert registry.notes == ()

    def test_a_package_init_runs_as_the_directorys_init(self, tmp_path):
        write(tmp_path, "__init__.py", "SHARED = 'from init'\n")
        write(
            tmp_path,
            "uses_init.py",
            check_source("uses_init") + "\nfrom . import SHARED\nGOT = SHARED\n",
        )
        registry = Registry()

        registry.load_directory(tmp_path)

        module = sys.modules[registry.get("uses_init").function.__module__]
        assert module.GOT == "from init"

    def test_loading_a_directory_twice_runs_its_code_once(self, tmp_path):
        source = (
            check_source("counted") + "\nfrom pathlib import Path\n"
            "Path(__file__).with_name('runs.log').open('a').write('x')\n"
        )
        write(tmp_path, "counted.py", source)
        registry = Registry()

        registry.load_directory(tmp_path)
        registry.load_directory(tmp_path)
        registry.load_directory(str(tmp_path) + "/")

        assert len(registry) == 1
        assert (tmp_path / "runs.log").read_text() == "x"
        assert registry.notes == ()

    def test_a_second_registry_reuses_the_imported_module(self, tmp_path):
        source = (
            check_source("counted") + "\nfrom pathlib import Path\n"
            "Path(__file__).with_name('runs.log').open('a').write('x')\n"
        )
        write(tmp_path, "counted.py", source)
        first, second = Registry(), Registry()

        first.load_directory(tmp_path)
        second.load_directory(tmp_path)

        assert second.get("counted") is first.get("counted")
        assert (tmp_path / "runs.log").read_text() == "x"
        assert second.notes == ()

    def test_a_missing_path_and_a_file_each_leave_a_note(self, tmp_path):
        a_file = write(tmp_path, "not_a_dir.py", "")
        registry = Registry()

        registry.load_directory(tmp_path / "missing")
        registry.load_directory(a_file)

        assert codes(registry) == ["plugin_invalid", "plugin_invalid"]
        assert "does not exist" in registry.notes[0].message
        assert "is not a directory" in registry.notes[1].message
        assert all(note.hint for note in registry.notes)

    def test_a_file_that_fails_to_import_leaves_a_note_and_the_rest_load(
        self, tmp_path
    ):
        write(tmp_path, "a_broken.py", "raise ZeroDivisionError('boom')\n")
        write(tmp_path, "b_fine.py", check_source("fine"))
        registry = Registry()

        registry.load_directory(tmp_path)

        assert [c.id for c in registry] == ["fine"]
        (note,) = registry.notes
        assert note.code == "plugin_load_failed"
        assert note.exception_type == "ZeroDivisionError"
        assert "a_broken.py" in note.message
        assert "boom" in note.message
        assert note.detail is not None and "Traceback" in note.detail
        assert not [m for m in sys.modules if m.endswith(".a_broken")]

    def test_a_directory_that_declares_no_checks_leaves_a_note(self, tmp_path):
        write(tmp_path, "empty.py", "X = 1\n")
        registry = Registry()

        registry.load_directory(tmp_path)

        assert codes(registry) == ["plugin_empty"]
        assert str(tmp_path.resolve()) in registry.notes[0].message

    def test_a_broken_init_leaves_a_note_and_loads_nothing_from_the_directory(
        self, tmp_path
    ):
        write(tmp_path, "__init__.py", "raise ImportError('no init')\n")
        write(tmp_path, "fine.py", check_source("fine"))
        registry = Registry()

        registry.load_directory(tmp_path)

        assert len(registry) == 0
        assert codes(registry) == ["plugin_load_failed"]

    def test_notes_belong_to_the_registry_that_loaded(self, tmp_path):
        write(tmp_path, "empty.py", "X = 1\n")
        first, second = Registry(), Registry()

        first.load_directory(tmp_path)

        assert len(first.notes) == 1
        assert second.notes == ()

    def test_an_id_taken_by_an_earlier_source_is_a_note_not_an_error(self, tmp_path):
        write(tmp_path / "first", "one.py", check_source("shared"))
        write(tmp_path / "second", "two.py", check_source("shared"))
        registry = Registry()

        registry.load_directory(tmp_path / "first")
        registry.load_directory(tmp_path / "second")

        assert registry.source_of("shared").endswith("first")
        (note,) = registry.notes
        assert note.code == "check_id_conflict"
        assert "'shared'" in note.message and "first" in note.message


class TestEntryPointsInProcess:
    @pytest.fixture
    def offer(self, monkeypatch, tmp_path):
        """Make ``plugins.entry_points`` return the given points per group."""
        monkeypatch.syspath_prepend(str(tmp_path))

        def install(**by_group: list[EntryPoint]) -> None:
            monkeypatch.setattr(
                plugins,
                "entry_points",
                lambda group: tuple(by_group.get(group.replace(".", "_"), ())),
            )

        return install

    def test_checks_and_packs_load_and_the_failing_one_leaves_a_note(
        self, offer, tmp_path
    ):
        write(
            tmp_path,
            "ep_plugin_a.py",
            check_source("from_ep")
            + "\nfrom dsbf.engine import Pack\nPACK = Pack('ep_pack', [from_ep])\n"
            "MANY = [from_ep]\n",
        )
        offer(
            dsbf_checks=[
                EntryPoint("good", "ep_plugin_a:from_ep", "dsbf.checks"),
                EntryPoint("many", "ep_plugin_a:MANY", "dsbf.checks"),
                EntryPoint("bad", "ep_plugin_missing:thing", "dsbf.checks"),
                EntryPoint("wrong", "json:dumps", "dsbf.checks"),
            ],
            dsbf_packs=[EntryPoint("pack", "ep_plugin_a:PACK", "dsbf.packs")],
        )
        registry = Registry()

        registry.load_entry_points()

        assert [c.id for c in registry] == ["from_ep"]
        assert registry.packs == {"ep_pack": ("from_ep",)}
        assert registry.source_of("from_ep") == "entry point dsbf.checks:good"
        assert sorted(codes(registry)) == ["plugin_invalid", "plugin_load_failed"]
        messages = " ".join(note.message for note in registry.notes)
        assert "dsbf.checks:bad" in messages and "dsbf.checks:wrong" in messages

    def test_a_pack_entry_point_must_load_a_pack(self, offer):
        offer(dsbf_packs=[EntryPoint("wrong", "json:dumps", "dsbf.packs")])
        registry = Registry()

        registry.load_entry_points()

        assert codes(registry) == ["plugin_invalid"]
        assert "expected a Pack" in registry.notes[0].message

    def test_loading_entry_points_twice_is_a_no_op(self, offer, tmp_path):
        write(tmp_path, "ep_plugin_b.py", check_source("only_once"))
        offer(dsbf_checks=[EntryPoint("one", "ep_plugin_b:only_once", "dsbf.checks")])
        registry = Registry()

        registry.load_entry_points()
        registry.load_entry_points()

        assert len(registry) == 1
        assert registry.notes == ()


class TestInstalledPackage:
    """A package whose metadata sits on ``sys.path``, found in a fresh interpreter."""

    def install(self, site: Path) -> None:
        package = site / "dsbf_demo_plugin"
        write(package, "__init__.py", "")
        write(
            package,
            "checks.py",
            check_source("demo_rule", ("demo",))
            + "\nfrom dsbf.engine import Pack\nPACK = Pack('demo_pack', [demo_rule])\n",
        )
        info = site / "dsbf_demo_plugin-1.0.0.dist-info"
        write(
            info,
            "METADATA",
            "Metadata-Version: 2.1\nName: dsbf-demo-plugin\nVersion: 1.0.0\n",
        )
        write(
            info,
            "entry_points.txt",
            """\
            [dsbf.checks]
            demo_rule = dsbf_demo_plugin.checks:demo_rule

            [dsbf.packs]
            demo = dsbf_demo_plugin.checks:PACK
            """,
        )

    def discover(self, site: Path | None) -> dict:
        prelude = f"sys.path.insert(0, {str(site)!r}); " if site else ""
        code = (
            "import sys, json; "
            + prelude
            + "from dsbf.engine import Registry; r = Registry(); "
            "r.load_entry_points(); "
            "print(json.dumps({'ids': [c.id for c in r], 'packs': r.packs, "
            "'source': r.source_of('demo_rule') if 'demo_rule' in r else None, "
            "'notes': [n.code for n in r.notes]}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    def test_an_installed_package_is_discovered_and_an_absent_one_is_not(
        self, tmp_path
    ):
        self.install(tmp_path)

        found = self.discover(tmp_path)
        absent = self.discover(None)

        assert found == {
            "ids": ["demo_rule"],
            "packs": {"demo": ["demo_rule"], "demo_pack": ["demo_rule"]},
            "source": "entry point dsbf.checks:demo_rule",
            "notes": [],
        }
        assert absent["ids"] == []
