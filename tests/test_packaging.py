"""Packaging and import-graph proofs (architecture contract, section 11).

v1.0.0's single worst defect class was a *packaging* one, not a logic one: the wheel
shipped four empty directories that resolved as namespace packages and shadowed their own
sibling modules, so `signledger.cli` imported as an empty namespace and the console script
was dead on arrival. No unit test could see it, because in the source tree the shadowing
directory and the real module sat side by side and the import happened to resolve.

Everything here is therefore about the *whole package*, not about one function:

* every module imports, and imports first, in a fresh interpreter;
* no subpackage is a namespace package;
* the built wheel contains every module the package list promises;
* the console script actually runs out of that wheel.
"""

import json
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "signledger"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _run_python(code: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run ``code`` in a *fresh* interpreter and return the completed process.

    A fresh interpreter is the whole point: an import cycle or a shadowing directory is
    invisible once ``sys.modules`` has been warmed by an earlier import in this process.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        **kwargs,
    )


#: Bootstrap that makes Django's app registry available.
#:
#: ``signledger.contrib.django.models`` defines a ``models.Model`` subclass at module
#: scope, and Django refuses to build one before ``INSTALLED_APPS`` is readable. That is
#: universal Django behaviour, not a signledger defect: ``import django.contrib.auth.models``
#: fails with the identical ``ImproperlyConfigured`` on a bare interpreter. Configuring the
#: registry is what a Django project does, so it is what the walk does.
_DJANGO_BOOTSTRAP = """
import django
from django.conf import settings
settings.configure(
    INSTALLED_APPS=["django.contrib.contenttypes", "signledger.contrib.django"],
    DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
    USE_TZ=True,
)
django.setup()
"""

_WALK = """
import importlib, json, pkgutil, sys
import signledger

names = sorted(m.name for m in pkgutil.walk_packages(signledger.__path__, "signledger."))
failures = []
for name in names:
    try:
        importlib.import_module(name)
    except BaseException as err:
        failures.append([name, type(err).__name__, str(err)[:200]])
print(json.dumps({"names": names, "failures": failures}))
"""


def _walk_modules(cwd: Path = REPO_ROOT, env: Any = None) -> Dict[str, Any]:
    """Import every module under ``signledger`` in a fresh interpreter."""
    result = _run_python(_DJANGO_BOOTSTRAP + _WALK, cwd=str(cwd), env=env)
    assert result.returncode == 0, f"walk crashed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _source_modules() -> List[str]:
    """Every importable module name in the source tree, derived from the files on disk."""
    names = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(REPO_ROOT)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return names


# --------------------------------------------------------------------------------------
# the import graph
# --------------------------------------------------------------------------------------


class TestImportGraph:
    """The one-line check that would have caught the entire v1 shadowing catastrophe."""

    def test_every_module_in_the_package_imports(self) -> None:
        walked = _walk_modules()
        assert walked["failures"] == []

    def test_the_walk_actually_reaches_every_module_on_disk(self) -> None:
        """A walk that silently visits nothing would pass the test above vacuously."""
        walked = set(_walk_modules()["names"]) | {"signledger"}
        assert set(_source_modules()) <= walked

    def test_no_subpackage_resolves_as_a_namespace_package(self) -> None:
        """Every directory under ``signledger`` must carry a real ``__init__.py``.

        A directory without one becomes an implicit namespace package that can shadow a
        sibling module of the same name. That is precisely how v1's empty ``cli/``
        directory killed ``cli.py``.
        """
        missing = [
            str(directory.relative_to(REPO_ROOT))
            for directory in sorted(PACKAGE_ROOT.rglob("*"))
            if directory.is_dir()
            and "__pycache__" not in directory.parts
            and not (directory / "__init__.py").exists()
        ]
        assert missing == []

    def test_no_directory_shadows_a_sibling_module_of_the_same_name(self) -> None:
        """``cli/`` next to ``cli.py`` is ambiguous; the directory wins and the module dies."""
        collisions = [
            str(directory.relative_to(REPO_ROOT))
            for directory in sorted(PACKAGE_ROOT.rglob("*"))
            if directory.is_dir()
            and "__pycache__" not in directory.parts
            and directory.with_suffix(".py").exists()
        ]
        assert collisions == []

    @pytest.mark.parametrize(
        "module",
        [
            "signledger",
            "signledger.core.entry",
            "signledger.core.ledger",
            "signledger.core.verification",
            "signledger.crypto.canonical",
            "signledger.crypto.hashing",
            "signledger.crypto.merkle",
            "signledger.crypto.signatures",
            "signledger.crypto.keys",
            "signledger.backends.base",
            "signledger.backends.sqlite",
            "signledger.anchor.sinks",
            "signledger.cli",
            "signledger.observability",
        ],
    )
    def test_the_module_can_be_the_first_thing_a_fresh_interpreter_imports(
        self, module: str
    ) -> None:
        """No module may depend on another having been imported first.

        Wave 2 found a live cycle here that the existing probe could not see, because the
        probe imported ``signledger`` first and that pre-warmed ``sys.modules``. This one
        imports **only** the target, in its own interpreter.
        """
        result = _run_python(f"import {module}", cwd=str(REPO_ROOT))
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        "module",
        [
            "signledger.core.entry",
            "signledger.core.exceptions",
            "signledger.crypto.canonical",
            "signledger.crypto.hashing",
            "signledger.crypto.merkle",
            "signledger.crypto.signatures",
            "signledger.crypto.keys",
            "signledger.backends.base",
            "signledger.backends.memory",
            "signledger.anchor.checkpoint",
        ],
    )
    def test_the_module_graph_has_no_cycle_the_package_init_is_hiding(
        self, module: str
    ) -> None:
        """The submodule graph must be acyclic *on its own*.

        ``signledger/__init__.py`` imports ``core`` before ``crypto``, which happens to
        walk the graph in an order that resolves. That is luck, not structure: a genuine
        cycle sits underneath, and reordering two lines in ``__init__`` would expose it as
        an ``ImportError`` in every consumer at once.

        Installing a bare namespace package for ``signledger`` removes the lucky ordering
        and leaves only the real dependency structure.
        """
        probe = f"""
import sys, types, importlib
package = types.ModuleType("signledger")
package.__path__ = [{str(PACKAGE_ROOT)!r}]
sys.modules["signledger"] = package
importlib.import_module({module!r})
"""
        result = _run_python(probe, cwd=str(REPO_ROOT))
        assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# the built wheel
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: Any) -> Path:
    """Build a real wheel from the repo and return its path."""
    pytest.importorskip("build", reason="the `build` package is not installed")
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="session")
def wheel_contents(wheel: Path) -> List[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


@pytest.fixture(scope="session")
def installed_wheel(wheel: Path, tmp_path_factory: Any) -> Iterator[Path]:
    """Unpack the wheel into its own directory, isolated from the source tree.

    Importing from here rather than from the repo is what makes the test about the
    *artifact* a user installs. The source tree can be perfectly healthy while the wheel
    is missing a whole subpackage - which is exactly the state this repo was in.
    """
    target = tmp_path_factory.mktemp("installed")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
    yield target


class TestWheelContents:
    """What a user actually receives from ``pip install signledger``."""

    def test_every_source_module_is_present_in_the_wheel(
        self, wheel_contents: List[str]
    ) -> None:
        """A module absent from the wheel is a module that does not exist for users.

        This is the check that catches an incomplete ``[tool.setuptools] packages`` list.
        v1 shipped four empty directories; this repo shipped a Django app with no
        migrations, so ``manage.py migrate`` created no table and every audited request
        returned 500 under the default fail-closed policy.
        """
        shipped = {name for name in wheel_contents if name.endswith(".py")}
        expected = {
            str(path.relative_to(REPO_ROOT))
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))
            if "__pycache__" not in path.parts
        }
        assert expected - shipped == set()

    def test_the_django_migration_ships(self, wheel_contents: List[str]) -> None:
        """Pinned on its own because its absence is silent and catastrophic."""
        assert "signledger/contrib/django/migrations/0001_initial.py" in wheel_contents

    def test_the_py_typed_marker_ships(self, wheel_contents: List[str]) -> None:
        """Without it every downstream ``mypy`` run silently ignores our annotations."""
        assert "signledger/py.typed" in wheel_contents

    def test_the_wheel_carries_the_current_version(self, wheel: Path) -> None:
        import signledger

        assert f"-{signledger.__version__}-" in wheel.name

    def test_the_wheel_ships_no_empty_namespace_directory(
        self, wheel_contents: List[str]
    ) -> None:
        """Every shipped package directory must contain an ``__init__.py``.

        The literal v1 defect: four directories in the wheel with no ``__init__.py``.
        """
        packages = {name.rsplit("/", 1)[0] for name in wheel_contents if name.endswith(".py")}
        without_init = sorted(
            package
            for package in packages
            if package.startswith("signledger") and f"{package}/__init__.py" not in wheel_contents
        )
        assert without_init == []


class TestInstalledWheel:
    """The wheel's contents, exercised as an installed package."""

    def _env(self, installed_wheel: Path) -> Dict[str, str]:
        """PYTHONPATH with the unpacked wheel *ahead of* everything else.

        Third-party dependencies still resolve from the real site-packages; only
        ``signledger`` itself is taken from the artifact.
        """
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = str(installed_wheel) + os.pathsep + sysconfig.get_paths()["purelib"]
        return env

    def test_every_module_imports_from_the_installed_wheel(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        walked = _walk_modules(cwd=tmp_path, env=self._env(installed_wheel))
        assert walked["failures"] == []

    def test_the_installed_package_is_the_one_being_imported(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        """Guards the test above: if the repo were still on the path, it would prove nothing."""
        result = _run_python(
            "import signledger; print(signledger.__file__)",
            cwd=str(tmp_path),
            env=self._env(installed_wheel),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith(str(installed_wheel))

    def test_the_console_script_entry_point_verifies_a_real_ledger(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        """v1's console script pointed at a module the wheel did not contain.

        ``--help`` would only prove the parser is intact. Building a ledger and then
        verifying it *through the entry point* proves the command does its job out of the
        artifact, which is the claim that actually matters.
        """
        database = tmp_path / "cli.db"
        script = f"""
import sys
from signledger import Ledger
from signledger.backends.sqlite import SQLiteBackend
from signledger.cli import main

with Ledger(SQLiteBackend({str(database)!r})) as ledger:
    ledger.append({{"event": "console-script"}})

sys.exit(main(["verify", "-b", "sqlite:{database}"]))
"""
        result = _run_python(script, cwd=str(tmp_path), env=self._env(installed_wheel))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    def test_the_console_script_reports_failure_on_an_empty_store(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        """Guards the test above: a `main` that always returned 0 would pass it.

        v1's CLI printed a tick on an empty ledger. Contract section 5 rule 1: zero
        entries verified is never success.
        """
        script = """
import sys
from signledger.cli import main
sys.exit(main(["verify", "-b", "sqlite://"]))
"""
        result = _run_python(script, cwd=str(tmp_path), env=self._env(installed_wheel))
        assert result.returncode != 0

    def test_the_package_runs_as_a_module_from_the_installed_wheel(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        """``python -m signledger`` must reach the same entry point."""
        result = subprocess.run(
            [sys.executable, "-m", "signledger", "--help"],
            cwd=str(tmp_path),
            env=self._env(installed_wheel),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "verify" in result.stdout

    def test_a_ledger_can_be_written_and_verified_using_only_the_installed_wheel(
        self, installed_wheel: Path, tmp_path: Path
    ) -> None:
        """End to end through the artifact: append, close, reopen, verify.

        A wheel that imports but cannot complete the library's one job is still broken.
        """
        script = f"""
import sys
from signledger import Ledger
from signledger.backends.sqlite import SQLiteBackend

path = {str(tmp_path / "wheel.db")!r}
with Ledger(SQLiteBackend(path)) as ledger:
    for index in range(5):
        ledger.append({{"event": "install-check", "index": index}})
    ledger_id = ledger.ledger_id

with Ledger(SQLiteBackend(path), ledger_id=ledger_id) as reopened:
    report = reopened.check_integrity()
print("OK" if report.ok and report.entries_verified == 5 else "BROKEN")
"""
        result = _run_python(script, cwd=str(tmp_path), env=self._env(installed_wheel))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("OK")


class TestPackageMetadata:
    """The declared package list must describe the tree that actually exists."""

    def test_the_declared_package_list_matches_the_directories_on_disk(self) -> None:
        """An omitted entry silently drops a subpackage from every wheel ever built."""
        tomllib = pytest.importorskip("tomllib")
        config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        declared = set(config["tool"]["setuptools"]["packages"])
        on_disk = {
            ".".join(directory.relative_to(REPO_ROOT).parts)
            for directory in sorted(PACKAGE_ROOT.rglob("*"))
            if directory.is_dir() and (directory / "__init__.py").exists()
        } | {"signledger"}
        assert declared == on_disk

    def test_the_console_script_target_is_importable_and_callable(self) -> None:
        tomllib = pytest.importorskip("tomllib")
        config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        target = config["project"]["scripts"]["signledger"]
        module_name, _, attribute = target.partition(":")
        result = _run_python(
            f"import {module_name} as m; assert callable(getattr(m, {attribute!r}))",
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
