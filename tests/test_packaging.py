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
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "signledger"

#: Every CPython minor release this package claims to support, oldest first.
#:
#: This tuple is the single normative statement of the support window, and it is
#: deliberately not a mirror of the classifier list: every entry is *exercised* below by
#: resolving the whole declared dependency set against it. Adding a version here without
#: the wheels to back it fails ``TestDependencyFloors``; adding it to ``pyproject.toml``
#: without adding it here fails ``TestDeclaredPythonSupport``.
#:
#: 2.0.0 declared ``requires-python = ">=3.9"`` and classifiers that stopped at 3.13,
#: a release after 3.14 had been verified end to end. Nothing could see the drift because
#: nothing compared the two lists. This is that comparison.
SUPPORTED_PYTHONS: Tuple[str, ...] = ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14")

#: Optional dependency groups that add a *runtime* integration.
RUNTIME_EXTRAS: Tuple[str, ...] = (
    "signatures",
    "postgresql",
    "mongodb",
    "flask",
    "fastapi",
    "django",
)

#: Every dependency group in ``pyproject.toml``: the base list plus every extra.
DEPENDENCY_GROUPS: Tuple[str, ...] = ("dependencies", *RUNTIME_EXTRAS, "all", "test", "dev")

#: The groups a *consumer* installs, and which must therefore install from wheels.
#:
#: ``dev`` is exempt and named rather than silently omitted: mutmut has never published a
#: wheel at any version, and its sdist is pure Python, so building it needs no toolchain.
#: Every other group reaches a C extension somewhere and must not.
WHEEL_ONLY_GROUPS: Tuple[str, ...] = ("dependencies", *RUNTIME_EXTRAS, "all", "test")

_VERSION_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")


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


# --------------------------------------------------------------------------------------
# declared dependency metadata
#
# Everything below exists because 2.0.0's metadata described a package that could not be
# installed. The bounds were plausible and none of them had ever been resolved:
#
#   * ``typing-extensions>=4.0.0`` was a hard dependency no module imports, and no version
#     in its range could coexist with ``pydantic>=2.0.0`` (pydantic 2.0 needs >=4.6.1);
#   * ``fastapi>=0.100.0`` and ``pydantic>=2.0.0`` were mutually unsatisfiable, because
#     every FastAPI release that supports pydantic v2 excludes exactly 2.0.0 and 2.0.1;
#   * ``psycopg[binary]>=3.1`` had no wheel for macOS arm64 or for cp314;
#   * ``flask>=2.0.0`` resolved but did not *import*: flask 2.0 caps nothing on Werkzeug,
#     so a fresh resolve pairs it with Werkzeug 3 and ``import flask`` dies on
#     ``url_quote`` before signledger is ever reached.
#
# A dependency bound is a claim about the world. These tests check the claim against the
# world, which is the only place the answer lives.
# --------------------------------------------------------------------------------------


def _pyproject() -> Dict[str, Any]:
    tomllib = pytest.importorskip("tomllib")
    config: Dict[str, Any] = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return config


def _declared_group(config: Dict[str, Any], group: str) -> List[str]:
    """The requirement strings declared for one dependency group."""
    if group == "dependencies":
        return list(config["project"]["dependencies"])
    return list(config["project"]["optional-dependencies"][group])


def _requirements_for(config: Dict[str, Any], group: str) -> List[str]:
    """What ``pip install signledger[<group>]`` actually asks for: base plus the extra.

    Resolving an extra on its own would have missed the fastapi/pydantic conflict
    entirely, because it is a conflict *between* an extra and the base list.
    """
    base = _declared_group(config, "dependencies")
    if group == "dependencies":
        return base
    return [*base, *_declared_group(config, group)]


def _pin_to_floor(requirement: str) -> str:
    """Rewrite ``pkg[extra]>=X ; marker`` as ``pkg[extra]==X ; marker``.

    A ``>=`` floor claims that version X is installable. Pinning to it is the only way to
    find out; the unpinned range hides an unsatisfiable floor behind a newer release.
    """
    requirements_module = pytest.importorskip("packaging.requirements")
    parsed = requirements_module.Requirement(requirement)
    floors = [spec.version for spec in parsed.specifier if spec.operator == ">="]
    assert len(floors) == 1, f"{requirement!r} must declare exactly one `>=` floor"
    extras = f"[{','.join(sorted(parsed.extras))}]" if parsed.extras else ""
    marker = f" ; {parsed.marker}" if parsed.marker is not None else ""
    return f"{parsed.name}{extras}=={floors[0]}{marker}"


def _first_applicable_pythons(requirements: Sequence[str]) -> List[str]:
    """The oldest supported Python at which each requirement's marker starts to apply.

    A floor guarded by ``python_version >= "3.10"`` is not a claim about 3.9, and pinning
    it there proves nothing: the marker evaluates false and the requirement disappears.
    The floors have to be checked at the oldest Python where they are actually requested.
    """
    requirements_module = pytest.importorskip("packaging.requirements")
    applicable = set()
    for requirement in requirements:
        marker = requirements_module.Requirement(requirement).marker
        matching = [
            version
            for version in SUPPORTED_PYTHONS
            if marker is None or marker.evaluate({"python_version": version})
        ]
        assert matching, f"{requirement!r} applies to no supported Python"
        applicable.add(matching[0])
    return sorted(applicable, key=SUPPORTED_PYTHONS.index)


def _resolve(uv: str, requirements: Sequence[str], python_version: str, *flags: str) -> Any:
    """Resolve ``requirements`` for ``python_version``, installing nothing.

    ``uv pip compile`` is a pure resolver: it reads no installed packages, so the answer
    cannot depend on whatever happens to be in the developer's virtualenv, and it honours
    wheel platform *and* ABI tags, which is what lets it see a floor that resolves on
    Linux CI and not on the machine in front of you. That asymmetry is exactly how
    ``psycopg[binary]>=3.1`` survived to a release.
    """
    return subprocess.run(
        [
            uv,
            "pip",
            "compile",
            "--no-config",
            "--quiet",
            "--no-header",
            "--python-version",
            python_version,
            *flags,
            "-",
        ],
        input="\n".join(requirements) + "\n",
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="session")
def resolver() -> str:
    """Path to ``uv``, once it has been shown to reach a package index.

    The probe matters. Without it a network outage would turn every bound in the project
    into a resolution failure and train everyone to ignore this file; with it, an outage
    skips and a genuinely unsatisfiable bound fails.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; dependency bounds cannot be resolved")
    probe = _resolve(uv, ["pydantic>=2"], SUPPORTED_PYTHONS[0])
    if probe.returncode != 0:
        pytest.skip(f"no usable package index: {probe.stderr.strip()[:300]}")
    return uv


class TestDeclaredPythonSupport:
    """``requires-python``, the classifiers and reality must all say the same thing."""

    def test_requires_python_starts_at_the_oldest_supported_release(self) -> None:
        assert _pyproject()["project"]["requires-python"] == f">={SUPPORTED_PYTHONS[0]}"

    def test_the_classifiers_name_every_supported_release_and_no_other(self) -> None:
        """The drift that shipped: classifiers a release behind ``requires-python``."""
        classifiers = _pyproject()["project"]["classifiers"]
        declared = [
            match.group(1)
            for match in (_VERSION_CLASSIFIER.match(entry) for entry in classifiers)
            if match is not None
        ]
        assert declared == list(SUPPORTED_PYTHONS)

    def test_the_classifier_versions_form_an_unbroken_range(self) -> None:
        """Derived from the classifiers alone, so a gap fails even if the tuple grows one."""
        classifiers = _pyproject()["project"]["classifiers"]
        minors = [
            int(match.group(1).split(".")[1])
            for match in (_VERSION_CLASSIFIER.match(entry) for entry in classifiers)
            if match is not None
        ]
        assert minors == list(range(minors[0], minors[-1] + 1))

    def test_the_running_interpreter_is_a_release_the_package_claims(self) -> None:
        """Stops ``SUPPORTED_PYTHONS`` from becoming fiction the suite never touches."""
        running = f"{sys.version_info[0]}.{sys.version_info[1]}"
        assert running in SUPPORTED_PYTHONS

    def test_the_package_declares_that_it_is_python_3_only(self) -> None:
        classifiers = _pyproject()["project"]["classifiers"]
        assert "Programming Language :: Python :: 3 :: Only" in classifiers

    def test_the_package_declares_the_interpreter_it_is_verified_on(self) -> None:
        classifiers = _pyproject()["project"]["classifiers"]
        assert "Programming Language :: Python :: Implementation :: CPython" in classifiers

    def test_no_interpreter_the_package_does_not_work_on_is_claimed(self) -> None:
        """``SQLiteBackend`` does not work on PyPy, so PyPy is not advertised.

        Pinned as a test because a classifier is a promise, and the tempting way to make
        an implementation list look complete is to add the one that is broken.
        """
        classifiers = _pyproject()["project"]["classifiers"]
        assert "Programming Language :: Python :: Implementation :: PyPy" not in classifiers


class TestDependencyFloors:
    """Every declared bound must describe a set of packages that can actually be built."""

    def test_the_groups_under_test_are_exactly_the_groups_declared(self) -> None:
        """A new extra must be resolved by this file, not silently exempt from it."""
        config = _pyproject()
        declared = {"dependencies", *config["project"]["optional-dependencies"]}
        assert declared == set(DEPENDENCY_GROUPS)

    def test_the_all_extra_is_exactly_the_union_of_the_runtime_extras(self) -> None:
        """``all`` is a copy of six other lists, and a copy drifts.

        It listed ``flask>=2.0.0`` and ``psycopg[binary]>=3.1`` for the same reason the
        individual extras did, and would have kept listing them after a partial fix.
        """
        config = _pyproject()
        union = {
            requirement
            for extra in RUNTIME_EXTRAS
            for requirement in _declared_group(config, extra)
        }
        assert set(_declared_group(config, "all")) == union

    @pytest.mark.parametrize("group", DEPENDENCY_GROUPS)
    def test_every_declared_floor_is_installable_at_its_exact_minimum(
        self, resolver: str, group: str
    ) -> None:
        """``>=X`` claims X can be installed. Pin to X and find out."""
        requirements = _requirements_for(_pyproject(), group)
        pinned = [_pin_to_floor(requirement) for requirement in requirements]
        resolved = {
            python_version: _resolve(resolver, pinned, python_version)
            for python_version in _first_applicable_pythons(requirements)
        }
        unsatisfiable = {
            python_version: result.stderr.strip()
            for python_version, result in resolved.items()
            if result.returncode != 0
        }
        assert unsatisfiable == {}, f"[{group}] floors {pinned}"

    @pytest.mark.parametrize("group", WHEEL_ONLY_GROUPS)
    def test_no_consumer_facing_floor_forces_a_source_build(
        self, resolver: str, group: str
    ) -> None:
        """A floor that resolves but has no wheel is a different, quieter defect.

        ``pymongo>=4.0.0`` installed - by compiling the ``_cmessage`` C extension from
        source, because 4.3.2 is the first pymongo with a macOS arm64 wheel. That needs a
        toolchain and Python headers the target may not have (slim images, Windows), and
        where it *does* succeed the user is silently running different code from everyone
        on a wheel. Not unsatisfiable, so the test above cannot see it; still wrong.
        """
        requirements = _requirements_for(_pyproject(), group)
        pinned = [_pin_to_floor(requirement) for requirement in requirements]
        built = {
            python_version: _resolve(resolver, pinned, python_version, "--only-binary", ":all:")
            for python_version in _first_applicable_pythons(requirements)
        }
        from_source = {
            python_version: result.stderr.strip()
            for python_version, result in built.items()
            if result.returncode != 0
        }
        assert from_source == {}, f"[{group}] floors {pinned}"

    @pytest.mark.parametrize("python_version", SUPPORTED_PYTHONS)
    @pytest.mark.parametrize("group", ("dependencies", "all", "test", "dev"))
    def test_every_declared_group_resolves_on_every_supported_python(
        self, resolver: str, group: str, python_version: str
    ) -> None:
        """A floor that installs is not enough; the declared *range* must too.

        ``all`` is a superset of every runtime extra, so covering it covers them: any
        solution to the superset restricts to a solution for each subset.
        """
        requirements = _requirements_for(_pyproject(), group)
        result = _resolve(resolver, requirements, python_version)
        assert result.returncode == 0, (
            f"[{group}] does not resolve on python {python_version}:\n{result.stderr}"
        )


class TestTestExtraRunsTheSuite:
    """The ``test`` extra exists to run the tests. It must be able to run them."""

    def _venv_python(self, uv: str, root: Path) -> Path:
        subprocess.run(
            [uv, "venv", "--no-config", "--quiet", "--python", sys.executable, str(root)],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        candidates = [root / "bin" / "python", root / "Scripts" / "python.exe"]
        found = [candidate for candidate in candidates if candidate.exists()]
        assert found, f"uv created no interpreter under {root}"
        return found[0]

    def test_the_test_extra_can_collect_the_fastapi_contrib_module(
        self, resolver: str, tmp_path: Path
    ) -> None:
        """The FastAPI contrib tests collected **zero** tests against a fresh resolve.

        ``starlette`` 1.3 deprecates using ``httpx`` with ``starlette.testclient`` and
        wants ``httpx2``; the deprecation warning becomes a hard error under this
        project's ``filterwarnings = ["error"]``, so ``fastapi.testclient`` raises at
        import and the whole module fails collection. 30 000 lines of FastAPI tests
        silently stopped existing, and a green run said nothing about it.

        Resolving the declared ranges rather than a lockfile is deliberate: this is the
        canary for the next such deprecation, and a pinned environment cannot be one.
        """
        config = _pyproject()
        python = self._venv_python(resolver, tmp_path / "venv")
        requirements = [
            *_requirements_for(config, "fastapi"),
            *_declared_group(config, "test"),
        ]
        install = subprocess.run(
            [resolver, "pip", "install", "--no-config", "--quiet", "--python", str(python)]
            + requirements,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert install.returncode == 0, install.stderr

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env.pop("VIRTUAL_ENV", None)
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "tests/test_contrib_fastapi.py",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "no tests collected" not in result.stdout
        assert "tests collected" in result.stdout


# --------------------------------------------------------------------------------------
# the built sdist
#
# The sdist is the artifact a distribution packager fetches to *re-verify* a security
# library from source. 2.0.0's shipped ``tests/*.py`` without ``tests/conftest.py``, its
# frozen SLCS/1 vectors, ``docs/``, ``examples/``, ``CHANGELOG.md`` or ``SECURITY.md``:
# a collection error and ~194 fixture errors for anyone who tried. Half-shipping a test
# suite is worse than shipping none, because it reads as a failing suite rather than an
# absent one.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sdist(tmp_path_factory: Any) -> Path:
    """Build a real sdist from the repo and return its path."""
    pytest.importorskip("build", reason="the `build` package is not installed")
    out = tmp_path_factory.mktemp("sdist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"sdist build failed:\n{result.stdout}\n{result.stderr}"
    archives = list(out.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected exactly one sdist, got {archives}"
    return archives[0]


@pytest.fixture(scope="session")
def sdist_contents(sdist: Path) -> List[str]:
    """Archive members, with the ``signledger-<version>/`` prefix stripped."""
    with tarfile.open(sdist) as archive:
        return sorted(name.split("/", 1)[1] for name in archive.getnames() if "/" in name)


@pytest.fixture(scope="session")
def unpacked_sdist(sdist: Path, tmp_path_factory: Any) -> Path:
    """The sdist unpacked into its own directory, exactly as a packager would."""
    target = tmp_path_factory.mktemp("unpacked")
    with tarfile.open(sdist) as archive:
        archive.extractall(target, filter="data")
    roots = [entry for entry in target.iterdir() if entry.is_dir()]
    assert len(roots) == 1, f"expected exactly one directory in the sdist, got {roots}"
    return roots[0]


class TestSdistContents:
    """What a packager receives from ``pip download --no-binary :all: signledger``."""

    def test_the_sdist_ships_the_conftest_the_suite_cannot_run_without(
        self, sdist_contents: List[str]
    ) -> None:
        """Every backend, ledger and signing fixture in the suite lives in this one file."""
        assert "tests/conftest.py" in sdist_contents

    def test_the_sdist_ships_the_frozen_canonical_vectors(
        self, sdist_contents: List[str]
    ) -> None:
        """SLCS/1 is a byte format; without the vectors it cannot be re-verified at all."""
        assert "tests/vectors/slcs1.json" in sdist_contents

    def test_the_sdist_ships_every_test_module_in_the_repository(
        self, sdist_contents: List[str]
    ) -> None:
        on_disk = {f"tests/{path.name}" for path in sorted((REPO_ROOT / "tests").glob("*.py"))}
        assert on_disk - set(sdist_contents) == set()

    def test_the_sdist_ships_the_files_the_suite_executes(
        self, sdist_contents: List[str]
    ) -> None:
        """``test_docs_execute.py`` runs the fenced blocks; ``test_examples.py`` runs these."""
        required = {
            "README.md",
            "SECURITY.md",
            "CHANGELOG.md",
            "LICENSE",
            "docs/ARCHITECTURE.md",
            "examples/README.md",
            "examples/quickstart.py",
            "examples/banking_sox.py",
            "examples/healthcare_hipaa.py",
        }
        assert required - set(sdist_contents) == set()

    def test_the_sdist_ships_no_build_or_scratch_artefact(
        self, sdist_contents: List[str]
    ) -> None:
        """Grafting whole directories is how a stray ledger.db or .pyc gets published.

        MANIFEST.in excludes only ``*.py[cod]``, on purpose: distutils warns for every
        pattern that matches nothing, so a speculative exclusion list makes every release
        build print warnings nobody reads. This is the guard instead, and it fails on the
        artifact rather than on a guess about what might end up in it.
        """
        junk = [
            name
            for name in sdist_contents
            if Path(name).name.startswith(".")
            or "__pycache__" in name
            or name.endswith((".pyc", ".pyo", ".so", ".db", ".db-wal", ".db-shm"))
        ]
        assert junk == []


class TestSdistIsSelfVerifying:
    """The acceptance test: unpack the sdist and run its own suite out of it."""

    def _pytest(self, root: Path, *arguments: str) -> Any:
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def test_the_unpacked_sdist_is_the_package_under_test(self, unpacked_sdist: Path) -> None:
        """Guards the two below: without this they could be testing the repo checkout."""
        result = _run_python(
            "import signledger; print(signledger.__file__)", cwd=str(unpacked_sdist)
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith(str(unpacked_sdist))

    def test_the_whole_suite_collects_from_the_unpacked_sdist(
        self, unpacked_sdist: Path
    ) -> None:
        """A missing conftest.py is a *collection* error: nothing runs, nothing reports."""
        result = self._pytest(unpacked_sdist, "tests", "--collect-only", "-q")
        assert result.returncode == 0, f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"

    def test_the_frozen_vectors_and_shared_fixtures_pass_from_the_unpacked_sdist(
        self, unpacked_sdist: Path
    ) -> None:
        """Collection is not enough: the data files the tests read must be there too."""
        result = self._pytest(
            unpacked_sdist,
            "tests/test_canonical_vectors.py",
            "tests/test_hashing.py",
            "-q",
        )
        assert result.returncode == 0, f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
