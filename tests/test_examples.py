"""Every shipped example is executed here, as a real subprocess, on every test run.

v1 shipped three examples and all three crashed on first use - the flagship Flask banking
app returned HTTP 500 from every endpoint. That is not an accident that better review
would have caught: an example nobody runs is an example nobody knows is broken, and the
only reliable way to know is to run it.

So these tests do not import the examples or inspect them. They run
``python examples/<name>.py`` end to end, require exit status 0, require a clean stderr,
and require the markers each example promises. An example that stops working fails CI.
"""

import os
import pathlib
import re
import subprocess
import sys
from typing import List, Tuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Skipped wholesale when the examples are not beside the tests.
#:
#: mutmut is the case this exists for. It copies ``source_paths`` and ``tests/`` into a
#: ``mutants/`` tree and runs a clean baseline there; ``examples/`` does not come along, so
#: every test below died on ``can't open file '.../mutants/examples/quickstart.py'``, the
#: baseline failed, and mutmut refused to check a single mutant while the CI job reported
#: success. That is the identical defect commit 6681640 fixed in the sibling
#: ``test_docs_execute.py`` and left standing here.
#:
#: The sibling walks up to find the real repository root. That is deliberately *not* done
#: here: it would make these tests run the real ``examples/`` against the real installed
#: package, so under mutmut they would execute unmutated source and could never kill a
#: mutant - a baseline that passes by testing the wrong tree. Being honestly absent from
#: the sweep beats being present and meaningless.
pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(),
    reason=(
        "examples/ is not beside the test suite; this tree was copied without it "
        "(mutmut does exactly that)"
    ),
)

#: Every example, with the stdout markers it promises to print. The markers are the
#: example's contract with this test: they name the *behaviour* being demonstrated, so a
#: script that still exits 0 while having quietly stopped verifying anything still fails.
EXPECTED: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "quickstart.py",
        (
            "entries appended",
            "integrity: OK",
            "the database refused the UPDATE",
            "tampering detected",
            "bad_hash",
            "assurance level: chain_only",
        ),
    ),
    (
        "banking_sox.py",
        (
            "loaded signing key from",
            "posted",
            "balance updated after the ledger accepted the entry",
            "refused: insufficient funds",
            "checkpoint published to",
            "ledger rejected the entry",
            "ACC-1001 balance unchanged: 2037.50 (was 2037.50)",
            "integrity: OK",
            "assurance level: worm",
        ),
    ),
    (
        "healthcare_hipaa.py",
        (
            "PHI access recorded",
            "captured in the ledger:",
            "NOT captured (redacted before the ledger ever saw it):",
            "no PHI found in the ledger payloads",
            "integrity: OK",
            "assurance level: signed",
        ),
    ),
]

EXAMPLE_NAMES = [name for name, _ in EXPECTED]

#: Examples that need the ``signatures`` extra. ``quickstart.py`` deliberately does not,
#: so a bare install still has something that runs.
NEEDS_CRYPTOGRAPHY = frozenset({"banking_sox.py", "healthcare_hipaa.py"})


def skip_without_dependencies(name: str) -> None:
    if name in NEEDS_CRYPTOGRAPHY:
        pytest.importorskip("cryptography", reason="install 'signledger[signatures]'")


def run_example(name: str) -> subprocess.CompletedProcess:
    """Run one example exactly as a reader would, from a clean working directory."""
    skip_without_dependencies(name)
    environment = dict(os.environ)
    # A reader would have run ``pip install signledger``. The checkout stands in for that
    # here; nothing is added to sys.path by the examples themselves, because an example
    # that needs a path hack to run is not an example of the installed package.
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONWARNINGS"] = "error"
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture(scope="module", params=EXAMPLE_NAMES)
def executed(request: pytest.FixtureRequest) -> subprocess.CompletedProcess:
    """Run each example once for the whole module rather than once per assertion."""
    return run_example(request.param)


def test_the_example_exits_cleanly(executed: subprocess.CompletedProcess) -> None:
    assert executed.returncode == 0, executed.stdout + executed.stderr


def test_the_example_writes_nothing_to_stderr(executed: subprocess.CompletedProcess) -> None:
    """``PYTHONWARNINGS=error`` is set, so a DeprecationWarning is a failure here too."""
    assert executed.stderr == ""


def test_the_example_prints_every_marker_it_promises(
    executed: subprocess.CompletedProcess,
) -> None:
    name = pathlib.Path(executed.args[1]).name
    markers = dict(EXPECTED)[name]
    missing = [marker for marker in markers if marker not in executed.stdout]

    assert missing == [], f"{name} did not print {missing}\n{executed.stdout}"


def test_the_example_states_the_assurance_level_it_actually_reached(
    executed: subprocess.CompletedProcess,
) -> None:
    """Contract section 1: the guarantee is deployment-dependent and reported at runtime,
    never asserted in prose. Every example has to say which rung it reached."""
    assert "assurance level:" in executed.stdout


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_the_example_never_binds_every_interface_or_enables_debug(name: str) -> None:
    """v1's flagship example ran a Flask app with ``debug=True`` bound to ``0.0.0.0``,
    which is a remote code execution console on every interface of the host."""
    source = (EXAMPLES / name).read_text(encoding="utf-8")

    assert "0.0.0.0" not in source
    assert not re.search(r"debug\s*=\s*True", source)


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_the_example_leaves_nothing_behind_in_the_repository(name: str) -> None:
    """An example that writes a ledger into the checkout is an example that fails the
    second time it is run."""
    skip_without_dependencies(name)
    before = {path.name for path in REPO_ROOT.iterdir()}
    run_example(name)

    assert {path.name for path in REPO_ROOT.iterdir()} == before


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_the_examples_readme_documents_the_example(name: str) -> None:
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert name in readme


def test_the_banking_example_moves_no_money_when_the_append_fails() -> None:
    """The marker above pins the printed balance, and this pins the closing one. If the
    example moved the money before appending, the failed transfer would still have taken
    10.00 out of ACC-1001 and both numbers would be 2027.50 - which is the v1 defect,
    reproduced end to end rather than asserted in a comment."""
    executed = run_example("banking_sox.py")

    assert "ACC-1001       2037.50" in executed.stdout


def test_the_readme_documents_no_example_that_does_not_exist() -> None:
    """v1's examples README advertised files by name and the commands it printed did not
    work. A name in the README is a promise that ``python examples/<name>`` runs."""
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    advertised = set(re.findall(r"examples/([A-Za-z0-9_]+\.py)", readme))
    missing = sorted(name for name in advertised if not (EXAMPLES / name).exists())

    assert missing == []
