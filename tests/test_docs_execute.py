"""Every documented Python example must actually run (contract section 11).

v1.0.0's README documented `InMemoryBackend`, `PostgreSQLBackend`, `RedisBackend`,
`signledger.compression.Compressor` and `ledger.get_entries()`. Not one of them existed.
The docs were not *out of date* - they were written against an API that had never been
implemented, and nothing in the project could tell, because no test ever ran them.

So these tests run them. Each fenced ``python`` block is executed in its own subprocess
with a fresh interpreter and a fresh working directory, and any traceback fails the test
with the block's line number in the source file.

Blocks that would need a live PostgreSQL, MongoDB or web server are not skipped by
guesswork: nothing in the documented examples requires one, and if that changes, the
example needs a marker rather than the test needing an exception.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Files whose Python examples must execute.
DOC_FILES = ["README.md", "examples/README.md"]

_FENCE = re.compile(r"^```(\w*)\s*$")


class Block(NamedTuple):
    """One fenced code block, with enough context to report it usefully."""

    path: str
    line: int
    language: str
    source: str

    def __str__(self) -> str:  # what pytest prints as the parametrization id
        return f"{self.path}:{self.line}"


def _blocks(relative_path: str) -> List[Block]:
    """Every fenced block in a documentation file, with its 1-based start line."""
    path = REPO_ROOT / relative_path
    if not path.exists():
        return []

    found: List[Block] = []
    language: str = ""
    start = 0
    buffer: List[str] = []
    inside = False

    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _FENCE.match(text)
        if match and not inside:
            inside, language, start, buffer = True, match.group(1), number, []
        elif match and inside:
            found.append(Block(relative_path, start, language, "\n".join(buffer) + "\n"))
            inside = False
        elif inside:
            buffer.append(text)

    assert not inside, f"{relative_path} has an unterminated code fence opened at line {start}"
    return found


def _python_blocks() -> List[Block]:
    return [
        block
        for name in DOC_FILES
        for block in _blocks(name)
        if block.language == "python"
    ]


PYTHON_BLOCKS = _python_blocks()


class TestDocumentedExamplesRun:
    def test_the_documentation_actually_contains_examples(self) -> None:
        """Guards every test below: an empty list would make them all pass vacuously.

        This is the exact shape of the v1 lie - a green suite that asserted nothing.
        """
        assert len(PYTHON_BLOCKS) >= 10

    @pytest.mark.parametrize("block", PYTHON_BLOCKS, ids=str)
    def test_the_example_executes_without_error(self, block: Block, tmp_path: Path) -> None:
        """Run one documented example in a fresh interpreter.

        ``cwd`` is a throwaway directory so an example that writes a file cannot leave one
        in the repo, and cannot accidentally read one another example left behind.
        """
        result = subprocess.run(
            [sys.executable, "-c", block.source],
            cwd=str(tmp_path),
            env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"\n{block.path} line {block.line} failed:\n"
            f"--- source ---\n{block.source}\n--- stderr ---\n{result.stderr}"
        )


def _env() -> dict:
    import os

    return dict(os.environ)


class TestDocumentedOutputIsTruthful:
    """A comment claiming a value is a claim. Check the ones that state a result."""

    def test_the_quick_start_reports_two_verified_entries(self, tmp_path: Path) -> None:
        """The README's first example prints `True` / `2` / `chain_only`."""
        source = """
from signledger import Ledger

with Ledger() as ledger:
    ledger.append({"actor": "alice", "action": "deleted-record", "record_id": 42})
    ledger.append({"actor": "bob", "action": "changed-role", "target": "carol"})
    report = ledger.check_integrity()

print(report.ok, report.entries_verified, report.assurance.value)
"""
        result = _run(source, tmp_path)
        assert result.stdout.strip().endswith("True 2 chain_only")

    def test_a_tampered_entry_really_does_report_not_ok(self, tmp_path: Path) -> None:
        """The verification example claims `False`. It must not print `True`."""
        source = """
from signledger import Ledger
from signledger.backends import MemoryBackend

backend = MemoryBackend()
with Ledger(backend) as ledger:
    ledger.append({"event": "one"})
    ledger.append({"event": "two"})
    stored = backend._entries[0]
    backend._entries[0] = stored.model_copy(update={"data": {"event": "tampered"}})
    report = ledger.check_integrity()

print(report.ok)
"""
        assert _run(source, tmp_path).stdout.strip().endswith("False")

    def test_the_decimal_example_prints_the_documented_message(self, tmp_path: Path) -> None:
        """The README quotes the exact error text; it must be the exact error text."""
        source = """
from decimal import Decimal
from signledger import Ledger
from signledger.core.exceptions import CanonicalizationError

with Ledger() as ledger:
    try:
        ledger.append({"amount": Decimal("1.50")})
    except CanonicalizationError as err:
        print(err)
"""
        expected = "data.amount: Decimal is not JSON-native; convert to str"
        assert _run(source, tmp_path).stdout.strip().endswith(expected)

    def test_the_empty_payload_example_names_validation_error(self, tmp_path: Path) -> None:
        source = """
from signledger import Ledger, SignLedgerError

with Ledger() as ledger:
    try:
        ledger.append({})
    except SignLedgerError as err:
        print(type(err).__name__)
"""
        assert _run(source, tmp_path).stdout.strip().endswith("ValidationError")

    def test_the_batch_example_commits_three_entries(self, tmp_path: Path) -> None:
        source = """
from signledger import Ledger
from signledger.core.batch import BatchItem, BatchProcessor

with Ledger() as ledger:
    result = BatchProcessor(ledger).append_batch([
        {"event": "one"},
        {"event": "two"},
        BatchItem(data={"event": "three"}, metadata={"source": "importer"}),
    ])

print(result.ok, result.committed)
"""
        assert _run(source, tmp_path).stdout.strip().endswith("True 3")


def _run(source: str, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(cwd),
        env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestDocumentationDoesNotPromiseWhatDoesNotExist:
    """The v1 README's real failure was naming things that were never built."""

    #: Names v1's README advertised. Every one of them was fiction.
    V1_FICTIONS = [
        "InMemoryBackend",
        "PostgreSQLBackend",
        "MongoDBBackend",
        "RedisBackend",
        "signledger.compression",
        "signledger.pool",
        "signledger.consensus",
        "signledger.distributed",
        "get_entries(",
        "get_latest(",
    ]

    @pytest.mark.parametrize("fiction", V1_FICTIONS)
    def test_the_readme_does_not_mention_a_v1_fiction(self, fiction: str) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert fiction not in text

    def test_every_import_in_the_readme_resolves(self) -> None:
        """A catch-all: any `from signledger... import X` in the docs must work.

        Broader than executing the blocks, because it also covers imports that appear in
        prose or in a block that is illustrative rather than runnable.
        """
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        pattern = re.compile(r"^\s*from\s+(signledger[\w.]*)\s+import\s+([^\n#]+)$", re.M)

        statements = []
        for module, names in pattern.findall(text):
            for name in names.split(","):
                cleaned = name.strip().split(" as ")[0].strip()
                if cleaned:
                    statements.append(f"from {module} import {cleaned}")

        assert statements, "no signledger imports found in the README"
        result = subprocess.run(
            [sys.executable, "-c", "\n".join(statements)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr

    def test_the_readme_does_not_claim_a_throughput_number(self) -> None:
        """v1 advertised "2,000+ entries/second" and "50-70% reduction". Both invented.

        A performance claim nothing measures is the same class of defect as an API
        nothing implements - and this project has already shipped one of each.
        """
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        invented = re.findall(r"[\d,]+\+?\s*(?:entries|transactions|TPS)\s*/?\s*sec", text, re.I)
        assert invented == []


class TestEveryShippedExampleRuns:
    """An example that does not run is worse than no example.

    v1 shipped `examples/` full of calls to `ledger.get_entries()`, `ledger.query()` and
    `ledger.verify_entry()` - methods that never existed on any version of `Ledger`. A
    reader copying from them got an `AttributeError`, and nothing in the project noticed
    because no test ever executed a single one.

    This walks the directory rather than naming files, so a *new* broken example is
    caught by the test that already exists.
    """

    EXAMPLES = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "examples").rglob("*.py")
    )

    def test_there_are_examples_to_run(self) -> None:
        """Guards the sweep below against passing on an empty directory."""
        assert len(self.EXAMPLES) >= 3

    @pytest.mark.parametrize("relative_path", EXAMPLES)
    def test_the_example_runs_to_completion(self, relative_path: str, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / relative_path)],
            cwd=str(tmp_path),
            env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"{relative_path} failed:\n{result.stderr[-2000:]}"

    @pytest.mark.parametrize("relative_path", EXAMPLES)
    def test_the_example_calls_no_method_the_ledger_does_not_have(
        self, relative_path: str
    ) -> None:
        """Static check, so a method missing on a branch never taken is still caught.

        Running an example only proves the lines that executed. `ledger.query(...)` inside
        an `if` nobody triggers is exactly how v1's examples looked healthy.
        """
        import ast

        from signledger import Ledger

        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ledger"
        }
        missing = sorted(name for name in called if not hasattr(Ledger, name))
        assert missing == []
