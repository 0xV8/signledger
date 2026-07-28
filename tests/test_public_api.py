"""The public API surface, as one integration-level contract.

Each wave built its own module and re-exported what it owned. Nothing checked that the
*assembled* package presents a coherent front door. v1.0.0's top-level ``__init__`` is the
cautionary tale: it advertised ``from signledger import Ledger, Entry`` against a module
that had been deleted, so ``import signledger`` - and therefore every ``import
signledger.anything`` - raised ``ModuleNotFoundError`` on a fresh interpreter.

These tests are deliberately about *names and versions*, the things unit tests never look
at and users hit first.
"""

import importlib
from typing import Any, List

import pytest

import signledger


class TestVersion:
    def test_the_package_reports_the_v2_rebuild_version(self) -> None:
        """v1.0.0's metadata was still being shipped by the rebuild."""
        assert signledger.__version__ == "2.0.0"

    def test_the_version_sorts_above_the_withdrawn_release(self) -> None:
        """The fix has to be reachable.

        1.0.0 is on PyPI and does not provide the tamper-evidence it documents.
        pip resolves to the highest version, so a rebuild numbered below 1.0.0
        would leave `pip install signledger` serving the broken release, and
        would never satisfy a `signledger>=1.0.0` constraint at all. The rebuild
        must sort strictly above it, and being a full API break, 2.0.0 is also
        what semver requires.
        """
        packaging_version = pytest.importorskip("packaging.version")
        assert packaging_version.Version(signledger.__version__) > packaging_version.Version(
            "1.0.0"
        )

    def test_the_version_is_pep440_parseable(self) -> None:
        """setuptools reads this attribute to name the wheel; a bad value fails the build."""
        packaging_version = pytest.importorskip("packaging.version")
        packaging_version.Version(signledger.__version__)

    def test_the_format_version_is_two(self) -> None:
        """The hash format this build writes. Section 12 versions it independently."""
        assert signledger.CURRENT_FORMAT_VERSION == 2


class TestTopLevelExports:
    """``from signledger import X`` for every X the package advertises."""

    def test_every_name_in_dunder_all_actually_exists(self) -> None:
        """An ``__all__`` entry with no attribute behind it is a broken promise.

        This is v1's exact defect shape, caught at the level it happened.
        """
        missing = [name for name in signledger.__all__ if not hasattr(signledger, name)]
        assert missing == []

    def test_dunder_all_has_no_duplicates(self) -> None:
        assert sorted(signledger.__all__) == sorted(set(signledger.__all__))

    @pytest.mark.parametrize(
        "name",
        [
            "Ledger",
            "Entry",
            "AssuranceLevel",
            "VerificationReport",
            "VerificationFailure",
            "FailureKind",
            "Checkpoint",
            "SignLedgerError",
            "IntegrityError",
            "ValidationError",
            "StorageError",
            "SignatureError",
            "CanonicalizationError",
            "ConfigurationError",
            "AnchorError",
        ],
    )
    def test_the_primary_name_is_importable_from_the_package_root(self, name: str) -> None:
        assert name in signledger.__all__
        assert getattr(signledger, name) is not None

    def test_the_package_imports_with_cryptography_unavailable(self) -> None:
        """``signatures`` is an optional extra; the base install must still work.

        ``cryptography`` is installed here, so the only honest way to test the base
        install is to make the import fail the way it would on a machine without it.
        An unguarded top-level import anywhere in the graph fails this immediately.
        """
        assert _runs_without("cryptography", "import signledger; signledger.Ledger")

    def test_a_ledger_still_works_with_cryptography_unavailable(self) -> None:
        """Importing is not enough - the unsigned path must actually run."""
        script = (
            "import signledger\n"
            "with signledger.Ledger() as ledger:\n"
            "    ledger.append({'event': 'no-crypto'})\n"
            "    assert ledger.check_integrity().ok\n"
        )
        assert _runs_without("cryptography", script)

    @pytest.mark.parametrize("driver", ["psycopg", "pymongo", "flask", "fastapi", "django"])
    def test_the_package_imports_without_the_optional_dependency(self, driver: str) -> None:
        assert _runs_without(driver, "import signledger; signledger.Ledger")


def _modules_loaded_by(statement: str) -> List[str]:
    """Names present in ``sys.modules`` after ``statement`` runs in a fresh interpreter.

    This is how laziness is distinguished from a guarded eager import: both survive the
    dependency being missing, but only the lazy one leaves ``sys.modules`` clean when it
    is present.
    """
    import json
    import subprocess
    import sys

    probe = f"{statement}\nimport json as _j, sys as _s\nprint(_j.dumps(sorted(_s.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _runs_without(blocked: str, statement: str) -> bool:
    """Run ``statement`` in a fresh interpreter where importing ``blocked`` fails.

    A ``sys.meta_path`` finder that refuses the name is a faithful stand-in for the
    package simply not being installed, and unlike uninstalling it, it is reversible and
    scoped to one subprocess.
    """
    import subprocess
    import sys

    probe = f"""
import sys

class _Blocker:
    def find_module(self, name, path=None):
        return None
    def find_spec(self, name, path=None, target=None):
        if name == {blocked!r} or name.startswith({blocked!r} + "."):
            raise ImportError("blocked for test: " + name)
        return None

sys.meta_path.insert(0, _Blocker())
for name in list(sys.modules):
    if name == {blocked!r} or name.startswith({blocked!r} + "."):
        del sys.modules[name]

{statement}
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return result.stdout.strip().endswith("OK")


class TestSubpackageExports:
    """Each subpackage's ``__init__`` must present its own wave's surface."""

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("signledger.core", "Ledger"),
            ("signledger.core", "Entry"),
            ("signledger.core", "VerificationReport"),
            ("signledger.core", "VerificationFailure"),
            ("signledger.core", "AssuranceLevel"),
            ("signledger.core", "FailureKind"),
            ("signledger.core", "verify_chain"),
            ("signledger.core", "BatchProcessor"),
            ("signledger.crypto", "canonicalize"),
            ("signledger.crypto", "entry_preimage"),
            ("signledger.crypto", "HashChain"),
            ("signledger.crypto", "genesis_hash"),
            ("signledger.crypto", "MerkleTree"),
            ("signledger.crypto", "verify_inclusion"),
            ("signledger.anchor", "Checkpoint"),
            ("signledger.anchor", "FileAnchorSink"),
            ("signledger.anchor", "NullAnchorSink"),
            ("signledger.backends", "MemoryBackend"),
            ("signledger.backends", "SQLiteBackend"),
            ("signledger.backends", "StorageBackend"),
        ],
    )
    def test_the_subpackage_exports_the_name(self, module: str, name: str) -> None:
        imported = importlib.import_module(module)
        assert name in imported.__all__
        assert hasattr(imported, name)

    @pytest.mark.parametrize(
        "module",
        [
            "signledger.core",
            "signledger.crypto",
            "signledger.anchor",
            "signledger.backends",
            "signledger.contrib",
        ],
    )
    def test_every_name_the_subpackage_advertises_exists(self, module: str) -> None:
        imported = importlib.import_module(module)
        missing = [name for name in imported.__all__ if not hasattr(imported, name)]
        assert missing == []

    def test_the_crypto_package_still_imports_without_cryptography(self) -> None:
        """``signledger.crypto`` must re-export the signature surface *lazily*.

        Eagerly importing ``signatures`` here would drag ``cryptography`` into the back
        door of every base install.

        Blocking the import would *not* catch this: ``signatures`` guards its own import
        and stays importable without ``cryptography``. Only checking what actually landed
        in ``sys.modules`` distinguishes a lazy re-export from an eager one.
        """
        assert "cryptography" not in _modules_loaded_by("import signledger.crypto")

    def test_the_lazily_exported_signature_names_do_resolve(self) -> None:
        """Guards the test above: laziness that never resolves is just a missing export."""
        import signledger.crypto as crypto

        assert crypto.Ed25519Signer is not None
        assert crypto.verify is not None

    def test_an_unknown_attribute_on_the_crypto_package_still_raises(self) -> None:
        """A module ``__getattr__`` that returns something for every name hides typos."""
        import signledger.crypto as crypto

        with pytest.raises(AttributeError):
            _ = crypto.NoSuchSigner

    def test_an_unknown_attribute_on_the_core_package_still_raises(self) -> None:
        import signledger.core as core

        with pytest.raises(AttributeError):
            _ = core.NoSuchLedger

    def test_the_unknown_attribute_error_names_the_attribute(self) -> None:
        """An ``AttributeError`` that does not say what was missing is a worse typo hunt."""
        import signledger.core as core

        with pytest.raises(AttributeError, match="NoSuchLedger"):
            _ = core.NoSuchLedger

    @pytest.mark.parametrize("module", ["signledger.core", "signledger.crypto"])
    def test_the_lazy_names_are_discoverable_by_dir(self, module: str) -> None:
        """Without ``__dir__`` the lazy exports are invisible to tab completion and help().

        A name that works but cannot be discovered is a name most users never find.
        """
        imported = importlib.import_module(module)
        listed = dir(imported)
        assert set(imported.__all__) <= set(listed)

    @pytest.mark.parametrize("module", ["signledger.core", "signledger.crypto"])
    def test_dir_is_sorted_and_free_of_duplicates(self, module: str) -> None:
        listed = dir(importlib.import_module(module))
        assert listed == sorted(set(listed))

    def test_a_lazy_name_is_cached_after_first_access(self) -> None:
        """The second lookup must not re-enter ``importlib`` on every attribute read."""
        import signledger.core as core

        first = core.Ledger
        assert core.__dict__["Ledger"] is first


class TestPrimaryWorkflow:
    """The three lines a README opens with must work from the names above alone."""

    def test_append_and_verify_through_only_top_level_names(self) -> None:
        from signledger import Ledger

        with Ledger() as ledger:
            entry = ledger.append({"event": "front-door"})
            report = ledger.check_integrity()

        assert entry.sequence == 0
        assert report.ok is True
        assert report.entries_verified == 1

    def test_the_assurance_level_of_a_plain_ledger_is_chain_only(self) -> None:
        """Contract section 1: the guarantee is reported, never asserted in prose."""
        from signledger import AssuranceLevel, Ledger

        with Ledger() as ledger:
            ledger.append({"event": "front-door"})
            assert ledger.assurance_level() is AssuranceLevel.CHAIN_ONLY

    def test_every_shipped_exception_descends_from_the_root_error(self) -> None:
        """One ``except SignLedgerError`` must be enough to contain the library."""
        from signledger import SignLedgerError

        for name in signledger.__all__:
            attribute: Any = getattr(signledger, name)
            if isinstance(attribute, type) and issubclass(attribute, BaseException):
                assert issubclass(attribute, SignLedgerError), name
