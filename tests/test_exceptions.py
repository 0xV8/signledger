"""Tests for the SignLedger exception hierarchy (contract section 10)."""

import inspect

import pytest

from signledger.core import exceptions as exc_module
from signledger.core.exceptions import (
    AnchorError,
    CanonicalizationError,
    ConfigurationError,
    IntegrityError,
    SignatureError,
    SignLedgerError,
    StorageError,
    ValidationError,
)

SUBCLASSES = [
    IntegrityError,
    ValidationError,
    StorageError,
    SignatureError,
    CanonicalizationError,
    ConfigurationError,
    AnchorError,
]


def test_signledger_error_is_an_exception_subclass():
    assert issubclass(SignLedgerError, Exception)


def test_signledger_error_carries_its_message():
    with pytest.raises(SignLedgerError, match="boom"):
        raise SignLedgerError("boom")


@pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
def test_error_derives_from_signledger_error(cls):
    assert issubclass(cls, SignLedgerError)


@pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
def test_error_is_catchable_as_signledger_error(cls):
    with pytest.raises(SignLedgerError):
        raise cls("x")


@pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
def test_error_is_a_distinct_type(cls):
    """No subclass may alias another - `except StorageError` must not swallow IntegrityError."""
    others = [o for o in SUBCLASSES if o is not cls]
    for other in others:
        assert not issubclass(cls, other)


def test_every_public_error_is_exported_in_dunder_all():
    exported = set(exc_module.__all__)
    defined = {
        name
        for name, obj in vars(exc_module).items()
        if inspect.isclass(obj) and issubclass(obj, SignLedgerError)
    }
    assert defined == exported


def test_module_defines_no_error_outside_the_signledger_root():
    """Every exception in the module must funnel through SignLedgerError."""
    stray = [
        name
        for name, obj in vars(exc_module).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and not issubclass(obj, SignLedgerError)
    ]
    assert stray == []
