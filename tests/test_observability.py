"""Tests for ``signledger.observability``."""

import contextlib
import json
import logging
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from signledger.anchor.checkpoint import Checkpoint
from signledger.backends.memory import MemoryBackend
from signledger.core.exceptions import ValidationError
from signledger.core.ledger import Ledger
from signledger.observability import (
    APPEND_COUNTER,
    APPEND_SPAN,
    FAILURE_COUNTER,
    VERIFY_COUNTER,
    VERIFY_SPAN,
    LedgerHealth,
    LedgerObserver,
    ledger_health,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_health_of_a_ledger_with_one_entry_is_ok() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    report = ledger.check_integrity()

    health = LedgerHealth.from_report(report, ledger_id=ledger.ledger_id)

    assert health.ok is True


def test_health_reports_the_assurance_level_as_a_plain_string() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.assurance == "chain_only"


def test_health_names_the_kind_of_the_first_failure() -> None:
    # An empty ledger is never evidence of integrity (contract section 5 rule 1).
    ledger = Ledger(MemoryBackend())

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.first_failure_kind == "count_mismatch"


def test_health_serializes_to_json_without_a_custom_encoder() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    restored = json.loads(json.dumps(health.to_dict()))

    assert restored["ledger_id"] == ledger.ledger_id


def test_health_reports_how_many_entries_were_verified() -> None:
    ledger = Ledger(MemoryBackend())
    for index in range(3):
        ledger.append({"event": "created", "index": index})

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.entries_verified == 3


def test_health_counts_every_failure_not_only_the_first() -> None:
    backend = MemoryBackend()
    ledger = Ledger(backend)
    ledger.append({"event": "created"})
    # Two independent problems: the checkpoint no longer matches, and it counted more
    # entries than the store now holds.
    backend.save_checkpoint(
        Checkpoint(
            ledger_id=ledger.ledger_id,
            sequence=0,
            head_hash="00" * 32,
            entry_count=9,
            merkle_root="11" * 32,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    )

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.failure_count == 2


def test_health_records_the_backend_capabilities_it_was_given() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})

    health = LedgerHealth.from_report(
        ledger.check_integrity(),
        ledger_id=ledger.ledger_id,
        capabilities=ledger.backend.capabilities(),
    )

    assert health.capabilities == ("atomic_append", "unique_chain_link")


def test_health_capabilities_default_to_nothing_claimed() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.capabilities == ()


def test_health_records_a_timezone_aware_instant() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert health.checked_at.tzinfo is not None


def test_health_serializes_the_instant_as_an_iso_string() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    restored = json.loads(json.dumps(health.to_dict()))

    assert datetime.fromisoformat(restored["checked_at"]) == health.checked_at


def test_health_carries_no_entry_payload_anywhere() -> None:
    # Contract section 10: nothing that leaves the library may quote user data.
    ledger = Ledger(MemoryBackend())
    ledger.append({"card_number": "4111111111111111"}, {"customer": "acme-corp"})

    health = LedgerHealth.from_report(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    rendered = json.dumps(health.to_dict())
    assert "4111111111111111" not in rendered
    assert "acme-corp" not in rendered


def test_ledger_health_measures_a_live_ledger_including_its_capabilities() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})

    health = ledger_health(ledger)

    assert health.ok is True
    assert health.ledger_id == ledger.ledger_id
    assert health.capabilities == ("atomic_append", "unique_chain_link")


# ======================================================================================
# LedgerObserver - counters
# ======================================================================================


def test_an_observer_starts_with_every_counter_at_zero() -> None:
    observer = LedgerObserver()

    assert observer.counters() == {
        APPEND_COUNTER: 0,
        VERIFY_COUNTER: 0,
        FAILURE_COUNTER: 0,
    }


def test_recording_an_append_increments_the_append_counter() -> None:
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"event": "created"})
    observer = LedgerObserver()

    observer.record_append(entry)

    assert observer.counters()[APPEND_COUNTER] == 1


def test_recording_a_verification_increments_the_verification_counter() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    observer = LedgerObserver()

    observer.record_verify(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert observer.counters()[VERIFY_COUNTER] == 1


def test_a_passing_verification_does_not_increment_the_failure_counter() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    observer = LedgerObserver()

    observer.record_verify(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert observer.counters()[FAILURE_COUNTER] == 0


def test_a_failing_verification_counts_every_failure_not_just_the_report() -> None:
    backend = MemoryBackend()
    ledger = Ledger(backend)
    ledger.append({"event": "created"})
    backend.save_checkpoint(
        Checkpoint(
            ledger_id=ledger.ledger_id,
            sequence=0,
            head_hash="00" * 32,
            entry_count=9,
            merkle_root="11" * 32,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    )
    observer = LedgerObserver()

    observer.record_verify(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert observer.counters()[FAILURE_COUNTER] == 2


def test_counters_hands_back_a_copy_so_a_caller_cannot_rewrite_history() -> None:
    observer = LedgerObserver()

    observer.counters()[APPEND_COUNTER] = 10_000

    assert observer.counters()[APPEND_COUNTER] == 0


def test_recording_an_integrity_failure_increments_the_failure_counter() -> None:
    ledger = Ledger(MemoryBackend())
    failure = ledger.check_integrity().failures[0]
    observer = LedgerObserver()

    observer.record_integrity_failure(failure, ledger_id=ledger.ledger_id)

    assert observer.counters()[FAILURE_COUNTER] == 1


# ======================================================================================
# LedgerObserver - structured logs
# ======================================================================================


def test_an_append_is_logged_with_the_ledger_id_sequence_and_entry_id(
    caplog: Any,
) -> None:
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"event": "created"})
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_append(entry)

    record = caplog.records[-1]
    assert (record.ledger_id, record.sequence, record.entry_id) == (
        ledger.ledger_id,
        0,
        entry.entry_id,
    )


def test_an_append_log_record_never_carries_the_entry_payload(caplog: Any) -> None:
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"card_number": "4111111111111111"}, {"customer": "acme-corp"})
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_append(entry)

    rendered = repr(caplog.records[-1].__dict__) + caplog.records[-1].getMessage()
    assert "4111111111111111" not in rendered
    assert "acme-corp" not in rendered


def test_a_failing_verification_is_logged_at_error_level(caplog: Any) -> None:
    ledger = Ledger(MemoryBackend())
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_verify(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert caplog.records[-1].levelno == logging.ERROR


def test_a_passing_verification_is_logged_at_info_level(caplog: Any) -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "created"})
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_verify(ledger.check_integrity(), ledger_id=ledger.ledger_id)

    assert caplog.records[-1].levelno == logging.INFO


def test_an_integrity_failure_log_names_the_kind_of_damage(caplog: Any) -> None:
    ledger = Ledger(MemoryBackend())
    failure = ledger.check_integrity().failures[0]
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_integrity_failure(failure, ledger_id=ledger.ledger_id)

    assert caplog.records[-1].failure_kind == "count_mismatch"


def test_a_record_event_field_outside_the_allowlist_is_refused() -> None:
    observer = LedgerObserver()

    with pytest.raises(ValidationError, match="data"):
        observer.record_event("custom", data={"card_number": "4111111111111111"})


def test_a_record_event_field_inside_the_allowlist_is_logged(caplog: Any) -> None:
    observer = LedgerObserver()

    with caplog.at_level(logging.INFO, logger="signledger"):
        observer.record_event("custom", ledger_id="abc", sequence=7)

    assert (caplog.records[-1].ledger_id, caplog.records[-1].sequence) == ("abc", 7)


def test_the_logger_can_be_supplied_by_the_caller(caplog: Any) -> None:
    observer = LedgerObserver(logger=logging.getLogger("myapp.audit"))

    with caplog.at_level(logging.INFO, logger="myapp.audit"):
        observer.record_event("custom", ledger_id="abc")

    assert caplog.records[-1].name == "myapp.audit"


# ======================================================================================
# OpenTelemetry doubles
#
# These stand in for the *external* OpenTelemetry API, never for anything in signledger.
# They are what makes both halves of the degradation story - present and absent -
# deterministic on any machine, whether or not the SDK happens to be installed.
# ======================================================================================


class StubSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict = {}
        self.exceptions: list = []
        self.ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, err: BaseException) -> None:
        self.exceptions.append(err)


class StubTracer:
    def __init__(self) -> None:
        self.spans: list = []

    @contextlib.contextmanager
    def start_as_current_span(self, name: str) -> Any:
        span = StubSpan(name)
        self.spans.append(span)
        try:
            yield span
        finally:
            span.ended = True


class StubCounter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.adds: list = []

    def add(self, amount: int, attributes: Any = None) -> None:
        self.adds.append((amount, attributes))


class StubMeter:
    def __init__(self) -> None:
        self.counters: dict = {}

    def create_counter(self, name: str, unit: str = "", description: str = "") -> StubCounter:
        counter = StubCounter(name)
        self.counters[name] = counter
        return counter


@contextlib.contextmanager
def stub_opentelemetry(tracer: Any = None, meter: Any = None) -> Any:
    """Make ``from opentelemetry import metrics, trace`` succeed against the stubs."""
    module = types.ModuleType("opentelemetry")
    module.trace = types.SimpleNamespace(get_tracer=lambda name: tracer)  # type: ignore[attr-defined]
    module.metrics = types.SimpleNamespace(get_meter=lambda name: meter)  # type: ignore[attr-defined]
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "opentelemetry" or name.startswith("opentelemetry.")
    }
    for name in saved:
        del sys.modules[name]
    sys.modules["opentelemetry"] = module
    try:
        yield
    finally:
        del sys.modules["opentelemetry"]
        sys.modules.update(saved)


class _Blocker:
    """An import hook that refuses one package, however it is installed."""

    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ImportError(f"{fullname} is blocked for this test")
        return


@contextlib.contextmanager
def no_opentelemetry() -> Any:
    """Guarantee ``import opentelemetry`` fails, even where the SDK is installed."""
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "opentelemetry" or name.startswith("opentelemetry.")
    }
    for name in saved:
        del sys.modules[name]
    blocker = _Blocker("opentelemetry")
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


# ======================================================================================
# LedgerObserver - spans and OpenTelemetry
# ======================================================================================


def test_an_observer_without_opentelemetry_reports_it_as_disabled() -> None:
    with no_opentelemetry():
        observer = LedgerObserver()

    assert observer.otel_enabled is False


def test_counters_keep_working_without_opentelemetry() -> None:
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"event": "created"})
    with no_opentelemetry():
        observer = LedgerObserver()

    observer.record_append(entry)

    assert observer.counters()[APPEND_COUNTER] == 1


def test_a_span_without_a_tracer_still_runs_its_body() -> None:
    with no_opentelemetry():
        observer = LedgerObserver()

    ran = False
    with observer.span(APPEND_SPAN) as span:
        ran = True

    assert (ran, span) == (True, None)


def test_an_exception_without_a_tracer_is_not_swallowed() -> None:
    with no_opentelemetry():
        observer = LedgerObserver()

    with pytest.raises(ZeroDivisionError), observer.span(APPEND_SPAN):
        raise ZeroDivisionError("boom")


def test_an_observer_finds_the_opentelemetry_tracer_when_it_is_installed() -> None:
    tracer = StubTracer()

    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    assert observer.otel_enabled is True


def test_a_span_is_opened_on_the_resolved_tracer() -> None:
    tracer = StubTracer()
    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    with observer.span(APPEND_SPAN):
        pass

    assert [span.name for span in tracer.spans] == [APPEND_SPAN]


def test_a_span_is_ended_even_when_its_body_raises() -> None:
    tracer = StubTracer()
    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    with pytest.raises(ZeroDivisionError), observer.span(APPEND_SPAN):
        raise ZeroDivisionError("boom")

    assert tracer.spans[0].ended is True


def test_a_span_records_the_exception_that_ended_it() -> None:
    tracer = StubTracer()
    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    with pytest.raises(ZeroDivisionError), observer.span(APPEND_SPAN):
        raise ZeroDivisionError("boom")

    assert isinstance(tracer.spans[0].exceptions[0], ZeroDivisionError)


def test_a_span_carries_the_attributes_it_was_given() -> None:
    tracer = StubTracer()
    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    with observer.span(APPEND_SPAN, ledger_id="abc", sequence=4):
        pass

    assert tracer.spans[0].attributes == {"ledger_id": "abc", "sequence": 4}


def test_a_span_attribute_outside_the_allowlist_is_refused() -> None:
    # A span is fanned out to a tracing backend just like a log line is; the payload rule
    # applies to it identically.
    tracer = StubTracer()
    with stub_opentelemetry(tracer=tracer, meter=StubMeter()):
        observer = LedgerObserver()

    with pytest.raises(ValidationError, match="metadata"), observer.span(
        APPEND_SPAN, metadata={"customer": "acme-corp"}
    ):
        pass


def test_an_append_is_added_to_the_opentelemetry_counter_too() -> None:
    meter = StubMeter()
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"event": "created"})
    with stub_opentelemetry(tracer=StubTracer(), meter=meter):
        observer = LedgerObserver()

    observer.record_append(entry)

    assert meter.counters[APPEND_COUNTER].adds == [(1, {})]


def test_every_counter_is_created_on_the_resolved_meter() -> None:
    meter = StubMeter()

    with stub_opentelemetry(tracer=StubTracer(), meter=meter):
        LedgerObserver()

    assert sorted(meter.counters) == sorted([APPEND_COUNTER, FAILURE_COUNTER, VERIFY_COUNTER])


def test_the_failure_counter_is_attributed_with_the_kind_of_damage() -> None:
    meter = StubMeter()
    ledger = Ledger(MemoryBackend())
    failure = ledger.check_integrity().failures[0]
    with stub_opentelemetry(tracer=StubTracer(), meter=meter):
        observer = LedgerObserver()

    observer.record_integrity_failure(failure, ledger_id=ledger.ledger_id)

    assert meter.counters[FAILURE_COUNTER].adds == [(1, {"failure_kind": "count_mismatch"})]


def test_a_caller_supplied_tracer_is_used_without_importing_opentelemetry() -> None:
    tracer = StubTracer()
    with no_opentelemetry():
        observer = LedgerObserver(tracer=tracer)

        with observer.span(VERIFY_SPAN):
            pass

    assert [span.name for span in tracer.spans] == [VERIFY_SPAN]


def test_a_caller_supplied_meter_is_used_without_importing_opentelemetry() -> None:
    meter = StubMeter()
    ledger = Ledger(MemoryBackend())
    entry = ledger.append({"event": "created"})
    with no_opentelemetry():
        observer = LedgerObserver(meter=meter)

    observer.record_append(entry)

    assert meter.counters[APPEND_COUNTER].adds == [(1, {})]


def test_the_service_name_is_passed_to_the_tracer_factory() -> None:
    seen: list = []
    module = types.ModuleType("opentelemetry")
    module.trace = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_tracer=lambda name: seen.append(name) or StubTracer()
    )
    module.metrics = types.SimpleNamespace(get_meter=lambda name: StubMeter())  # type: ignore[attr-defined]
    with no_opentelemetry():
        sys.modules["opentelemetry"] = module
        try:
            LedgerObserver(service_name="billing-audit")
        finally:
            del sys.modules["opentelemetry"]

    assert seen == ["billing-audit"]


def test_the_module_imports_in_a_fresh_interpreter_without_opentelemetry() -> None:
    # The real proof that telemetry is optional: a subprocess where the package cannot be
    # imported at all still gets a working observability module.
    program = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'opentelemetry':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from signledger.observability import LedgerObserver, APPEND_COUNTER\n"
        "o = LedgerObserver()\n"
        "print(o.otel_enabled, o.counters()[APPEND_COUNTER])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.stdout.strip() == "False 0", result.stderr
