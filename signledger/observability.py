"""Health reporting and telemetry hooks.

Two things live here:

* :class:`LedgerHealth` - a small, JSON-native statement about one ledger, suitable for a
  health endpoint or a dashboard.
* :class:`LedgerObserver` - OpenTelemetry-style counters, spans and structured log
  records, which **degrade to plain in-process counters** when ``opentelemetry`` is not
  installed. Telemetry is never a hard dependency of an audit log.

Every field that leaves this module is a measurement or a structural label - a ledger id,
a sequence number, an entry id, a failure kind. Entry ``data`` and ``metadata`` are never
included, in a log line, in a span attribute or in a metric attribute (contract section
10). Telemetry is the most widely fanned-out output a library has; a payload that reaches
it has effectively been published.
"""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Iterator, Optional, Tuple

from signledger.core.exceptions import ValidationError

__all__ = [
    "APPEND_COUNTER",
    "APPEND_SPAN",
    "FAILURE_COUNTER",
    "LOG_FIELDS",
    "VERIFY_COUNTER",
    "VERIFY_SPAN",
    "LedgerHealth",
    "LedgerObserver",
    "ledger_health",
]

#: Metric names. Dotted and namespaced, as OpenTelemetry expects.
APPEND_COUNTER = "signledger.entries.appended"
VERIFY_COUNTER = "signledger.verifications"
FAILURE_COUNTER = "signledger.integrity.failures"

#: Span names.
APPEND_SPAN = "signledger.append"
VERIFY_SPAN = "signledger.verify"

#: The only field names that may appear on a log record, a span or a metric attribute.
#: An allowlist rather than a denylist: a new field has to be *added* here deliberately,
#: so no future caller can leak a payload into telemetry by accident.
LOG_FIELDS = frozenset(
    {
        "assurance",
        "entries_verified",
        "entry_id",
        "event",
        "failure_count",
        "failure_kind",
        "ledger_id",
        "ok",
        "sequence",
    }
)

#: Where records go when the caller does not supply a logger.
DEFAULT_LOGGER_NAME = "signledger"


def _utcnow() -> datetime:
    """The current instant, always timezone-aware."""
    return datetime.now(timezone.utc)


def ledger_health(ledger: Any) -> "LedgerHealth":
    """Run a real verification against ``ledger`` and describe what it found."""
    return LedgerHealth.from_report(
        ledger.check_integrity(),
        ledger_id=ledger.ledger_id,
        capabilities=ledger.backend.capabilities(),
    )


@dataclass(frozen=True)
class LedgerHealth:
    """A point-in-time statement about one ledger.

    Every field is a measurement or a structural label. Nothing here is ever taken from
    ``data`` or ``metadata``: a health record is the thing most likely to be shipped to a
    dashboard, a pager or a log aggregator, which is exactly where entry payloads must
    not go (contract section 10).
    """

    ledger_id: str
    ok: bool
    assurance: str
    entries_verified: int = 0
    failure_count: int = 0
    first_failure_kind: Optional[str] = None
    capabilities: Tuple[str, ...] = ()
    checked_at: datetime = field(default_factory=_utcnow)

    @classmethod
    def from_report(
        cls,
        report: Any,
        *,
        ledger_id: str,
        capabilities: FrozenSet[str] = frozenset(),
    ) -> "LedgerHealth":
        """Summarise a :class:`~signledger.core.verification.VerificationReport`."""
        return cls(
            ledger_id=ledger_id,
            ok=report.ok,
            # ``.value``, never ``str(...)``: ``AssuranceLevel`` is a ``(str, Enum)`` and
            # ``__format__``/``__str__`` on those changed between 3.11 and 3.12.
            assurance=report.assurance.value,
            entries_verified=report.entries_verified,
            failure_count=len(report.failures),
            first_failure_kind=report.failures[0].kind.value if report.failures else None,
            # Sorted so two runs of the same deployment produce identical output; a
            # frozenset's iteration order is not stable across processes.
            capabilities=tuple(sorted(capabilities)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-native mapping. No custom encoder, no payload."""
        return {
            "ledger_id": self.ledger_id,
            "ok": self.ok,
            "assurance": self.assurance,
            "entries_verified": self.entries_verified,
            "failure_count": self.failure_count,
            "first_failure_kind": self.first_failure_kind,
            "capabilities": list(self.capabilities),
            "checked_at": self.checked_at.isoformat(),
        }


class LedgerObserver:
    """Counters, spans and structured logs for a ledger.

    The in-process counters are always live. OpenTelemetry instruments, when they can be
    resolved, are updated *in addition* - never instead - so ``counters()`` answers the
    same thing on a machine with the SDK and on one without it.
    """

    def __init__(
        self,
        *,
        service_name: str = DEFAULT_LOGGER_NAME,
        logger: Optional[logging.Logger] = None,
        tracer: Any = None,
        meter: Any = None,
    ) -> None:
        self.service_name = service_name
        self._logger = logging.getLogger(DEFAULT_LOGGER_NAME) if logger is None else logger
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {
            APPEND_COUNTER: 0,
            VERIFY_COUNTER: 0,
            FAILURE_COUNTER: 0,
        }

        # An injected tracer or meter is used as-is, so a caller can wire up telemetry
        # without this module importing anything at all.
        trace_api, metrics_api = _otel_apis()
        if tracer is None and trace_api is not None:
            tracer = trace_api.get_tracer(service_name)
        if meter is None and metrics_api is not None:
            meter = metrics_api.get_meter(service_name)
        self._tracer = tracer
        self._instruments: Dict[str, Any] = _build_instruments(meter)

    @property
    def otel_enabled(self) -> bool:
        """Whether any OpenTelemetry instrument was resolved.

        ``False`` is a normal, fully-working state: the counters below are maintained
        either way. An audit log that stopped working because a telemetry package was
        missing would be trading the guarantee for the dashboard.
        """
        return self._tracer is not None or bool(self._instruments)

    # -- counters ----------------------------------------------------------------------

    def counters(self) -> Dict[str, int]:
        """A snapshot of every counter. A copy, so a reader cannot rewrite history."""
        with self._lock:
            return dict(self._counts)

    def _bump(self, name: str, **attributes: Any) -> None:
        checked = _checked_fields(attributes)
        with self._lock:
            self._counts[name] += 1
        instrument = self._instruments.get(name)
        if instrument is not None:
            instrument.add(1, checked)

    # -- spans -------------------------------------------------------------------------

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Run a block inside a span, or transparently without one.

        Attributes go through the same allowlist as log fields: a span is fanned out to a
        tracing backend exactly like a log line is, so the payload rule applies to it
        identically.
        """
        checked = _checked_fields(attributes)
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as active:
            for key in sorted(checked):
                active.set_attribute(key, checked[key])
            try:
                yield active
            except BaseException as err:
                # Recorded, then re-raised untouched. Telemetry observes failures; it
                # never absorbs them.
                active.record_exception(err)
                raise

    # -- structured logging ------------------------------------------------------------

    def record_event(self, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
        """Emit one structured log record.

        ``fields`` are checked against :data:`LOG_FIELDS` before anything is emitted, so a
        payload cannot reach a log aggregator through this method.
        """
        extra = _checked_fields(fields)
        extra["event"] = event
        self._logger.log(level, event, extra=extra)

    # -- hooks -------------------------------------------------------------------------

    def record_append(self, entry: Any) -> None:
        """Count and log one appended entry."""
        self._bump(APPEND_COUNTER)
        self.record_event(
            APPEND_SPAN,
            ledger_id=entry.ledger_id,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
        )

    def record_integrity_failure(self, failure: Any, *, ledger_id: str) -> None:
        """Count and log one piece of damage found by verification."""
        # ``failure_kind`` is a bounded set, so it is safe as a metric dimension; a
        # ledger id is not, and is deliberately left to the log record.
        self._bump(FAILURE_COUNTER, failure_kind=failure.kind.value)
        self.record_event(
            "signledger.integrity_failure",
            level=logging.ERROR,
            ledger_id=ledger_id,
            sequence=failure.sequence,
            entry_id=failure.entry_id,
            failure_kind=failure.kind.value,
        )

    def record_verify(self, report: Any, *, ledger_id: str) -> None:
        """Count and log one verification run, and every failure it reported."""
        self._bump(VERIFY_COUNTER)
        for failure in report.failures:
            self.record_integrity_failure(failure, ledger_id=ledger_id)
        # The summary goes last so a reader sees the detail before the verdict.
        self.record_event(
            VERIFY_SPAN,
            level=logging.INFO if report.ok else logging.ERROR,
            ledger_id=ledger_id,
            ok=report.ok,
            entries_verified=report.entries_verified,
            failure_count=len(report.failures),
            assurance=report.assurance.value,
        )


def _otel_apis() -> Tuple[Any, Any]:
    """The OpenTelemetry trace and metrics APIs, or ``(None, None)`` when absent.

    Imported inside the function on purpose: a module-level import would make the
    availability of telemetry a property of interpreter start-up rather than something a
    deployment can be tested for.
    """
    try:
        from opentelemetry import metrics, trace
    except ImportError:
        return None, None
    return trace, metrics


def _build_instruments(meter: Any) -> Dict[str, Any]:
    """Create one counter per metric on ``meter``, or nothing at all without one."""
    if meter is None:
        return {}
    return {
        APPEND_COUNTER: meter.create_counter(
            APPEND_COUNTER, unit="1", description="Entries appended to a ledger"
        ),
        VERIFY_COUNTER: meter.create_counter(
            VERIFY_COUNTER, unit="1", description="Chain verifications run"
        ),
        FAILURE_COUNTER: meter.create_counter(
            FAILURE_COUNTER, unit="1", description="Integrity failures reported"
        ),
    }


def _checked_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse any field outside :data:`LOG_FIELDS`, naming the offenders."""
    unknown = sorted(set(fields) - LOG_FIELDS)
    if unknown:
        raise ValidationError(
            "telemetry fields must come from the allowlist; refused: " + ", ".join(unknown)
        )
    return dict(fields)
