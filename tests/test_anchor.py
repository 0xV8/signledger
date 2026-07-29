"""Checkpoints and external anchoring (architecture contract, section 8).

An anchor is the second of the two things that make a log tamper-*evident* rather than
merely hash-chained: a chain head published somewhere the adversary does not control.
Without it, an insider who can edit rows can also re-derive every digest after the one
they changed, and a bare chain will happily report itself intact.
"""

import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

import pytest

from signledger.anchor.checkpoint import Checkpoint
from signledger.anchor.sinks import (
    AnchorReceipt,
    AnchorSink,
    FileAnchorSink,
    NullAnchorSink,
    checkpoint_signing_payload,
)
from signledger.backends.memory import MemoryBackend
from signledger.core.exceptions import AnchorError
from signledger.core.ledger import Ledger
from signledger.core.verification import AssuranceLevel, FailureKind

MOMENT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_checkpoint(sequence=3, ledger_id="ledger-a", **overrides):
    fields = {
        "ledger_id": ledger_id,
        "sequence": sequence,
        "head_hash": f"{sequence:064x}",
        "entry_count": sequence + 1,
        "merkle_root": f"{sequence:064X}",
        "timestamp": MOMENT + timedelta(minutes=sequence),
    }
    fields.update(overrides)
    return Checkpoint(**fields)


@pytest.fixture
def anchor_path(tmp_path):
    return str(tmp_path / "anchors.jsonl")


@pytest.fixture
def sink(anchor_path):
    return FileAnchorSink(anchor_path)


# --------------------------------------------------------------------------------------
# the null sink
# --------------------------------------------------------------------------------------


def test_the_null_sink_names_itself():
    assert NullAnchorSink().name == "null"


def test_the_null_sink_returns_a_receipt_for_the_checkpoint_it_was_given():
    receipt = NullAnchorSink().publish(make_checkpoint(7))

    assert (receipt.sink, receipt.ledger_id, receipt.sequence) == ("null", "ledger-a", 7)


def test_the_null_sink_never_hands_a_checkpoint_back():
    """It discards, so it can never confer ANCHORED assurance - which is the honest answer
    for a sink that stores nothing outside the process."""
    null = NullAnchorSink()
    null.publish(make_checkpoint(7))

    assert null.fetch_latest("ledger-a") is None


# --------------------------------------------------------------------------------------
# the file sink
# --------------------------------------------------------------------------------------


def test_the_file_sink_names_itself(sink):
    assert sink.name == "file"


def test_a_published_checkpoint_comes_back_field_for_field(sink):
    published = make_checkpoint(4, signature="c2ln", key_id="kid-1")
    sink.publish(published)

    assert sink.fetch_latest("ledger-a") == published


def test_an_unsigned_checkpoint_comes_back_unsigned(sink):
    sink.publish(make_checkpoint(2))
    fetched = sink.fetch_latest("ledger-a")

    assert (fetched.signature, fetched.key_id) == (None, None)


def test_fetching_from_a_sink_that_has_published_nothing_is_none(sink):
    assert sink.fetch_latest("ledger-a") is None


def test_fetching_a_ledger_the_sink_has_never_seen_is_none(sink):
    sink.publish(make_checkpoint(1, ledger_id="ledger-a"))

    assert sink.fetch_latest("ledger-b") is None


def test_the_newest_checkpoint_wins(sink):
    for sequence in (2, 9, 5):
        sink.publish(make_checkpoint(sequence))

    assert sink.fetch_latest("ledger-a").sequence == 9


def test_checkpoints_for_other_ledgers_are_ignored(sink):
    sink.publish(make_checkpoint(3, ledger_id="ledger-a"))
    sink.publish(make_checkpoint(99, ledger_id="ledger-b"))

    assert sink.fetch_latest("ledger-a").sequence == 3


def test_publishing_appends_rather_than_replacing(sink, anchor_path):
    """An anchor file that can be rewritten is not an anchor."""
    for sequence in (1, 2, 3):
        sink.publish(make_checkpoint(sequence))

    with open(anchor_path, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line]

    assert len(lines) == 3


def test_each_record_is_one_self_describing_json_object(sink, anchor_path):
    """A second implementation must be able to read the anchor without this library."""
    sink.publish(make_checkpoint(4, signature="c2ln", key_id="kid-1"))

    with open(anchor_path, encoding="utf-8") as handle:
        record = json.loads(handle.read().strip())

    assert record["head_hash"] == f"{4:064x}"


def test_a_receipt_records_where_the_checkpoint_landed(sink, anchor_path):
    receipt = sink.publish(make_checkpoint(4))

    assert receipt.location == anchor_path


def test_a_receipt_is_stamped_with_a_timezone_aware_instant(sink):
    receipt = sink.publish(make_checkpoint(4))

    assert receipt.published_at.utcoffset() is not None


def test_publishing_fsyncs_so_the_record_survives_a_power_cut(sink, monkeypatch):
    """An append-only JSONL that is not fsynced is a durability claim the sink cannot keep."""
    import signledger.anchor.sinks as sinks_module

    synced = []
    monkeypatch.setattr(sinks_module.os, "fsync", lambda fd: synced.append(fd))

    sink.publish(make_checkpoint(4))

    assert len(synced) == 1


def test_a_second_reader_sees_a_published_checkpoint_immediately(anchor_path):
    FileAnchorSink(anchor_path).publish(make_checkpoint(6))

    assert FileAnchorSink(anchor_path).fetch_latest("ledger-a").sequence == 6


# --------------------------------------------------------------------------------------
# a damaged anchor is reported, never skipped
# --------------------------------------------------------------------------------------


def test_a_corrupt_record_raises_rather_than_being_skipped(sink, anchor_path):
    """Silently skipping a bad line lets an attacker delete an anchor by corrupting it."""
    sink.publish(make_checkpoint(3))
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_missing_a_checkpoint_field_raises(sink, anchor_path):
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ledger_id": "ledger-a"}) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_that_is_not_a_json_object_raises(sink, anchor_path):
    """Matched on the message: a bare ``in`` test against a list would also raise, but for
    the wrong reason, and the shape check could then be deleted with the suite still green.
    """
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("[1, 2, 3]\n")

    with pytest.raises(AnchorError, match="not an object"):
        sink.fetch_latest("ledger-a")


def test_blank_lines_are_tolerated(sink, anchor_path):
    sink.publish(make_checkpoint(3))
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert sink.fetch_latest("ledger-a").sequence == 3


def test_publishing_into_a_directory_that_does_not_exist_raises_anchor_error(tmp_path):
    sink = FileAnchorSink(str(tmp_path / "nope" / "anchors.jsonl"))

    with pytest.raises(AnchorError):
        sink.publish(make_checkpoint(1))


# --------------------------------------------------------------------------------------
# the signing payload
# --------------------------------------------------------------------------------------


def test_the_signing_payload_is_deterministic():
    checkpoint = make_checkpoint(5)

    assert checkpoint_signing_payload(checkpoint) == checkpoint_signing_payload(checkpoint)


def test_the_signing_payload_excludes_the_signature_it_will_carry():
    """Otherwise a checkpoint could never be signed: the payload would depend on the
    signature that does not exist yet."""
    unsigned = make_checkpoint(5)
    signed = make_checkpoint(5, signature="c2ln")

    assert checkpoint_signing_payload(unsigned) == checkpoint_signing_payload(signed)


def test_the_signing_payload_covers_the_head_hash():
    left = make_checkpoint(5)
    right = make_checkpoint(5, head_hash="f" * 64)

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_covers_the_entry_count():
    left = make_checkpoint(5)
    right = make_checkpoint(5, entry_count=99)

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_covers_the_key_identity():
    """``key_id`` is inside the payload so an attacker cannot re-point a checkpoint at a
    key they control without invalidating the signature."""
    left = make_checkpoint(5, key_id="kid-1")
    right = make_checkpoint(5, key_id="kid-2")

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_cannot_collide_with_a_genesis_preimage():
    """Both are canonical JSON objects, so they need distinct type tags."""
    payload = checkpoint_signing_payload(make_checkpoint(5))

    assert b'"type":"signledger/checkpoint"' in payload


# --------------------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------------------


def test_the_file_sink_satisfies_the_anchor_sink_protocol(sink):
    assert isinstance(sink, AnchorSink)


def test_the_null_sink_satisfies_the_anchor_sink_protocol():
    assert isinstance(NullAnchorSink(), AnchorSink)


def test_an_unreadable_anchor_path_raises_anchor_error(tmp_path):
    """A directory where a file should be. Failing loud beats reporting 'not anchored'."""
    directory = tmp_path / "anchors.jsonl"
    directory.mkdir()

    with pytest.raises(AnchorError):
        FileAnchorSink(str(directory)).fetch_latest("ledger-a")


def test_a_record_with_an_unreadable_timestamp_raises(sink, anchor_path):
    sink.publish(make_checkpoint(3))
    with open(anchor_path, encoding="utf-8") as handle:
        record = json.loads(handle.read().strip())
    record["timestamp"] = "the third of never"
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_with_a_numeric_timestamp_raises(sink, anchor_path):
    """A bare epoch number cannot be read back: its unit is not recoverable."""
    record = dict.fromkeys(
        ("ledger_id", "head_hash", "merkle_root", "signature", "key_id"), "x"
    )
    record.update({"sequence": 1, "entry_count": 1, "format_version": 2, "timestamp": 1717243200})
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_an_anchor_error_never_quotes_the_path(tmp_path):
    """Contract section 10: a path can carry a deployment layout an operator would rather
    not see in a log."""
    directory = tmp_path / "secret-deployment-layout.jsonl"
    directory.mkdir()

    with pytest.raises(AnchorError) as raised:
        FileAnchorSink(str(directory)).fetch_latest("ledger-a")

    assert "secret-deployment-layout" not in str(raised.value)


# --------------------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------------------


def test_the_anchor_package_exports_everything_a_caller_needs():
    import signledger.anchor as anchor

    exported = {
        "AnchorReceipt",
        "AnchorSink",
        "Checkpoint",
        "FileAnchorSink",
        "NullAnchorSink",
        "checkpoint_signing_payload",
    }

    assert exported <= set(anchor.__all__)


def test_the_anchor_package_binds_every_name_it_advertises():
    import signledger.anchor as anchor

    assert all(hasattr(anchor, name) for name in anchor.__all__)


# --------------------------------------------------------------------------------------
# an anchor sink over real HTTP - the extension point of contract section 8
# --------------------------------------------------------------------------------------
#
# There is no shipped ``HTTPAnchorSink``. Section 8 names two sinks - file and null - and
# lists HTTP and notarization as *extension points*, and ``signledger/anchor/sinks.py``
# makes a promise about them at the top of the module: "implement AnchorSink and any
# ledger can use it". Nothing had ever tested that promise.
#
# So it is tested here, the only way that means anything: against a real HTTP server on a
# real socket in a real thread, with the sink a user would actually write. Nothing below
# mocks a sink, patches ``urlopen`` or fakes a response. What is under test is the shipped
# half - that ``AnchorSink`` really is implementable, that ``Ledger.checkpoint()`` really
# publishes through it, that ``check_integrity()`` really fetches back through it, and
# that a sink which fails cannot leave a ledger claiming to be anchored.

#: The sink gives up long before the slow server answers, so "did not hang" is provable.
SLOW_SERVER_DELAY = 1.0
IMPATIENT_SINK_TIMEOUT = 0.1

CHECKPOINT_FIELDS = (
    "ledger_id",
    "sequence",
    "head_hash",
    "entry_count",
    "merkle_root",
    "timestamp",
    "signature",
    "key_id",
    "format_version",
)


def checkpoint_to_json(cp):
    record = {field: getattr(cp, field) for field in CHECKPOINT_FIELDS}
    record["timestamp"] = cp.timestamp.astimezone(timezone.utc).isoformat()
    return record


def checkpoint_from_json(record):
    restored = {field: record[field] for field in CHECKPOINT_FIELDS}
    restored["timestamp"] = datetime.fromisoformat(restored["timestamp"])
    return Checkpoint(**restored)


def split_credentials(url):
    """Lift any ``user:password`` out of the URL.

    A credential in a URL ends up in receipts, in logs and in whatever the driver puts in
    its exception message. Contract section 10 forbids all three, so it travels in a
    header and the URL that is kept is the clean one.
    """
    parts = urlsplit(url)
    if parts.username is None:
        return url, None
    netloc = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
    clean = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return clean, f"{parts.username}:{parts.password or ''}"


class HTTPAnchorSink:
    """A checkpoint sink that POSTs to an HTTP endpoint and GETs the newest one back.

    This is the extension point, written out. It is deliberately the smallest thing that
    satisfies :class:`~signledger.anchor.sinks.AnchorSink`, so that what the tests below
    exercise is the library's side of the contract rather than any cleverness here.
    """

    name = "http"

    def __init__(self, url, *, timeout=5.0):
        self.url, self._credentials = split_credentials(url)
        self.timeout = timeout

    def _call(self, url, body=None):
        headers = {"Content-Type": "application/json"}
        if self._credentials is not None:
            token = base64.b64encode(self._credentials.encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        request = urllib.request.Request(
            url, data=body, headers=headers, method="POST" if body is not None else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            # An HTTPError *is* the response, and it still holds the socket: urllib only
            # closes the connection for errors raised while opening it, not for the 4xx/5xx
            # it turns into an exception afterwards.
            err.close()
            raise AnchorError(
                f"could not reach the anchor service: {type(err).__name__}"
            ) from err
        except OSError as err:
            # One clause for the rest: URLError subclasses OSError and a socket timeout is
            # one. Only the type is reported - the driver's own message quotes the URL,
            # which can carry a credential.
            raise AnchorError(
                f"could not reach the anchor service: {type(err).__name__}"
            ) from err

    def publish(self, cp):
        self._call(self.url, json.dumps(checkpoint_to_json(cp)).encode("utf-8"))
        return AnchorReceipt(
            sink=self.name,
            ledger_id=cp.ledger_id,
            sequence=cp.sequence,
            published_at=datetime.now(timezone.utc),
            location=self.url,
        )

    def fetch_latest(self, ledger_id):
        record = json.loads(
            self._call(f"{self.url}?ledger_id={quote(ledger_id)}").decode("utf-8")
        )
        return None if record is None else checkpoint_from_json(record)


class _AnchorHandler(BaseHTTPRequestHandler):
    """The notary end: it stores what it is given and hands the newest one back."""

    def log_message(self, fmt, *args):
        """Silence. The default writes a line to stderr for every request."""

    def do_POST(self):
        service = self.server.service
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        service.authorizations.append(self.headers.get("Authorization"))
        if service.delay:
            time.sleep(service.delay)
        service.bodies.append(body)
        if service.publish_status != 200:
            self._reply(service.publish_status, b'{"error": "the anchor service refused"}')
            return
        service.records.append(json.loads(body.decode("utf-8")))
        self._reply(200, b'{"anchored": true}')

    def do_GET(self):
        service = self.server.service
        wanted = (parse_qs(urlsplit(self.path).query).get("ledger_id") or [""])[0]
        matching = [record for record in service.records if record["ledger_id"] == wanted]
        newest = max(matching, key=lambda record: record["sequence"], default=None)
        self._reply(200, json.dumps(newest).encode("utf-8"))

    def _reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _AnchorServer(ThreadingHTTPServer):
    daemon_threads = True
    #: Teardown must not wait on the handler thread the timeout test deliberately stranded.
    block_on_close = False

    def handle_error(self, request, client_address):
        """A client that timed out and walked away is a test's point, not a fault."""


class AnchorService:
    """A live HTTP anchor endpoint. Real socket, real thread, real bytes on the wire."""

    def __init__(self, *, publish_status=200, delay=0.0):
        self.publish_status = publish_status
        self.delay = delay
        #: Every request body the server actually received, in arrival order.
        self.bodies = []
        #: The Authorization header of each, so a redaction test cannot pass by dropping it.
        self.authorizations = []
        #: The checkpoints it accepted.
        self.records = []
        self.server = _AnchorServer(("127.0.0.1", 0), _AnchorHandler)
        self.server.service = self
        # A short poll interval only matters at teardown: ``shutdown()`` waits for the
        # accept loop to notice, and the default 0.5s per service dominates the run.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/anchor"

    def url_with_credentials(self, user, password):
        host, port = self.server.server_address[:2]
        return f"http://{user}:{password}@{host}:{port}/anchor"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.fixture
def anchor_service():
    """Factory: a live HTTP anchor endpoint, shut down again afterwards."""
    running = []

    def _start(**options):
        service = AnchorService(**options)
        running.append(service)
        return service

    try:
        yield _start
    finally:
        for service in running:
            service.stop()


@pytest.fixture
def unreachable_url():
    """A port nothing listens on: bound only long enough to learn a free number."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/anchor"


@pytest.fixture
def signer():
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.signatures import Ed25519Signer

    return Ed25519Signer.generate()


@pytest.fixture(params=["connection_refused", "server_error", "timeout"])
def broken_anchor(request, anchor_service, unreachable_url):
    """A sink aimed at an anchor service that is down, refusing, or too slow to matter."""
    if request.param == "connection_refused":
        return HTTPAnchorSink(unreachable_url, timeout=IMPATIENT_SINK_TIMEOUT)
    if request.param == "server_error":
        return HTTPAnchorSink(
            anchor_service(publish_status=500).url, timeout=IMPATIENT_SINK_TIMEOUT
        )
    return HTTPAnchorSink(
        anchor_service(delay=SLOW_SERVER_DELAY).url, timeout=IMPATIENT_SINK_TIMEOUT
    )


# -- the sink satisfies the shipped protocol --------------------------------------------


def test_a_third_party_http_sink_satisfies_the_anchor_sink_protocol(anchor_service):
    """The claim in the sinks module docstring, measured rather than asserted."""
    assert isinstance(HTTPAnchorSink(anchor_service().url), AnchorSink)


# -- what actually goes over the wire ---------------------------------------------------


def test_a_checkpoint_published_over_http_actually_arrives(anchor_service):
    service = anchor_service()

    HTTPAnchorSink(service.url).publish(make_checkpoint(4))

    assert len(service.bodies) == 1


def test_the_body_that_arrives_is_the_checkpoint_the_sink_was_given(anchor_service):
    """Compared field by field against the ``Checkpoint`` itself, not against the encoder
    that produced the body - otherwise this only proves the encoder agrees with itself."""
    service = anchor_service()
    published = make_checkpoint(4, signature="c2ln", key_id="kid-1")

    HTTPAnchorSink(service.url).publish(published)
    received = json.loads(service.bodies[0].decode("utf-8"))

    assert (
        received["ledger_id"],
        received["sequence"],
        received["head_hash"],
        received["entry_count"],
        received["merkle_root"],
        received["signature"],
        received["key_id"],
        received["format_version"],
    ) == (
        published.ledger_id,
        published.sequence,
        published.head_hash,
        published.entry_count,
        published.merkle_root,
        published.signature,
        published.key_id,
        published.format_version,
    )


def test_a_checkpoint_published_over_http_comes_back_field_for_field(anchor_service):
    sink = HTTPAnchorSink(anchor_service().url)
    published = make_checkpoint(4, signature="c2ln", key_id="kid-1")
    sink.publish(published)

    assert sink.fetch_latest("ledger-a") == published


def test_fetching_a_ledger_the_http_service_has_never_seen_is_none(anchor_service):
    sink = HTTPAnchorSink(anchor_service().url)
    sink.publish(make_checkpoint(4, ledger_id="ledger-a"))

    assert sink.fetch_latest("ledger-b") is None


def test_an_http_receipt_records_where_the_checkpoint_landed(anchor_service):
    service = anchor_service()

    receipt = HTTPAnchorSink(service.url).publish(make_checkpoint(4))

    assert receipt.location == service.url


# -- failure is AnchorError, every time -------------------------------------------------


def test_an_anchor_service_that_fails_makes_publish_raise_anchor_error(broken_anchor):
    """Refused, 500, and too slow. All three, against real sockets."""
    with pytest.raises(AnchorError):
        broken_anchor.publish(make_checkpoint(4))


def test_a_slow_anchor_service_is_given_up_on_rather_than_waited_out(anchor_service):
    """"Raises eventually" is not the property. Giving up *before* the server answers is."""
    service = anchor_service(delay=SLOW_SERVER_DELAY)
    sink = HTTPAnchorSink(service.url, timeout=IMPATIENT_SINK_TIMEOUT)

    started = time.monotonic()
    with pytest.raises(AnchorError):
        sink.publish(make_checkpoint(4))

    assert time.monotonic() - started < SLOW_SERVER_DELAY


def test_a_url_carrying_credentials_is_never_echoed_into_the_error(anchor_service):
    """Contract section 10: a connection string is exactly what must not reach a log."""
    service = anchor_service(publish_status=500)
    sink = HTTPAnchorSink(service.url_with_credentials("auditor", "hunter2-do-not-log"))

    with pytest.raises(AnchorError) as raised:
        sink.publish(make_checkpoint(4))

    assert "hunter2-do-not-log" not in str(raised.value)


def test_a_receipt_never_carries_the_credentials_from_its_url(anchor_service):
    service = anchor_service()
    sink = HTTPAnchorSink(service.url_with_credentials("auditor", "hunter2-do-not-log"))

    receipt = sink.publish(make_checkpoint(4))

    assert "hunter2-do-not-log" not in (receipt.location or "")


def test_credentials_taken_out_of_the_url_are_still_presented_to_the_service(anchor_service):
    """Otherwise the two tests above would pass just as well by throwing the credential
    away, and the sink would silently stop authenticating."""
    service = anchor_service()
    sink = HTTPAnchorSink(service.url_with_credentials("auditor", "hunter2-do-not-log"))
    expected = base64.b64encode(b"auditor:hunter2-do-not-log").decode("ascii")

    sink.publish(make_checkpoint(4))

    assert service.authorizations == [f"Basic {expected}"]


# -- a real ledger, anchored over a real network hop ------------------------------------


def test_a_ledger_publishes_its_checkpoint_to_an_http_sink(anchor_service, signer):
    service = anchor_service()

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "opened"})
        ledger.checkpoint()

    assert len(service.bodies) == 1


def test_the_checkpoint_a_ledger_publishes_records_the_real_head_and_count(
    anchor_service, signer
):
    service = anchor_service()

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "first"})
        head = ledger.append({"event": "second"})
        ledger.checkpoint()
    received = json.loads(service.bodies[0].decode("utf-8"))

    assert (received["head_hash"], received["entry_count"]) == (head.hash, 2)


def test_the_checkpoint_a_ledger_publishes_carries_no_entry_payload(anchor_service, signer):
    """Contract section 10. A checkpoint leaves the building; the records do not."""
    service = anchor_service()

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"diagnosis": "MARKER-9c41-do-not-publish"})
        ledger.checkpoint()

    assert b"MARKER-9c41-do-not-publish" not in service.bodies[0]


def test_a_ledger_anchored_over_real_http_earns_anchored(anchor_service, signer):
    """ANCHORED conferred by a checkpoint that made a round trip over a socket."""
    service = anchor_service()

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "opened"})
        ledger.checkpoint()

        assert ledger.assurance_level() is AssuranceLevel.ANCHORED


def test_a_ledger_verifies_against_the_checkpoint_that_came_back_over_http(
    anchor_service, signer
):
    """Not the copy it kept: the one the notary handed back, decoded from the wire."""
    service = anchor_service()

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "opened"})
        published = ledger.checkpoint()
        report = ledger.check_integrity()

    assert report.checkpoint_used == published


def test_a_tail_truncated_after_an_http_anchor_is_reported_as_head_mismatch(
    anchor_service, signer
):
    """The reason anchoring exists. What is left after the deletion is self-consistent;
    only the record held at the far end of that socket contradicts it."""
    service = anchor_service()
    store = MemoryBackend()

    with Ledger(
        store, enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        for index in range(4):
            ledger.append({"event": "kept", "index": index})
        ledger.checkpoint()
        del store._entries[2:]
        report = ledger.check_integrity()

    assert FailureKind.HEAD_MISMATCH in {failure.kind for failure in report.failures}


def test_a_tail_truncated_after_an_http_anchor_is_also_reported_as_count_mismatch(
    anchor_service, signer
):
    service = anchor_service()
    store = MemoryBackend()

    with Ledger(
        store, enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        for index in range(4):
            ledger.append({"event": "kept", "index": index})
        ledger.checkpoint()
        del store._entries[2:]
        report = ledger.check_integrity()

    assert FailureKind.COUNT_MISMATCH in {failure.kind for failure in report.failures}


# -- a sink that failed must not leave a ledger looking anchored ------------------------


def test_a_ledger_whose_anchor_service_refuses_raises_anchor_error(anchor_service, signer):
    service = anchor_service(publish_status=500)

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "opened"})

        with pytest.raises(AnchorError):
            ledger.checkpoint()


def test_a_checkpoint_the_anchor_service_refused_confers_no_anchored_assurance(
    anchor_service, signer
):
    """The local copy was already saved when the publish failed. It must count for
    nothing: a checkpoint nobody outside this process holds is not an anchor."""
    service = anchor_service(publish_status=500)

    with Ledger(
        enable_signatures=True, signer=signer, anchor_sink=HTTPAnchorSink(service.url)
    ) as ledger:
        ledger.append({"event": "opened"})
        with pytest.raises(AnchorError):
            ledger.checkpoint()

        assert ledger.assurance_level() is AssuranceLevel.SIGNED


def test_an_unreachable_anchor_service_makes_verification_raise_not_quietly_downgrade(
    unreachable_url, signer
):
    """If a failed fetch fell back to the local checkpoint, cutting the network would be
    enough to stop a ledger ever checking itself against its anchor again."""
    sink = HTTPAnchorSink(unreachable_url, timeout=IMPATIENT_SINK_TIMEOUT)

    with Ledger(enable_signatures=True, signer=signer, anchor_sink=sink) as ledger:
        ledger.append({"event": "opened"})

        with pytest.raises(AnchorError):
            ledger.check_integrity()


def test_a_failing_anchor_never_leaks_the_entry_payload_into_the_error(broken_anchor, signer):
    """Contract section 10, on all three failure modes at once."""
    with Ledger(enable_signatures=True, signer=signer, anchor_sink=broken_anchor) as ledger:
        ledger.append({"diagnosis": "MARKER-4b7e-do-not-log"})

        with pytest.raises(AnchorError) as raised:
            ledger.checkpoint()

    assert "MARKER-4b7e-do-not-log" not in str(raised.value)
