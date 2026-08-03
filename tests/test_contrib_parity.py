"""Cross-framework parity: one shape for Flask, FastAPI and Django, or a red suite.

:mod:`signledger.contrib` promises, in as many words, "One shape for all three
frameworks, so a query an auditor writes against a Flask service still works against the
Django one." That promise was measurably false in every direction this file now checks.
On byte-identical traffic the three integrations disagreed about how many entries a
request produced, about whether an error response was ``provisional``, about whether the
``error`` field existed at all, and about what a ``HEAD`` request to a streaming route
recorded. Every one of those divergences was invisible to every single-framework test,
because a single-framework test cannot see a disagreement.

What replaced what
------------------

The only cross-framework test this repository had built a Flask app and a Django client
and compared ``set(...)`` of the entry **keys** of ``entries[0]``, for one ``200`` ``GET``,
with FastAPI absent entirely. Key sets, not values; one entry, not the sequence; two
frameworks, not three; and in-process test clients, which is the part that matters most -
werkzeug's test client does not close the response iterable, and that close is the only
moment a streamed response's outcome is knowable. It could not have caught any of the
defects that shipped. It is deleted, and this file is what stands in its place.

How this suite is driven
------------------------

Over **real sockets**, against real servers: ``werkzeug.serving.make_server`` for Flask,
``uvicorn`` for FastAPI, and ``django.core.servers.basehttp`` driving a real
``WSGIHandler`` for Django, each bound to ``127.0.0.1`` and spoken to with a plain
``socket`` HTTP/1.1 client written here. In-process test clients are not acceptable for
the streaming scenarios and are not used for any of them: the point of the matrix is that
one wire-level request is issued to all three stacks and the resulting ledger rows are
compared, so the transport has to be the same transport for all three.

Every scenario is exercised once, in a module-scoped fixture, and the assertions are then
made against what was recorded. That is what keeps a nine-scenario, three-framework,
real-socket matrix cheap enough to run on every push.

Nothing here carries the ``slow`` marker, and that is a measurement rather than an
oversight. Standing up all three servers and driving all twenty-seven requests is the
whole socket cost of this module and it is charged once, to the fixture: ``--durations``
puts it at 1.4s, against a file that completes in under three. ``slow`` in this repository
means the concurrency and stress suites, which are two orders of magnitude above that.
Marking this module would mislabel the one suite that most needs to run unfiltered.

What is asserted, and what is deliberately not
----------------------------------------------

For every scenario, all three frameworks must agree on the number of entries, the ``kind``
of each, the ``completion`` of each (which must never be absent), whether ``error`` is
present and what it names, the ``outcome`` of a settlement, the full **key set** of a
request entry and of a settlement entry, and the values of every field whose value is not
allowed to depend on which framework wrote the row. ``request_id`` must correlate a
provisional entry with its settlement, no settlement may be an orphan, and no provisional
may be left unsettled. Each framework's ledger must still verify.

Three things legitimately differ and are named rather than smoothed over: ``integration``
(that is the field's job), ``request_id`` (a fresh uuid per request), and the ``host``
request header (each server is on its own ephemeral port).

Two behavioural divergences are pinned as documented behaviour rather than treated as
defects. One is in the library and intended - see
:func:`test_fastapi_records_a_body_its_endpoint_never_read_as_omitted`. The other is in
the transport this suite is required to use and is not signledger's at all: on a request
with no body, the Django row carries a ``content-length`` and a ``content-type`` nothing
sent, because ``django.core.servers.basehttp`` manufactures both into the WSGI environ
and ``django.http.HttpHeaders`` republishes them. It is pinned to that exact pair, with
the cause executed rather than asserted, in
:func:`test_only_django_reports_headers_the_client_never_sent_and_only_the_wsgi_pair` and
:func:`test_djangos_extra_headers_come_from_the_wsgi_environ_and_not_from_signledger`.

Provenance of the assertions in here
------------------------------------

The divergences listed at the top were fixed before this file existed, so most of what is
below is a **coverage hole being closed, not a bug being caught**: the assertions pass
against the code as it stands. They were each shown to be load-bearing by breaking the
code and watching the named test fail - ``will_send_body`` reverted to testing the status
alone (the ``HEAD`` case), ``completion`` dropped from ``build_audit_record`` (the key-set
and completion cases), and ``kind`` stamped by the callers again rather than by the
builder (the ``kind`` case). None of those breakages is present in the tree; the tests
that caught them are.
"""

import contextlib
import dataclasses
import inspect
import json
import socket
import threading
import time
from typing import Any, AsyncIterator, Dict, FrozenSet, Iterator, List, Optional, Tuple

import django
import pytest
import uvicorn
from django.conf import settings

#: The audit middleware by path, so the Django server can be built around it alone.
AUDIT_MIDDLEWARE = "signledger.contrib.django.middleware.SignLedgerAuditMiddleware"

# Django's settings are process-global: whichever test module is imported first owns them
# for the life of the process, and ``settings.configure`` cannot be called twice. Under
# the normal alphabetical collection order ``tests/test_contrib_django.py`` is imported
# first and configures them; this block is only what makes
# ``pytest tests/test_contrib_parity.py`` on its own work as well.
#
# It configures by *delegation* rather than by keeping a second copy. A narrower copy here
# is not a harmless duplicate - it is a landmine: it would win the race whenever this file
# is named first on the command line, and the Django suite would then die during collection
# with "No installed app with label 'admin'", because it needs admin, auth, sessions,
# templates and a shared-cache database that this suite has no use for. Delegating means
# there is one definition of those settings and it cannot drift out from under the module
# that actually depends on it.
#
# Nothing this suite asserts rides on which settings won: ``django_serving`` re-states
# every setting the parity servers depend on through ``override_settings``, deliberately
# including cutting ``MIDDLEWARE`` back to the audit middleware alone.
if not settings.configured:  # pragma: no cover - depends on collection order
    try:
        import test_contrib_django  # noqa: F401
    except ImportError:  # pragma: no cover - only when this file is run detached from it
        settings.configure(
            DEBUG=False,
            DEBUG_PROPAGATE_EXCEPTIONS=False,
            SECRET_KEY="signledger-parity-key",
            ALLOWED_HOSTS=["*"],
            USE_TZ=True,
            DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
            INSTALLED_APPS=["signledger.contrib.django"],
            MIDDLEWARE=[AUDIT_MIDDLEWARE],
            ROOT_URLCONF=__name__,
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        )
        django.setup()

from django.core.handlers.wsgi import WSGIHandler  # noqa: E402
from django.core.servers.basehttp import (  # noqa: E402
    ThreadedWSGIServer,
    WSGIRequestHandler,
)
from django.http import (  # noqa: E402
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.test import override_settings  # noqa: E402
from django.urls import path as url  # noqa: E402
from django.views.decorators.http import require_http_methods  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi import Request as FastAPIRequest  # noqa: E402
from flask import Flask  # noqa: E402
from flask import Response as FlaskResponse  # noqa: E402
from flask import request as flask_request  # noqa: E402
from starlette.responses import Response as StarletteResponse  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

from signledger.backends.memory import MemoryBackend  # noqa: E402
from signledger.contrib import (  # noqa: E402
    COMPLETION_FINAL,
    COMPLETION_PROVISIONAL,
    KIND_COMPLETION,
    KIND_REQUEST,
    OUTCOME_COMPLETE,
    OUTCOME_INCOMPLETE,
)
from signledger.contrib.django import reset_ledger  # noqa: E402
from signledger.contrib.fastapi import SignLedgerAuditMiddleware  # noqa: E402
from signledger.contrib.flask import SignLedgerAudit  # noqa: E402
from signledger.contrib.redaction import REDACTED, RedactionPolicy  # noqa: E402
from signledger.core.ledger import Ledger  # noqa: E402

# --------------------------------------------------------------------------------------
# The traffic
# --------------------------------------------------------------------------------------

#: Every framework is asked the identical questions on the identical paths. A scenario
#: owns its path outright so its entries can be picked out of a shared ledger by
#: ``entry.data["path"]`` without any ordering assumption.
PLAIN = "/s/plain"
ABSENT = "/s/absent"
GET_ONLY = "/s/get-only"
BOOM = "/s/boom"
STREAM = "/s/stream"
STREAM_BROKEN = "/s/stream-broken"
STREAM_HEAD = "/s/stream-head"
NO_CONTENT = "/s/no-content"
ECHO = "/s/echo"

#: Appended to every request, so ``query`` is a non-trivial value in every scenario.
QUERY = "?page=2"

#: The frameworks, in the order failures are reported in.
FRAMEWORKS: Tuple[str, ...] = ("flask", "fastapi", "django")

#: Sent on every request. ``authorization`` must come back ``[REDACTED]`` from all three,
#: and ``x-parity-tag`` must come back verbatim from all three - the two halves of "the
#: header policy is the same policy".
COMMON_HEADERS: Dict[str, str] = {
    "Accept": "*/*",
    "Authorization": "Bearer parity-secret",
    "X-Parity-Tag": "matrix",
}

#: The body every POST carries. ``password`` is here to prove body redaction is the same
#: redaction in all three, not merely that a body was captured.
POST_BODY: bytes = json.dumps({"note": "parity", "password": "hunter2"}).encode("utf-8")

#: What that body must look like once it is in the ledger.
CAPTURED_BODY: Dict[str, Any] = {"note": "parity", "password": REDACTED}

#: What every integration records for a request that declared no length at all.
NO_LENGTH = "[OMITTED:no-content-length]"

#: What FastAPI records for a body its endpoint never read. See the divergence test.
NOT_READ = "[OMITTED:not-read]"

#: Raised by the failing view and by the failing generator, so the ``error`` field has a
#: name every framework has to agree on.
BOOM_MESSAGE = "the view exploded"

#: The two chunks a healthy stream emits.
CHUNKS: Tuple[bytes, ...] = (b'{"row": 0}\n', b'{"row": 1}\n')


def _chunks() -> Iterator[bytes]:
    yield from CHUNKS


def _broken_chunks() -> Iterator[bytes]:
    yield CHUNKS[0]
    raise RuntimeError(BOOM_MESSAGE)


async def _async_chunks() -> AsyncIterator[bytes]:
    for chunk in CHUNKS:
        yield chunk


async def _broken_async_chunks() -> AsyncIterator[bytes]:
    yield CHUNKS[0]
    raise RuntimeError(BOOM_MESSAGE)


# --------------------------------------------------------------------------------------
# A real socket client
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Reply:
    """What actually came back over the wire."""

    status: int
    reason: str
    headers: Dict[str, str]
    body: bytes


def request_over_a_socket(
    port: int,
    method: str,
    target: str,
    *,
    body: Optional[bytes] = None,
    timeout: float = 20.0,
) -> Reply:
    """One HTTP/1.1 request on a real TCP connection, read to end of stream.

    Hand-written rather than driven through ``requests``/``httpx`` for two reasons. The
    frameworks must be given byte-identical traffic, which means controlling the exact
    header set and their order; and a streamed response that dies mid-body has to be
    readable as "what arrived before the connection went away" rather than raising, which
    is what a client library does with a truncated chunked body.

    ``Connection: close`` makes end-of-response and end-of-stream the same event, so this
    needs no chunked decoder to know when a response has finished.
    """
    headers = {"Host": f"127.0.0.1:{port}", "Connection": "close", **COMMON_HEADERS}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    head = f"{method} {target} HTTP/1.1\r\n" + "".join(
        f"{name}: {value}\r\n" for name, value in headers.items()
    )
    wire = head.encode("latin-1") + b"\r\n" + (body or b"")

    received: List[bytes] = []
    connection = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    with contextlib.closing(connection):
        connection.sendall(wire)
        while True:
            try:
                data = connection.recv(65536)
            except (ConnectionResetError, TimeoutError, socket.timeout):
                # A server whose body generator raised mid-stream drops the connection.
                # That is the scenario, not a harness failure.
                break
            if not data:
                break
            received.append(data)
    return _parse(b"".join(received))


def _parse(raw: bytes) -> Reply:
    head, _, payload = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    if not lines or not lines[0]:
        raise AssertionError("the server closed the connection without sending a status line")
    _, _, rest = lines[0].partition(" ")
    code, _, reason = rest.partition(" ")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return Reply(status=int(code), reason=reason, headers=headers, body=payload)


# --------------------------------------------------------------------------------------
# Flask
# --------------------------------------------------------------------------------------


def build_flask_app(ledger: Ledger) -> Flask:
    app = Flask("signledger_parity_flask")
    SignLedgerAudit(app, ledger=ledger, redaction=RedactionPolicy(capture_body=True))

    @app.get(PLAIN)
    def plain() -> Dict[str, Any]:
        return {"ok": True}

    @app.get(GET_ONLY)
    def get_only() -> Dict[str, Any]:
        return {"ok": True}

    @app.get(BOOM)
    def boom() -> Dict[str, Any]:
        raise RuntimeError(BOOM_MESSAGE)

    @app.get(STREAM)
    def stream() -> FlaskResponse:
        return FlaskResponse(_chunks(), mimetype="application/x-ndjson")

    @app.get(STREAM_HEAD)
    def stream_head() -> FlaskResponse:
        return FlaskResponse(_chunks(), mimetype="application/x-ndjson")

    @app.get(STREAM_BROKEN)
    def stream_broken() -> FlaskResponse:
        return FlaskResponse(_broken_chunks(), mimetype="application/x-ndjson")

    @app.get(NO_CONTENT)
    def no_content() -> FlaskResponse:
        return FlaskResponse(status=204)

    @app.post(ECHO)
    def echo() -> Dict[str, Any]:
        # Reads the body, which is what makes this the "endpoint did read it" half of the
        # documented FastAPI divergence.
        flask_request.get_json()
        return {"ok": True}

    return app


@contextlib.contextmanager
def flask_serving(ledger: Ledger) -> Iterator[int]:
    server = make_server("127.0.0.1", 0, build_flask_app(ledger), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.port)
    finally:
        server.shutdown()
        thread.join(timeout=15)
        server.server_close()


# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------


def build_fastapi_app(ledger: Ledger) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        SignLedgerAuditMiddleware, ledger=ledger, redaction=RedactionPolicy(capture_body=True)
    )

    @app.get(PLAIN)
    async def plain() -> Dict[str, Any]:
        return {"ok": True}

    @app.get(GET_ONLY)
    async def get_only() -> Dict[str, Any]:
        return {"ok": True}

    @app.get(BOOM)
    async def boom() -> Dict[str, Any]:
        raise RuntimeError(BOOM_MESSAGE)

    @app.get(STREAM)
    async def stream() -> StreamingResponse:
        return StreamingResponse(_async_chunks(), media_type="application/x-ndjson")

    # ``api_route(methods=["GET", "HEAD"])`` rather than ``get``. Starlette's plain
    # ``Route`` adds ``HEAD`` wherever ``GET`` is registered and FastAPI's ``APIRoute``
    # does not, so ``@app.get`` here would answer the HEAD scenario with a 405 and the
    # matrix would be comparing a 405 against two 200s. That is a routing difference
    # between the two web frameworks, not an audit difference, and the point of this
    # route is to give all three the same request to record.
    @app.api_route(STREAM_HEAD, methods=["GET", "HEAD"])
    async def stream_head() -> StreamingResponse:
        return StreamingResponse(_async_chunks(), media_type="application/x-ndjson")

    @app.get(STREAM_BROKEN)
    async def stream_broken() -> StreamingResponse:
        return StreamingResponse(_broken_async_chunks(), media_type="application/x-ndjson")

    @app.get(NO_CONTENT)
    async def no_content() -> StarletteResponse:
        return StarletteResponse(status_code=204)

    @app.post(ECHO)
    async def echo(request: FastAPIRequest) -> Dict[str, Any]:
        await request.json()
        return {"ok": True}

    return app


@contextlib.contextmanager
def fastapi_serving(ledger: Ledger) -> Iterator[int]:
    """A real uvicorn server on a real port, in a thread, for the duration of the suite.

    ``lifespan="off"`` because this application has no startup or shutdown hooks and the
    lifespan protocol is not what is under test; ``access_log=False`` because the traffic
    below includes deliberate 500s and a stream that dies, and the log lines for those are
    noise rather than evidence. ``uvicorn.Server.capture_signals`` is a no-op off the main
    thread, so running it here does not touch the interpreter's signal handlers.

    ``ws="none"`` is required, not tidiness. ``pyproject.toml`` sets
    ``filterwarnings = ["error"]``, and uvicorn's default ``ws="auto"`` imports
    ``uvicorn.protocols.websockets.websockets_impl``, which imports
    ``websockets.legacy`` - deprecated since websockets 14.0, and it warns at import.
    Inside the server thread that becomes an exception, ``Server.run`` dies before
    ``started`` is ever set, and the only symptom is a server that never came up. This
    suite speaks no websockets, so the protocol is not loaded at all.
    """
    config = uvicorn.Config(
        build_fastapi_app(ledger),
        host="127.0.0.1",
        port=0,
        log_level="critical",
        access_log=False,
        lifespan="off",
        ws="none",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():  # pragma: no cover
            raise AssertionError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield int(server.servers[0].sockets[0].getsockname()[1])
    finally:
        server.should_exit = True
        thread.join(timeout=30)


# --------------------------------------------------------------------------------------
# Django
# --------------------------------------------------------------------------------------

#: The ledger ``LEDGER_FACTORY`` hands back. Django resolves its ledger through a
#: process-global memo (``signledger.contrib.django.get_ledger``), so the factory has to
#: read a module global rather than close over one.
_DJANGO_LEDGER: Optional[Ledger] = None


def django_ledger_factory() -> Ledger:
    assert _DJANGO_LEDGER is not None, "the Django parity server was not started"
    return _DJANGO_LEDGER


def django_plain(request: Any) -> HttpResponse:
    return JsonResponse({"ok": True})


@require_http_methods(["GET"])
def django_get_only(request: Any) -> HttpResponse:
    return JsonResponse({"ok": True})


def django_boom(request: Any) -> HttpResponse:
    raise RuntimeError(BOOM_MESSAGE)


def django_stream(request: Any) -> HttpResponse:
    return StreamingHttpResponse(_chunks(), content_type="application/x-ndjson")


def django_stream_broken(request: Any) -> HttpResponse:
    return StreamingHttpResponse(_broken_chunks(), content_type="application/x-ndjson")


def django_no_content(request: Any) -> HttpResponse:
    return HttpResponse(status=204)


def django_echo(request: Any) -> HttpResponse:
    json.loads(request.body or b"{}")
    return JsonResponse({"ok": True})


#: ``ROOT_URLCONF`` points at this module, so these are the routes the Django server
#: serves. The paths are declared without their leading slash, as Django wants them.
urlpatterns = [
    url(PLAIN[1:], django_plain),
    url(GET_ONLY[1:], django_get_only),
    url(BOOM[1:], django_boom),
    url(STREAM[1:], django_stream),
    url(STREAM_HEAD[1:], django_stream),
    url(STREAM_BROKEN[1:], django_stream_broken),
    url(NO_CONTENT[1:], django_no_content),
    url(ECHO[1:], django_echo),
]

#: Everything the parity server needs and nothing else. ``INDEX_ENTRIES`` is off because
#: the ORM mirror is a Django-only convenience with no counterpart in the other two, and
#: switching it on would make this suite depend on a migrated database to say something
#: about a field it is not comparing.
DJANGO_SIGNLEDGER: Dict[str, Any] = {
    "LEDGER_FACTORY": django_ledger_factory,
    "CAPTURE_BODY": True,
    "INDEX_ENTRIES": False,
}


@contextlib.contextmanager
def django_serving(ledger: Ledger) -> Iterator[int]:
    """A real ``django.core.servers.basehttp`` server driving a real ``WSGIHandler``.

    ``MIDDLEWARE`` is cut down to the audit middleware alone on purpose. The session and
    auth middlewares that ``tests/test_contrib_django.py`` configures would make
    ``request.user`` - and therefore ``AuditRecorder``'s ``user`` field - depend on a
    migrated database, which is a property of that harness and not of the integration.
    The other two frameworks resolve no user either, so this is what makes the ``user``
    field comparable at all.

    ``WSGIHandler`` is constructed *inside* the override because it loads the middleware
    chain in ``__init__``.
    """
    global _DJANGO_LEDGER
    _DJANGO_LEDGER = ledger
    reset_ledger()
    with override_settings(
        ROOT_URLCONF=__name__,
        MIDDLEWARE=[AUDIT_MIDDLEWARE],
        ALLOWED_HOSTS=["*"],
        DEBUG=False,
        DEBUG_PROPAGATE_EXCEPTIONS=False,
        SIGNLEDGER=DJANGO_SIGNLEDGER,
    ):
        server = ThreadedWSGIServer(("127.0.0.1", 0), WSGIRequestHandler, allow_reuse_address=False)
        server.set_app(WSGIHandler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield int(server.server_address[1])
        finally:
            server.shutdown()
            thread.join(timeout=15)
            server.server_close()
            reset_ledger()
            _DJANGO_LEDGER = None


# --------------------------------------------------------------------------------------
# The scenario matrix
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One question, asked of all three stacks, and the answer all three must give.

    ``kinds``, ``completions``, ``errors`` and ``outcomes`` are positional: index ``i``
    describes the ``i``-th entry this scenario's path produced. ``errors`` and ``outcomes``
    use ``None`` for "this field must be **absent**", which is a different assertion from
    "this field must be null" and is the one the record shape actually makes.
    """

    name: str
    method: str
    path: str
    status: int
    kinds: Tuple[str, ...]
    completions: Tuple[str, ...]
    errors: Tuple[Optional[str], ...]
    outcomes: Tuple[Optional[str], ...]
    body: Optional[bytes] = None
    #: What each framework must record in ``body``. A single value means all three agree;
    #: a mapping names a divergence, and the only one in this suite is documented in
    #: :func:`test_fastapi_records_a_body_its_endpoint_never_read_as_omitted`.
    recorded_body: Any = NO_LENGTH
    diverging_recorded_body: Optional[Dict[str, Any]] = None

    def expected_body(self, framework: str) -> Any:
        if self.diverging_recorded_body is not None:
            return self.diverging_recorded_body[framework]
        return self.recorded_body


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        name="plain-200-get",
        method="GET",
        path=PLAIN,
        status=200,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
    ),
    Scenario(
        name="unrouted-404",
        method="GET",
        path=ABSENT,
        status=404,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
    ),
    Scenario(
        name="method-not-allowed-405",
        method="POST",
        path=GET_ONLY,
        status=405,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
        body=POST_BODY,
        # The one deliberate divergence in the suite. FastAPI's ASGI middleware only
        # buffers what the endpoint reads off ``receive``, and nothing reads a body before
        # answering 405.
        diverging_recorded_body={
            "flask": CAPTURED_BODY,
            "django": CAPTURED_BODY,
            "fastapi": NOT_READ,
        },
    ),
    Scenario(
        name="view-raises-500",
        method="GET",
        path=BOOM,
        status=500,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=("RuntimeError",),
        outcomes=(None,),
    ),
    Scenario(
        name="stream-completes",
        method="GET",
        path=STREAM,
        status=200,
        kinds=(KIND_REQUEST, KIND_COMPLETION),
        completions=(COMPLETION_PROVISIONAL, COMPLETION_FINAL),
        errors=(None, None),
        outcomes=(None, OUTCOME_COMPLETE),
    ),
    Scenario(
        name="stream-raises-after-first-chunk",
        method="GET",
        path=STREAM_BROKEN,
        status=200,
        kinds=(KIND_REQUEST, KIND_COMPLETION),
        completions=(COMPLETION_PROVISIONAL, COMPLETION_FINAL),
        errors=(None, "RuntimeError"),
        outcomes=(None, OUTCOME_INCOMPLETE),
    ),
    Scenario(
        name="head-on-a-streaming-route",
        method="HEAD",
        path=STREAM_HEAD,
        status=200,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
    ),
    Scenario(
        name="no-content-204",
        method="GET",
        path=NO_CONTENT,
        status=204,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
    ),
    Scenario(
        name="post-with-a-json-body",
        method="POST",
        path=ECHO,
        status=200,
        kinds=(KIND_REQUEST,),
        completions=(COMPLETION_FINAL,),
        errors=(None,),
        outcomes=(None,),
        body=POST_BODY,
        recorded_body=CAPTURED_BODY,
    ),
)

BY_NAME: Dict[str, Scenario] = {scenario.name: scenario for scenario in SCENARIOS}

#: The exact key set of a request entry, per :func:`signledger.contrib.build_audit_record`.
#: Written out rather than derived from the builder: a test that asks the code under test
#: what the shape is cannot notice the shape changing.
REQUEST_KEYS = frozenset(
    {
        "integration",
        "method",
        "path",
        "status",
        "client_ip",
        "headers",
        "query",
        "body",
        "user",
        "kind",
        "request_id",
        "completion",
    }
)

#: A settlement is a request entry plus how the promise ended.
SETTLEMENT_KEYS = REQUEST_KEYS | {"outcome"}

#: Fields whose value may not depend on which framework wrote the row. ``integration`` and
#: ``request_id`` are excluded because differing is their entire job; ``headers`` has its
#: own test because exactly one of its values (``host``) legitimately differs.
FIELDS_THAT_MUST_AGREE = ("method", "path", "status", "kind", "completion", "query", "user")


@dataclasses.dataclass(frozen=True)
class Observation:
    """What one framework did with one scenario."""

    framework: str
    scenario: str
    reply: Reply
    entries: Tuple[Dict[str, Any], ...]


def _entries_for(ledger: Ledger, path: str) -> List[Dict[str, Any]]:
    return [entry.data for entry in ledger.iter_entries() if entry.data.get("path") == path]


def _await_entries(ledger: Ledger, path: str, expected: int, timeout: float = 20.0) -> List[Any]:
    """Poll until ``expected`` entries exist for ``path``, then stop.

    A settlement is written by the server's own request thread once the body has stopped
    moving, so there is nothing in this thread to join on. Returning early on timeout is
    deliberate: the assertion that follows reports what is really there, which is a far
    more useful failure than a timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = _entries_for(ledger, path)
        if len(found) >= expected:
            return found
        time.sleep(0.02)
    return _entries_for(ledger, path)


@dataclasses.dataclass(frozen=True)
class Matrix:
    observations: Dict[Tuple[str, str], Observation]
    ledgers: Dict[str, Ledger]

    def at(self, scenario: str, framework: str) -> Observation:
        return self.observations[(scenario, framework)]

    def across(self, scenario: str) -> Dict[str, Observation]:
        return {name: self.at(scenario, name) for name in FRAMEWORKS}


@pytest.fixture(scope="module")
def matrix() -> Iterator[Matrix]:
    """Run every scenario against all three real servers, once, and keep what happened.

    Module-scoped because the servers are the expensive part and the traffic is not: nine
    scenarios times three frameworks is twenty-seven requests. Each scenario owns its path,
    so a shared per-framework ledger is still unambiguous, and it lets
    :func:`test_the_whole_ledger_of_every_framework_still_verifies` check a chain with
    every scenario in it rather than one.
    """
    ledgers = {
        name: Ledger(backend=MemoryBackend(), ledger_id=f"parity-{name}") for name in FRAMEWORKS
    }
    observations: Dict[Tuple[str, str], Observation] = {}
    try:
        with flask_serving(ledgers["flask"]) as flask_port, fastapi_serving(
            ledgers["fastapi"]
        ) as fastapi_port, django_serving(ledgers["django"]) as django_port:
            ports = {"flask": flask_port, "fastapi": fastapi_port, "django": django_port}
            for scenario in SCENARIOS:
                for name in FRAMEWORKS:
                    reply = request_over_a_socket(
                        ports[name], scenario.method, scenario.path + QUERY, body=scenario.body
                    )
                    recorded = _await_entries(
                        ledgers[name], scenario.path, len(scenario.kinds)
                    )
                    observations[(scenario.name, name)] = Observation(
                        framework=name,
                        scenario=scenario.name,
                        reply=reply,
                        entries=tuple(recorded),
                    )
        yield Matrix(observations=observations, ledgers=ledgers)
    finally:
        for ledger in ledgers.values():
            ledger.close()


def scenarios() -> Any:
    return pytest.mark.parametrize(
        "scenario", SCENARIOS, ids=[scenario.name for scenario in SCENARIOS]
    )


# --------------------------------------------------------------------------------------
# What must be true of all three, for every scenario
# --------------------------------------------------------------------------------------


@scenarios()
def test_every_framework_returns_the_same_status_over_a_real_socket(
    matrix: Matrix, scenario: Scenario
) -> None:
    """The premise. If the traffic is not equivalent, nothing below means anything."""
    statuses = {name: obs.reply.status for name, obs in matrix.across(scenario.name).items()}
    assert statuses == dict.fromkeys(FRAMEWORKS, scenario.status)


@scenarios()
def test_every_framework_writes_the_same_number_of_entries(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Entry counts diverged on real traffic: a ``HEAD`` to a streaming route was one
    entry on Flask and two on FastAPI, and every error response answered by Flask's error
    handler earned a spurious settlement entry it did not need."""
    counts = {name: len(obs.entries) for name, obs in matrix.across(scenario.name).items()}
    assert counts == dict.fromkeys(FRAMEWORKS, len(scenario.kinds))


@scenarios()
def test_every_framework_writes_the_same_kinds_in_the_same_order(
    matrix: Matrix, scenario: Scenario
) -> None:
    """``kind`` was stamped on by Flask and FastAPI after the shared builder returned and
    not by Django at all, so ``WHERE kind = 'request'`` saw two frameworks out of three."""
    kinds = {
        name: tuple(entry["kind"] for entry in obs.entries)
        for name, obs in matrix.across(scenario.name).items()
    }
    assert kinds == dict.fromkeys(FRAMEWORKS, scenario.kinds)


@scenarios()
def test_completion_is_present_on_every_entry_of_every_framework(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Absent is not a value. An auditor must not have to know which framework - or which
    version of which framework - wrote a row in order to know whether its status was a
    fact or a promise."""
    for name, obs in matrix.across(scenario.name).items():
        for position, entry in enumerate(obs.entries):
            assert "completion" in entry, f"{name} entry {position} of {scenario.name}"


@scenarios()
def test_every_framework_agrees_on_which_entries_are_provisional(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Whether a response was "provisional" is the field that disagreed most: Flask
    called every ``HTTPException``-rendered error streamed because werkzeug replaces the
    body with a ``ClosingIterator``, and FastAPI called a ``HEAD`` on a streaming route
    provisional and then waited forever for a settlement no server would produce."""
    completions = {
        name: tuple(entry["completion"] for entry in obs.entries)
        for name, obs in matrix.across(scenario.name).items()
    }
    assert completions == dict.fromkeys(FRAMEWORKS, scenario.completions)


@scenarios()
def test_every_framework_agrees_on_whether_error_is_present_and_what_it_names(
    matrix: Matrix, scenario: Scenario
) -> None:
    """``error`` is present only when there was one, and then it names the exception type.

    Both halves are load-bearing. A field that is always there and always null teaches
    people to stop reading it; and a 500 recorded with no ``error`` at all made
    ``WHERE error IS NOT NULL`` come back structurally empty however many views had blown
    up, which is what both the Flask and the Django integration used to do."""
    observed = {
        name: tuple(entry.get("error", None) for entry in obs.entries)
        for name, obs in matrix.across(scenario.name).items()
    }
    assert observed == dict.fromkeys(FRAMEWORKS, scenario.errors)
    for name, obs in matrix.across(scenario.name).items():
        for position, (entry, expected) in enumerate(zip(obs.entries, scenario.errors)):
            present = "error" in entry
            assert present is (expected is not None), (
                f"{name} entry {position} of {scenario.name}: "
                f"error present={present}, expected present={expected is not None}"
            )


@scenarios()
def test_every_framework_agrees_on_the_outcome_of_a_settlement(
    matrix: Matrix, scenario: Scenario
) -> None:
    """``outcome`` belongs to settlement entries and to nothing else."""
    observed = {
        name: tuple(entry.get("outcome", None) for entry in obs.entries)
        for name, obs in matrix.across(scenario.name).items()
    }
    assert observed == dict.fromkeys(FRAMEWORKS, scenario.outcomes)
    for name, obs in matrix.across(scenario.name).items():
        for entry in obs.entries:
            assert ("outcome" in entry) is (entry["kind"] == KIND_COMPLETION), (
                f"{name}: {entry['kind']} entry carries outcome={('outcome' in entry)}"
            )


@scenarios()
def test_a_request_entry_has_the_same_key_set_in_every_framework(
    matrix: Matrix, scenario: Scenario
) -> None:
    """The full key set, not merely the keys one framework happened to have.

    The test this file replaces compared key sets between two frameworks and no values;
    this compares key sets *and* pins them to the shape ``build_audit_record`` documents,
    so a field appearing in one integration only fails here even if all three grew it."""
    for name, obs in matrix.across(scenario.name).items():
        for position, (entry, kind) in enumerate(zip(obs.entries, scenario.kinds)):
            if kind != KIND_REQUEST:
                continue
            expected = REQUEST_KEYS | ({"error"} if scenario.errors[position] else set())
            assert set(entry) == expected, f"{name} entry {position} of {scenario.name}"


@scenarios()
def test_a_settlement_entry_has_the_same_key_set_in_every_framework(
    matrix: Matrix, scenario: Scenario
) -> None:
    """A settlement carries everything a request entry does, plus ``outcome``.

    That is not decoration. Both the Flask and the FastAPI settlement used to be
    hand-assembled after the request context was gone, so they named no client, no user
    and no headers - on precisely the half of the request that went wrong."""
    for name, obs in matrix.across(scenario.name).items():
        for position, (entry, kind) in enumerate(zip(obs.entries, scenario.kinds)):
            if kind != KIND_COMPLETION:
                continue
            expected = SETTLEMENT_KEYS | ({"error"} if scenario.errors[position] else set())
            assert set(entry) == expected, f"{name} entry {position} of {scenario.name}"


@scenarios()
def test_every_framework_records_the_same_values_for_the_fields_that_must_agree(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Values, not just keys. Agreeing on the *name* of a field the three fill in
    differently is the divergence this suite exists to catch, dressed as parity."""
    for field in FIELDS_THAT_MUST_AGREE:
        observed = {
            name: [entry[field] for entry in obs.entries]
            for name, obs in matrix.across(scenario.name).items()
        }
        # Compared by equality against one framework's answer rather than through a set:
        # ``query`` is a dict and a set would raise ``TypeError: unhashable type`` before
        # it ever compared anything.
        assert observed == dict.fromkeys(FRAMEWORKS, observed["flask"]), (
            f"{field} diverged across frameworks: {observed}"
        )


@scenarios()
def test_every_framework_records_the_expected_values_and_not_merely_the_same_ones(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Three frameworks agreeing on a wrong answer is still a wrong answer.

    Several tests in this repository have asserted the product's own output back at it,
    so the parity assertions above are paired with this one, which states independently
    what each field has to hold."""
    for name, obs in matrix.across(scenario.name).items():
        for position, entry in enumerate(obs.entries):
            where = f"{name} entry {position} of {scenario.name}"
            assert entry["integration"] == name, where
            assert entry["method"] == scenario.method, where
            assert entry["path"] == scenario.path, where
            assert entry["status"] == scenario.status, where
            assert entry["kind"] == scenario.kinds[position], where
            assert entry["completion"] == scenario.completions[position], where
            assert entry["query"] == {"page": ["2"]}, where
            assert entry["user"] is None, where
            assert entry["client_ip"] == "127.0.0.1", where
            assert entry["body"] == scenario.expected_body(name), where


def _client_header_names(scenario: Scenario) -> FrozenSet[str]:
    """The header names :func:`request_over_a_socket` actually put on the wire."""
    sent = {name.lower() for name in COMMON_HEADERS} | {"host", "connection"}
    if scenario.body is not None:
        sent |= {"content-type", "content-length"}
    return frozenset(sent)


@scenarios()
def test_every_framework_records_every_header_the_client_actually_sent(
    matrix: Matrix, scenario: Scenario
) -> None:
    """Header names are lower-cased by ``redact_headers`` precisely so one ledger holds
    one spelling, so the recorded mappings are directly comparable.

    ``host`` is compared for presence only: each server is on its own ephemeral port, so
    its value is the one header value that legitimately differs. Every other header the
    client sent - which names survived, and which values were replaced by the redaction
    marker - must match exactly across the three, because the three integrations are
    handed byte-identical bytes."""
    recorded = {
        name: dict(obs.entries[0]["headers"]) for name, obs in matrix.across(scenario.name).items()
    }
    sent = _client_header_names(scenario)
    for name, headers in recorded.items():
        assert sent <= set(headers), f"{name} dropped {sorted(sent - set(headers))}"
        assert headers["host"].startswith("127.0.0.1:"), name
    of_the_sent = {
        name: {key: headers[key] for key in sent if key != "host"}
        for name, headers in recorded.items()
    }
    assert of_the_sent == dict.fromkeys(FRAMEWORKS, of_the_sent["flask"]), of_the_sent
    for name, headers in recorded.items():
        assert headers["authorization"] == REDACTED, name
        assert headers["x-parity-tag"] == "matrix", name


#: What Django's stack adds to a request that carried no body at all, and what it puts in
#: them. Neither value was sent by anything: ``wsgiref.simple_server.WSGIServer``
#: seeds ``CONTENT_LENGTH=''`` into the base environ of every request it ever serves, and
#: its ``WSGIRequestHandler.get_environ`` sets ``CONTENT_TYPE`` unconditionally, falling
#: back to ``email.message``'s ``text/plain`` default when the client sent none.
WSGI_MANUFACTURED_HEADERS: Dict[str, str] = {"content-length": "", "content-type": "text/plain"}


@scenarios()
def test_only_django_reports_headers_the_client_never_sent_and_only_the_wsgi_pair(
    matrix: Matrix, scenario: Scenario
) -> None:
    """A named divergence in the transport, deliberately not "fixed" in the integration.

    On a request with no body, the Django row carries two headers the client never sent.
    They are not signledger's: ``django.http.HttpHeaders`` treats ``CONTENT_TYPE`` and
    ``CONTENT_LENGTH`` as unprefixed headers and surfaces them from the WSGI environ
    whenever they are in it, and ``django.core.servers.basehttp`` - the server this suite
    is required to drive Django with - always puts them there. ``content-length: ""`` is
    the giveaway: no HTTP client can send an empty ``Content-Length``, so that value can
    only have been manufactured below the application. See
    :func:`test_djangos_extra_headers_come_from_the_wsgi_environ_and_not_from_signledger`,
    which reproduces both of them with no signledger in the picture at all.

    Recording it is right either way - the integration's job is to write down the environ
    it was handed, not to second-guess which of its keys the server invented. What must
    not happen is a *third* name appearing, or Flask and FastAPI growing one, which is
    what this test would fail on."""
    sent = _client_header_names(scenario)
    extra = {
        name: {key: value for key, value in obs.entries[0]["headers"].items() if key not in sent}
        for name, obs in matrix.across(scenario.name).items()
    }
    assert extra["flask"] == {}, extra["flask"]
    assert extra["fastapi"] == {}, extra["fastapi"]
    assert extra["django"] == ({} if scenario.body is not None else WSGI_MANUFACTURED_HEADERS)


def test_djangos_extra_headers_come_from_the_wsgi_environ_and_not_from_signledger() -> None:
    """The proof for the divergence above, with signledger absent from the picture.

    If this ever stops holding, the test above is blaming the wrong layer and its
    docstring is a lie - which is the failure mode this repository has the most history
    of, so the claim is executed rather than asserted in prose."""
    from wsgiref.simple_server import WSGIServer

    from django.http.request import HttpHeaders

    # The empty ``Content-Length`` is written into every environ before a request is read.
    assert "env['CONTENT_LENGTH']=''" in inspect.getsource(WSGIServer.setup_environ)
    # And Django republishes exactly those two environ keys as request headers.
    seeded = HttpHeaders({"HTTP_ACCEPT": "*/*", "CONTENT_LENGTH": "", "CONTENT_TYPE": "text/plain"})
    assert {key.lower(): value for key, value in seeded.items()} == {
        "accept": "*/*",
        **WSGI_MANUFACTURED_HEADERS,
    }
    assert dict(HttpHeaders({"HTTP_ACCEPT": "*/*"})) == {"Accept": "*/*"}


@scenarios()
def test_a_body_carrying_request_records_identical_content_headers_everywhere(
    matrix: Matrix, scenario: Scenario
) -> None:
    """The other half of the divergence above. Once the client really does send
    ``Content-Type`` and ``Content-Length``, all three record the sent values and Django
    manufactures nothing - so the divergence is confined to fabrication and does not touch
    how a real header is handled."""
    if scenario.body is None:
        pytest.skip("this scenario sends no body, so it sends no content headers")
    recorded = {
        name: dict(obs.entries[0]["headers"]) for name, obs in matrix.across(scenario.name).items()
    }
    expected = {"content-type": "application/json", "content-length": str(len(POST_BODY))}
    for name, headers in recorded.items():
        assert {key: headers[key] for key in expected} == expected, name


# --------------------------------------------------------------------------------------
# Correlation: a promise and its settlement
# --------------------------------------------------------------------------------------


@scenarios()
def test_a_settlement_carries_the_request_id_of_the_entry_it_settles(
    matrix: Matrix, scenario: Scenario
) -> None:
    """The settlement is a second row about the same request. Without a shared
    ``request_id`` the two have one field in common - ``integration`` - and no auditor can
    tell they describe the same thing."""
    if KIND_COMPLETION not in scenario.kinds:
        pytest.skip("this scenario promises nothing, so there is nothing to settle")
    for name, obs in matrix.across(scenario.name).items():
        promises = [e for e in obs.entries if e["completion"] == COMPLETION_PROVISIONAL]
        settlements = [e for e in obs.entries if e["kind"] == KIND_COMPLETION]
        assert len(promises) == 1 and len(settlements) == 1, name
        assert settlements[0]["request_id"] == promises[0]["request_id"], name


@scenarios()
def test_no_provisional_entry_is_left_unsettled(matrix: Matrix, scenario: Scenario) -> None:
    """A promise nobody ever settles is a worse record than the overclaim it replaced.

    This is the failure mode ``will_send_body`` exists for: an entry written provisional
    for a response whose body no server will ever consume waits forever."""
    for name, obs in matrix.across(scenario.name).items():
        settled = {e["request_id"] for e in obs.entries if e["kind"] == KIND_COMPLETION}
        unsettled = [
            e["request_id"]
            for e in obs.entries
            if e["completion"] == COMPLETION_PROVISIONAL and e["request_id"] not in settled
        ]
        assert unsettled == [], f"{name} left {len(unsettled)} promise(s) unsettled"


@scenarios()
def test_no_settlement_is_an_orphan(matrix: Matrix, scenario: Scenario) -> None:
    """A settlement whose ``request_id`` has no request entry is a claim about a request
    the ledger has no record of."""
    for name, obs in matrix.across(scenario.name).items():
        requested = {e["request_id"] for e in obs.entries if e["kind"] == KIND_REQUEST}
        orphans = [
            e["request_id"]
            for e in obs.entries
            if e["kind"] == KIND_COMPLETION and e["request_id"] not in requested
        ]
        assert orphans == [], f"{name} wrote {len(orphans)} orphan settlement(s)"


def test_no_promise_anywhere_in_any_ledger_is_left_unsettled(matrix: Matrix) -> None:
    """The same question asked of the whole chain rather than one scenario, so a promise
    settled by *another* scenario's traffic cannot hide one that was not."""
    for name, ledger in matrix.ledgers.items():
        rows = [entry.data for entry in ledger.iter_entries()]
        settled = {row["request_id"] for row in rows if row["kind"] == KIND_COMPLETION}
        unsettled = [
            row["path"]
            for row in rows
            if row["completion"] == COMPLETION_PROVISIONAL and row["request_id"] not in settled
        ]
        assert unsettled == [], f"{name} left promises unsettled at: {unsettled}"


def test_the_whole_ledger_of_every_framework_still_verifies(matrix: Matrix) -> None:
    """Everything above reads entries; this is the one that says they are trustworthy.

    A middleware that writes from a server's request thread while the response is being
    torn down is exactly where a chain gets forked, and a parity suite that compared
    payloads while the chain was broken would be proving the wrong thing."""
    for name, ledger in matrix.ledgers.items():
        report = ledger.check_integrity()
        assert report.ok, f"{name}: {[failure.message for failure in report.failures]}"
        assert report.entries_verified == ledger.count()


# --------------------------------------------------------------------------------------
# The divergence that is deliberate
# --------------------------------------------------------------------------------------


def test_fastapi_records_a_body_its_endpoint_never_read_as_omitted(matrix: Matrix) -> None:
    """Pinned as documented behaviour, not filed as a defect.

    FastAPI's middleware is pure ASGI and observes the request body *as the endpoint reads
    it off* ``receive``. Anything that answers before reading - a 404, a 405, a 401 - has
    therefore been given no body to record, and it says so with ``[OMITTED:not-read]``
    while Flask and Django, which read the body up front, record the redacted body.

    Making the ASGI middleware drain the stream itself would fix the divergence and break
    something worse: it changes ASGI semantics for every downstream application, and it
    can hang against a client that announced a body it never sends. The record is honest
    either way - both spellings say what was and was not seen - so the divergence stays,
    and this test is what makes it a decision rather than an accident.

    If this test fails because FastAPI now captures the body, that is a change to ASGI
    behaviour and needs to be argued for, not merely observed."""
    across = matrix.across("method-not-allowed-405")
    assert across["fastapi"].entries[0]["body"] == NOT_READ
    assert across["flask"].entries[0]["body"] == CAPTURED_BODY
    assert across["django"].entries[0]["body"] == CAPTURED_BODY


def test_all_three_capture_the_same_redacted_body_when_the_endpoint_does_read_it(
    matrix: Matrix,
) -> None:
    """The other half of the divergence above: once the endpoint reads the body, the three
    record byte-identical redacted payloads, secrets included."""
    bodies = {
        name: obs.entries[0]["body"] for name, obs in matrix.across("post-with-a-json-body").items()
    }
    assert bodies == dict.fromkeys(FRAMEWORKS, CAPTURED_BODY)
    assert bodies["flask"]["password"] == REDACTED


# --------------------------------------------------------------------------------------
# The four divergences that shipped, each named by the test that would catch it again
# --------------------------------------------------------------------------------------


def test_a_head_request_to_a_streaming_route_is_one_final_entry_in_every_framework(
    matrix: Matrix,
) -> None:
    """Flask excluded ``HEAD`` and FastAPI tested the status alone, so byte-identical
    traffic produced one final entry on Flask and two entries with a provisional first on
    FastAPI. No server consumes the body of a ``HEAD`` response, so the FastAPI promise
    could never be settled."""
    for name, obs in matrix.across("head-on-a-streaming-route").items():
        assert [e["kind"] for e in obs.entries] == [KIND_REQUEST], name
        assert obs.entries[0]["completion"] == COMPLETION_FINAL, name


def test_a_204_is_one_final_entry_in_every_framework(matrix: Matrix) -> None:
    """A 204 has no body for any server to consume, so recording it provisional would
    leave a promise waiting on a settlement that cannot happen."""
    for name, obs in matrix.across("no-content-204").items():
        assert [e["kind"] for e in obs.entries] == [KIND_REQUEST], name
        assert obs.entries[0]["completion"] == COMPLETION_FINAL, name


def test_an_error_response_is_final_and_settles_nothing_in_every_framework(
    matrix: Matrix,
) -> None:
    """Werkzeug's ``Response.is_streamed`` is "``len(self.response)`` raises", and
    ``make_response`` runs every ``HTTPException`` through ``force_type``, which replaces
    the body with a ``ClosingIterator`` that has no ``__len__``. Every 404, 405 and 500
    therefore looked streamed on Flask: recorded provisional, then given a second
    settlement entry saying it completed. Two signed rows per error response, from any
    anonymous caller hitting bad URLs, against a store that by design cannot be pruned."""
    for name in ("unrouted-404", "method-not-allowed-405", "view-raises-500"):
        for framework, obs in matrix.across(name).items():
            assert len(obs.entries) == 1, f"{framework}/{name}"
            assert obs.entries[0]["completion"] == COMPLETION_FINAL, f"{framework}/{name}"


def test_a_500_names_the_exception_type_in_every_framework(matrix: Matrix) -> None:
    """``after_request`` and Django's exception middleware are both handed a *response*
    and no exception, so a handled 500 was recorded with no ``error`` field at all on both
    - and ``WHERE error IS NOT NULL`` came back structurally empty on a service whose
    views were failing."""
    for name, obs in matrix.across("view-raises-500").items():
        assert obs.entries[0]["error"] == "RuntimeError", name
        assert obs.entries[0]["status"] == 500, name


def test_a_stream_that_dies_mid_body_settles_incomplete_and_names_the_cause_everywhere(
    matrix: Matrix,
) -> None:
    """The scenario every in-process test client is blind to. werkzeug's test client does
    not close the response iterable, and closing it is the only moment a streamed
    response's outcome becomes knowable - so a generator that raised on its second chunk
    sat permanently in the ledger as a plain 200."""
    for name, obs in matrix.across("stream-raises-after-first-chunk").items():
        assert [e["kind"] for e in obs.entries] == [KIND_REQUEST, KIND_COMPLETION], name
        settlement = obs.entries[1]
        assert settlement["outcome"] == OUTCOME_INCOMPLETE, name
        assert settlement["error"] == "RuntimeError", name
        assert settlement["completion"] == COMPLETION_FINAL, name
        # The client saw the status line and the first chunk before it died.
        assert obs.reply.status == 200, name
        assert CHUNKS[0] in obs.reply.body, name
