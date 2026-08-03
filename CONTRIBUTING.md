# Contributing

## The one rule

**Nothing is true here because it is written down.** This project's git history is largely
a record of discovering that its own documentation, docstrings and commit messages
described behaviour that had never existed: an integration shadowed by an empty package so
none of it ran, a `RedisBackend` in the README that was never implemented, a mutation gate
that reported success while checking zero mutants, a coverage figure of "100%" that was
measured over 96% of the statements.

So: **no claim ships without something that would fail if it were false.** Not a comment,
not a docstring, not a README line, not a commit message. If you cannot write a test that
fails when the claim is false, the claim does not go in.

Two corollaries that have each caught real defects here:

- **A green check is not proof the thing it names happened.** Two CI gates in this
  repository reported success while doing nothing at all. When you add one, follow the
  exit-code path end to end and prove it fails when it should.
- **A passing test is not proof of the behaviour it is named after.** Tests here have
  asserted the product's wrong answer, hand-passed values the product cannot produce, and
  simulated a server event in-process in a way that could not fail. Before writing a test,
  name the production change that would make it fail.

## Setup

```
python -m pip install -e ".[dev,flask,fastapi,django,postgresql,mongodb]"
python -m pytest -q
```

Roughly 645 tests skip without live PostgreSQL and MongoDB. That is a real gap in local
coverage, not noise — CI runs them against service containers. To close it locally, point
the suite at your own instances:

```
SIGNLEDGER_TEST_POSTGRES_DSN="postgresql://user@127.0.0.1:5432/signledger_test" \
SIGNLEDGER_TEST_MONGO_URI="mongodb://127.0.0.1:27017" \
python -m pytest -q
```

Before opening a pull request:

```
python -m pytest -q
ruff check .
mypy --strict -p signledger
bandit -q -c pyproject.toml -r signledger
```

## Working on this codebase

**Test first.** Write the failing test, run it, confirm it fails for the reason you expect,
then implement. A test written afterwards passes immediately, which proves nothing — you
never saw it catch anything.

**Drive real things.** No backend is mocked anywhere in this suite, and adversary tests
inflict damage directly on the rows the way an adversary would. Where behaviour depends on
a real server — a streamed body being closed, a client disconnecting — use a real socket.
An in-process shortcut is what let several of the defects in `git log` ship.

**Prove it on every backend.** There are four: `memory`, `sqlite`, `postgresql`, `mongodb`.
A behaviour proved on one is not proved.

**Write comments that say why.** The surrounding code explains reasoning and cites measured
evidence, not mechanics. Match it. If you are describing a defect, say what was measured.

**Correct the record in the same change.** If you find a docstring, README line or changelog
number that is false, fix it where you found it. Leaving it is how the next person inherits
it as fact.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md), which also
states plainly what each assurance level does and does not detect, and where the
mechanisms' limits are.

## Releasing

See [RELEASING.md](RELEASING.md). Publishing is automated from a tag and deliberately
cannot be done from a laptop.
