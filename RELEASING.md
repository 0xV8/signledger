# Releasing

This file exists because the last release went out without one. 1.0.0 was published
manually, documented features that did not exist, and is still installable today — which
is why several steps below are about refusing to publish rather than about publishing.

## What is automated and what is not

`.github/workflows/release.yml` runs on a `v*` tag. It refuses to publish unless the tag,
`signledger.__version__` and a dated `CHANGELOG.md` heading all agree, and unless the
tagged commit is an ancestor of the default branch. It then builds, inspects the artifacts
from the inside, installs the wheel into a clean interpreter and runs a real append and
verification, and only then publishes via PyPI trusted publishing.

Nothing publishes from a laptop. There is no API token in this repository, and there
should never be one.

## One-time setup

1. **Configure trusted publishing on PyPI.** Project `signledger` → Publishing → add a
   GitHub publisher: owner `0xV8`, repository `signledger`, workflow `release.yml`,
   environment `pypi`. Without this the `publish` job fails with an OIDC error.
2. **Create the `pypi` environment** in repository settings, and add required reviewers.
   That environment gate is the last human checkpoint before an artifact becomes
   permanent.
3. **Protect the default branch**, requiring the existing CI jobs — `test`, `os-matrix`,
   `multi-worker`, `imports`, `lint`, `package` — as status checks. Until this is on, the
   ancestry guard in `release.yml` is checking membership of a branch anyone can
   force-push.

## Releasing

1. **Land everything on the default branch through a pull request.** The release workflow
   refuses to publish a commit that is not an ancestor of it.
2. **Set the version** in `signledger/__init__.py`.
3. **Date the changelog heading.** `## [X.Y.Z] — unreleased` becomes
   `## [X.Y.Z] — YYYY-MM-DD`. The guard job greps for exactly that shape, and
   `tests/test_packaging.py` asserts it matches `__version__`, because `CHANGELOG.md` is
   grafted into the sdist and is the target of the `Changelog` project URL — publishing a
   version it does not mention ships a page saying the release being installed does not
   exist.
4. **Re-read the measured claims.** `README.md`, `CHANGELOG.md` and `docs/ARCHITECTURE.md`
   quote test counts, coverage percentages and benchmark numbers. Every one of them has
   been stale at some point in this project's history. Re-measure or delete them; do not
   carry them forward on trust.
5. **Tag and push.**

   ```
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

6. **Approve the `pypi` environment** when the workflow asks.
7. **Check the rendered project page.** Specifically that the sidebar links resolve —
   they are baked into the wheel metadata and point at paths on the default branch.

## Rehearsing

Run the workflow manually (`workflow_dispatch`). `guard` and `build` run in full;
`publish` and `github-release` are gated on the tag ref and are skipped. This is the way
to find a packaging problem without spending a version number on it.

## Never

- **Never `twine upload dist/*` from a checkout.** A local `dist/` is gitignored, so it can
  hold artifacts built from a tree that has since moved on. This project's own `dist/` once
  predated `HEAD` by 39 minutes and shipped an sdist containing the exact files the newest
  commit had fixed. The `package` CI job builds and discards for the same reason: only the
  release workflow's artifacts are ever published.
- **Never publish a version whose changelog entry says `unreleased`.**
- **Never reuse a version number.** PyPI will refuse it, and if it did not, a yanked
  version stays installable to anyone who pins it.

## Withdrawing a release

Yanking is not deletion. A yanked version is skipped by resolvers but still installs when
pinned exactly, so it is a signal, not a remedy.

1. Yank on PyPI, with a reason that names the replacement.
2. Say so in `SECURITY.md` with what a deployer should assume, and in `CHANGELOG.md`.
3. Publish the fixed version **first** where possible. A resolver with nothing to fall
   back to will keep the yanked release.
