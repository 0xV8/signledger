"""``python -m signledger`` - the same entry point as the console script.

Kept to a single delegation on purpose. Two entry points that each build their own
argument parser drift apart, and the one nobody runs in CI is the one that breaks.
"""

import sys

from signledger.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by subprocess, see tests
    sys.exit(main())
