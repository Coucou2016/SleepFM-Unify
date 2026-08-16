"""CI-friendly test runner (avoids broken global pytest plugins on some hosts)."""

import os
import sys

import pytest


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pytest.main(["-q", "--tb=short", os.path.join(root, "tests")])


if __name__ == "__main__":
    sys.exit(main())
