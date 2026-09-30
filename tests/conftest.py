"""
Test bootstrap: make the repo root importable and locate the afh harness.

`afh` is a dependency installed from GitHub (scripts/install_afh.sh). When the wheel build is
not possible (upstream pyproject without package discovery, see the script), the pinned
checkout under .deps/alpamayo-faithfulness is put on sys.path instead. Its fixtures are
exposed to the tests through `afh_fixture_dir` (None when the checkout is absent).
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AFH_CHECKOUT = os.environ.get("AFH_REPO", os.path.join(ROOT, ".deps", "alpamayo-faithfulness"))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if importlib.util.find_spec("afh") is None and os.path.isdir(os.path.join(AFH_CHECKOUT, "afh")):
    sys.path.insert(0, AFH_CHECKOUT)


@pytest.fixture(scope="session")
def afh_fixture_dir():
    path = os.path.join(AFH_CHECKOUT, "fixtures")
    return path if os.path.isdir(path) else None
