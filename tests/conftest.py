"""
Test bootstrap: make the repo root importable and locate the harness fixtures.

`afh` itself is a pip dependency installed from GitHub (scripts/install_afh.sh). The
fixtures the tests compare against are not in the wheel; they are read from the pinned
checkout the same script leaves under .deps/alpamayo-faithfulness (override with AFH_REPO).
`afh_fixture_dir` is None when that checkout is absent and the fixture tests skip.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AFH_CHECKOUT = os.environ.get("AFH_REPO", os.path.join(ROOT, ".deps", "alpamayo-faithfulness"))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def afh_fixture_dir():
    path = os.path.join(AFH_CHECKOUT, "fixtures")
    return path if os.path.isdir(path) else None


def _pinned_checkout(name, env):
    path = os.environ.get(env, os.path.join(ROOT, ".deps", name))
    return path if os.path.isdir(os.path.join(path, ".git")) else None


@pytest.fixture(scope="session")
def edge_src():
    """Pinned source checkout of NVlabs/alpamayo2-edge (scripts/fetch_edge_sources.sh), or None."""
    return _pinned_checkout("alpamayo2-edge", "EDGE_REPO")


@pytest.fixture(scope="session")
def recipes_src():
    """Pinned source checkout of NVlabs/alpamayo-recipes (scripts/fetch_edge_sources.sh), or None."""
    return _pinned_checkout("alpamayo-recipes", "RECIPES_REPO")
