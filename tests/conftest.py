"""
Test bootstrap: make the repo root importable and locate the harness fixtures.

`afh` itself is a pip dependency installed from GitHub (scripts/install_afh.sh). The
fixtures the tests compare against are not in the wheel; they are read from the pinned
checkout the same script leaves under .deps/alpamayo-faithfulness (override with AFH_REPO).
`afh_fixture_dir` is None when that checkout is absent and the fixture tests skip.
"""

import importlib.util
import os
import subprocess
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


def _git_head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _package_dir(name):
    """Directory of an importable top-level package, found without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


@pytest.fixture(scope="session")
def edge_source():
    """
    Where the alpamayo2_edge sources checked by tests/test_edge_pinned_api_cold.py come from.

    Default: the pinned checkout of NVlabs/alpamayo2-edge (scripts/fetch_edge_sources.sh),
    commit read from git; None when absent (the tests skip).
    EDGE_INSTALLED=1: the package installed in the current environment (the recipe venv on
    the pod), commit read from its PEP 610 record. A missing package is then a failure.
    """
    if os.environ.get("EDGE_INSTALLED"):
        pkg = _package_dir("alpamayo2_edge")
        if pkg is None:
            pytest.fail("EDGE_INSTALLED is set but alpamayo2_edge is not installed in this environment")
        from runners.probe_common import package_provenance
        dist = package_provenance("alpamayo2-edge")
        return {"mode": "installed", "pkg": pkg, "root": None, "commit": dist.get("commit"),
                "distribution": dist}
    root = _pinned_checkout("alpamayo2-edge", "EDGE_REPO")
    if root is None:
        return None
    return {"mode": "checkout", "pkg": os.path.join(root, "src", "alpamayo2_edge"), "root": root,
            "commit": _git_head(root), "distribution": None}


@pytest.fixture(scope="session")
def recipes_src():
    """Pinned git checkout of NVlabs/alpamayo-recipes: .deps/ by default, $RECIPES_REPO on the
    pod (the checkout the recipe environment was synced from). None when absent, except with
    EDGE_INSTALLED=1 where a missing checkout is a failure."""
    path = _pinned_checkout("alpamayo-recipes", "RECIPES_REPO")
    if path is None and os.environ.get("EDGE_INSTALLED"):
        pytest.fail("EDGE_INSTALLED is set but RECIPES_REPO is not a git checkout of alpamayo-recipes")
    return path


@pytest.fixture(scope="session")
def installed_recipe_packages():
    """Directories of the recipe packages as the current environment resolves them."""
    return {name: _package_dir(name) for name in ("alpamayo2edge_sft", "alpamayo")}


# --------------------------------------------------------------------------- --fail-on-skip

def pytest_addoption(parser):
    parser.addoption("--fail-on-skip", action="store_true", default=False,
                     help="exit with an error if any test is skipped: a check that did not run "
                          "must not look like a check that passed (docs/gpu_session_02.md, block 0)")


def _skipped(config):
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    return len(reporter.stats.get("skipped", [])) if reporter is not None else 0


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--fail-on-skip") and _skipped(config):
        terminalreporter.write_sep(
            "=", f"--fail-on-skip: {_skipped(config)} test(s) skipped, the check is INCOMPLETE", red=True)


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--fail-on-skip") and _skipped(session.config) and session.exitstatus == 0:
        session.exitstatus = 1
