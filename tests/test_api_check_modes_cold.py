#!/usr/bin/env python3
"""
Cold tests of HOW the pinned-API check runs — the check itself is tests/test_edge_pinned_api_cold.py.

Block 0 of docs/gpu_session_02.md runs that check on the pod against the sources the probes
will really import, with --fail-on-skip. Two things must hold, and are exercised here by
running pytest in a subprocess:

  * a skipped check fails the command (--fail-on-skip), instead of looking like a pass;
  * EDGE_INSTALLED=1 reads the installed `alpamayo2_edge` package and its PEP 610 commit,
    requires the recipe packages to come from RECIPES_REPO, and never skips. The pod's
    environment is imitated with the pinned checkouts and a hand-made dist-info.

This file does not launch itself: it only targets the other test module.
"""

import json
import os
import subprocess
import sys

import pytest

import runners.probe_edge_common as ec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = "tests/test_edge_pinned_api_cold.py"


def _run(extra_env, *flags):
    env = {k: v for k, v in os.environ.items()
           if k not in ("EDGE_INSTALLED", "EDGE_REPO", "RECIPES_REPO", "PYTHONPATH")}
    env.update(extra_env)
    return subprocess.run([sys.executable, "-m", "pytest", TARGET, "-q", "-rs", "-p", "no:cacheprovider", *flags],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_a_skipped_check_fails_the_command_only_with_the_flag(tmp_path):
    nowhere = {"EDGE_REPO": str(tmp_path / "none"), "RECIPES_REPO": str(tmp_path / "none")}
    plain = _run(nowhere)
    assert plain.returncode == 0 and "9 skipped" in plain.stdout       # the trap: exit status 0
    strict = _run(nowhere, "--fail-on-skip")
    assert strict.returncode != 0, strict.stdout
    assert "--fail-on-skip: 9 test(s) skipped, the check is INCOMPLETE" in strict.stdout


def _fake_installation(tmp_path, edge, commit):
    """A directory that looks like the pod's site-packages for the two NVlabs packages."""
    site = tmp_path / "site"
    site.mkdir()
    os.symlink(os.path.join(edge, "src", "alpamayo2_edge"), site / "alpamayo2_edge")
    info = site / "alpamayo2_edge-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: alpamayo2-edge\nVersion: 0.1.0\n")
    (info / "direct_url.json").write_text(json.dumps({
        "url": "https://github.com/NVlabs/alpamayo2-edge.git",
        "vcs_info": {"vcs": "git", "commit_id": commit}}))
    return site


def _installed_env(site, recipes, with_recipe_packages=True):
    paths = [str(site)]
    if with_recipe_packages:
        os.symlink(os.path.join(recipes, "recipes", "alpamayo2edge_sft"), site / "alpamayo2edge_sft")
        paths.append(os.path.join(recipes, "src"))
    return {"EDGE_INSTALLED": "1", "RECIPES_REPO": recipes, "PYTHONPATH": os.pathsep.join(paths)}


@pytest.fixture()
def checkouts(edge_source, recipes_src):
    if edge_source is None or recipes_src is None or edge_source["mode"] != "checkout":
        pytest.skip("pinned checkouts absent: run scripts/fetch_edge_sources.sh")
    return edge_source["root"], recipes_src


def test_installed_mode_passes_on_the_pinned_installation(tmp_path, checkouts):
    edge, recipes = checkouts
    site = _fake_installation(tmp_path, edge, ec.EDGE_REPO_COMMIT)
    result = _run(_installed_env(site, recipes), "--fail-on-skip")
    assert result.returncode == 0, result.stdout[-3000:]
    assert "9 passed" in result.stdout and "skipped" not in result.stdout


def test_installed_mode_fails_on_another_commit(tmp_path, checkouts):
    edge, recipes = checkouts
    site = _fake_installation(tmp_path, edge, "0" * 40)
    result = _run(_installed_env(site, recipes), "--fail-on-skip")
    assert result.returncode != 0 and "1 failed" in result.stdout
    assert "test_sources_are_at_the_pinned_commits" in result.stdout


def test_installed_mode_fails_when_recipe_packages_come_from_elsewhere(tmp_path, checkouts):
    edge, recipes = checkouts
    site = _fake_installation(tmp_path, edge, ec.EDGE_REPO_COMMIT)
    result = _run(_installed_env(site, recipes, with_recipe_packages=False), "--fail-on-skip")
    assert result.returncode != 0 and "is not importable in this environment" in result.stdout


def test_installed_mode_never_skips_when_the_package_is_missing(tmp_path, checkouts):
    _, recipes = checkouts
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run({"EDGE_INSTALLED": "1", "RECIPES_REPO": recipes, "PYTHONPATH": str(empty)}, "--fail-on-skip")
    assert result.returncode != 0 and "skipped" not in result.stdout
    assert "alpamayo2_edge is not installed in this environment" in result.stdout
    # and a missing recipe checkout is a failure too, not a skip
    result = _run({"EDGE_INSTALLED": "1", "RECIPES_REPO": str(tmp_path / "none"), "PYTHONPATH": str(empty)})
    assert result.returncode != 0 and "skipped" not in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
