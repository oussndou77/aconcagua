#!/usr/bin/env bash
# fetch_edge_sources.sh — pinned source checkouts of the two NVlabs repositories the Edge
# probes call (D-016), for the cold API check of tests/test_edge_pinned_api_cold.py.
#
# Nothing is installed and nothing is vendored: the checkouts land under .deps/ (git-ignored)
# and are read as text by the test, which verifies that every function, argument and config
# value the probes rely on exists at the pinned commits. The GPU environment itself is the
# recipe's `uv sync --locked` (docs/gpu_session_02.md), not this script.
set -euo pipefail

DEPS_DIR=${DEPS_DIR:-.deps}
EDGE_REF=${EDGE_REF:-02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e}        # NVlabs/alpamayo2-edge
RECIPES_REF=${RECIPES_REF:-622f9d23ff9e4961ba8c6aac71367776cdf6b016}  # NVlabs/alpamayo-recipes

fetch() {  # name url ref
  local dir="$DEPS_DIR/$1"
  mkdir -p "$DEPS_DIR"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$2" "$dir"
  fi
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$3"
  echo "$1 @ $(git -C "$dir" rev-parse --short HEAD)"
}

fetch alpamayo2-edge https://github.com/NVlabs/alpamayo2-edge.git "$EDGE_REF"
fetch alpamayo-recipes https://github.com/NVlabs/alpamayo-recipes.git "$RECIPES_REF"
