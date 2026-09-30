#!/usr/bin/env bash
# install_afh.sh — install the afh harness (oussndou77/alpamayo-faithfulness) from GitHub.
#
# The harness is a dependency of this repo, never vendored (ROADMAP §2, principle 5). This
# script is the single place where its version is pinned.
#
# It does two things:
#   1. a pinned checkout under .deps/alpamayo-faithfulness (git-ignored): the wheel does not
#      ship the fixtures/ the cold tests compare against, nor the GPU runners of the harness;
#   2. `pip install` of that checkout.
#
# Known upstream issue (2026-09-30, commit 4604031): the harness's pyproject.toml has no
# package discovery section, so setuptools refuses the flat layout
# ("Multiple top-level packages discovered in a flat-layout: ['afh', 'runners', 'fixtures']")
# and `pip install git+https://…` fails. The two-line fix upstream is
#     [tool.setuptools.packages.find]
#     include = ["afh*"]
# Until it lands, this script falls back to putting the checkout on PYTHONPATH (tests do
# this automatically through tests/conftest.py).
set -euo pipefail

AFH_REPO_URL=${AFH_REPO_URL:-https://github.com/oussndou77/alpamayo-faithfulness.git}
AFH_REF=${AFH_REF:-4604031fa520f446d5fd39cc2af57030897ddc6c}
AFH_DEPS_DIR=${AFH_DEPS_DIR:-.deps}
CHECKOUT="$AFH_DEPS_DIR/alpamayo-faithfulness"
PYTHON=${PYTHON:-python}

mkdir -p "$AFH_DEPS_DIR"
if [ ! -d "$CHECKOUT/.git" ]; then
  git clone --quiet "$AFH_REPO_URL" "$CHECKOUT"
fi
git -C "$CHECKOUT" fetch --quiet origin
git -C "$CHECKOUT" checkout --quiet "$AFH_REF"
echo "afh checkout: $CHECKOUT @ $(git -C "$CHECKOUT" rev-parse --short HEAD)"

if "$PYTHON" -m pip install --quiet "$CHECKOUT" 2>/dev/null; then
  echo "afh installed with pip:"
  "$PYTHON" -c "import afh, os; print('  ', os.path.dirname(afh.__file__))"
else
  echo "pip could not build the harness wheel (upstream pyproject lacks package discovery)."
  echo "Fallback: add the checkout to PYTHONPATH in every shell that runs a runner:"
  echo "    export PYTHONPATH=\"$(cd "$CHECKOUT" && pwd):\${PYTHONPATH:-}\""
  echo "(tests/conftest.py already does this for the cold tests)"
fi
