#!/usr/bin/env bash
# install_afh.sh — install the afh harness (oussndou77/alpamayo-faithfulness) from GitHub.
#
# The harness is a dependency of this repo, never vendored (ROADMAP §2, principle 5). This
# script is the single place where its version is pinned.
#
#   1. `pip install` straight from GitHub at the pinned commit (the wheel ships the `afh`
#      package only; since alpamayo-faithfulness PR #6 the pyproject declares its packages);
#   2. a pinned checkout under .deps/alpamayo-faithfulness (git-ignored), used only for the
#      fixtures/ the cold tests compare against — it is never put on sys.path.
set -euo pipefail

AFH_REPO_URL=${AFH_REPO_URL:-https://github.com/oussndou77/alpamayo-faithfulness.git}
AFH_REF=${AFH_REF:-7fff9e6}   # main after PRs #6 (packaging), #7 (camera ids), #8 (camera_indices required)
AFH_DEPS_DIR=${AFH_DEPS_DIR:-.deps}
CHECKOUT="$AFH_DEPS_DIR/alpamayo-faithfulness"
PYTHON=${PYTHON:-python}

"$PYTHON" -m pip install --quiet "alpamayo-faithfulness @ git+${AFH_REPO_URL}@${AFH_REF}"
"$PYTHON" -c "import afh, afh.cameras, os; print('afh installed:', os.path.dirname(afh.__file__))"

mkdir -p "$AFH_DEPS_DIR"
if [ ! -d "$CHECKOUT/.git" ]; then
  git clone --quiet "$AFH_REPO_URL" "$CHECKOUT"
fi
git -C "$CHECKOUT" fetch --quiet origin
git -C "$CHECKOUT" checkout --quiet "$AFH_REF"
echo "fixtures checkout: $CHECKOUT @ $(git -C "$CHECKOUT" rev-parse --short HEAD)"
