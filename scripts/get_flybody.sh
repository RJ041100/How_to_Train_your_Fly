#!/usr/bin/env bash
# Fetch flybody's fly model (only flybody/fruitfly/assets) at the commit the results were produced with,
# into third_party/flybody, where walk/paths.py finds it without any environment variable.
#
#   bash scripts/get_flybody.sh            # -> third_party/flybody
#   bash scripts/get_flybody.sh /some/dir  # -> /some/dir (then: export FLYBODY_ASSETS=/some/dir/flybody/fruitfly/assets)
set -euo pipefail

COMMIT=d015e9bfe441bd90ae431bac24c55cb74bdbce26
FRUITFLY_SHA256=d14946fd0311025ecca70c8eeb5de80e1fe18700d3072be37ecbb18d33d80fd8
URL=https://github.com/TuragaLab/flybody.git

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/third_party/flybody}"
ASSETS="$DEST/flybody/fruitfly/assets"

if [ -f "$ASSETS/fruitfly.xml" ]; then
    echo "flybody assets already present: $ASSETS"
else
    if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
        echo "error: $DEST exists and is not empty, but has no flybody/fruitfly/assets/fruitfly.xml" >&2
        exit 1
    fi
    mkdir -p "$DEST"
    git -C "$DEST" init -q
    git -C "$DEST" remote add origin "$URL"
    git -C "$DEST" sparse-checkout set flybody/fruitfly/assets
    git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
    git -C "$DEST" checkout -q FETCH_HEAD
    echo "fetched flybody $COMMIT (fruitfly/assets only) -> $DEST"
fi

if command -v sha256sum >/dev/null; then
    got=$(sha256sum "$ASSETS/fruitfly.xml" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$ASSETS/fruitfly.xml" | cut -d' ' -f1)
fi
if [ "$got" != "$FRUITFLY_SHA256" ]; then
    echo "warning: fruitfly.xml sha256 $got differs from the one the results were produced with ($FRUITFLY_SHA256)" >&2
else
    echo "fruitfly.xml sha256 OK"
fi

if [ "$DEST" != "$ROOT/third_party/flybody" ]; then
    echo "now run: export FLYBODY_ASSETS=$ASSETS"
fi
