#!/usr/bin/env bash
# Build the Leptos/WASM frontend and place the artifacts in minimappr/frontend/.
# Run this before `python -m build` to ensure the wheel includes the WASM bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT/minimappr-frontend"
trunk build --release

DIST="$REPO_ROOT/minimappr/frontend"
echo "Frontend built → $DIST"
ls -lh "$DIST/"

# Pre-publish check: fail if WASM is missing (would produce a broken wheel).
if ! ls "$DIST/"*.wasm 1>/dev/null 2>&1; then
  echo "ERROR: no .wasm file found in $DIST — aborting." >&2
  exit 1
fi
