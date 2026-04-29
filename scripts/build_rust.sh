#!/usr/bin/env bash
# Build MinimapPR Rust deliverables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_FRONTEND=0
BUILD_SIDECAR=0

if [[ $# -eq 0 ]]; then
  BUILD_FRONTEND=1
  BUILD_SIDECAR=1
fi

for arg in "$@"; do
  case "$arg" in
    --frontend) BUILD_FRONTEND=1 ;;
    --sidecar) BUILD_SIDECAR=1 ;;
    --all) BUILD_FRONTEND=1; BUILD_SIDECAR=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$BUILD_FRONTEND" -eq 1 ]]; then
  cd "$REPO_ROOT/minimappr-frontend"
  trunk build --release
  DIST="$REPO_ROOT/minimappr/frontend"
  echo "Frontend built -> $DIST"
  ls -lh "$DIST/"
  if ! ls "$DIST/"*.wasm 1>/dev/null 2>&1; then
    echo "ERROR: no .wasm file found in $DIST; aborting." >&2
    exit 1
  fi
fi

if [[ "$BUILD_SIDECAR" -eq 1 ]]; then
  cd "$REPO_ROOT/minimappr-ingest-sidecar"
  cargo build --release
  mkdir -p "$REPO_ROOT/dist"
  cp "target/release/minimappr-ingest-sidecar" "$REPO_ROOT/dist/minimappr-ingest-sidecar"
  echo "Ingest sidecar built -> $REPO_ROOT/dist/minimappr-ingest-sidecar"
  ls -lh "$REPO_ROOT/dist/minimappr-ingest-sidecar"
fi
