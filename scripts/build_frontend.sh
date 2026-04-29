#!/usr/bin/env bash
# Compatibility wrapper for the historical frontend-only build command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/build_rust.sh" --frontend
