#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../minimappr-frontend"
trunk build --release
