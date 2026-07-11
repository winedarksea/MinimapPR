#!/bin/sh
# MinimapPR bootstrapper: installs uv (if missing), then `uv tool install` for MinimapPR.
# Usage:
#   curl -LsSf https://minimappr.com/install.sh | sh
#   curl -LsSf https://minimappr.com/install.sh | sh -s -- --base   # skip full extras
set -eu

EXTRA="full"
for arg in "$@"; do
  case "$arg" in
    --base) EXTRA="" ;;
  esac
done

PACKAGE_SPEC="minimappr"
if [ -n "$EXTRA" ]; then
  PACKAGE_SPEC="minimappr[${EXTRA}]"
fi

# Interactive prompts must read from the controlling TTY, since stdin is the
# piped script itself when run as `curl ... | sh`.
TTY="/dev/tty"
if [ ! -r "$TTY" ]; then
  TTY=""
fi

info() { printf '==> %s\n' "$1"; }

prompt_yes_no() {
  # $1 = prompt text, default is "no"
  if [ -z "$TTY" ]; then
    echo "n"
    return
  fi
  reply=""
  # Guard the whole read (prompt + input) as an `if` condition so a TTY that
  # exists as a device node but has no controlling terminal (e.g. sandboxed/
  # non-interactive runs) degrades to the default answer instead of
  # aborting the script under `set -e`.
  if ! { printf '%s [y/N] ' "$1" > "$TTY"; read -r reply < "$TTY"; } 2>/dev/null; then
    echo "n"
    return
  fi
  case "$reply" in
    y|Y|yes|YES) echo "y" ;;
    *) echo "n" ;;
  esac
}

# ---------------------------------------------------------------------------
# 1. Install uv if missing
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  info "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv installs to ~/.local/bin (or $UV_INSTALL_DIR) by default; make sure it's
# on PATH for the rest of this script even if the current shell hasn't
# re-sourced its profile yet.
UV_BIN_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
case ":$PATH:" in
  *":$UV_BIN_DIR:"*) ;;
  *) PATH="$UV_BIN_DIR:$PATH" ;;
esac
export PATH

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv installation failed or uv is not on PATH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Install MinimapPR
# ---------------------------------------------------------------------------
info "Installing $PACKAGE_SPEC via uv tool install (this may take a while for the full extras)..."
uv tool install "$PACKAGE_SPEC"

TOOL_BIN_DIR="$(uv tool dir --bin 2>/dev/null || true)"
if [ -z "$TOOL_BIN_DIR" ]; then
  TOOL_BIN_DIR="$HOME/.local/bin"
fi
case ":$PATH:" in
  *":$TOOL_BIN_DIR:"*) ;;
  *) PATH="$TOOL_BIN_DIR:$PATH" ;;
esac
export PATH

if ! command -v minimappr >/dev/null 2>&1; then
  echo "warning: minimappr was installed but is not yet on PATH." >&2
  echo "         add $TOOL_BIN_DIR to your PATH, or restart your shell." >&2
fi

# ---------------------------------------------------------------------------
# 3. Optional desktop shortcut
# ---------------------------------------------------------------------------
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
MINIMAPPR_BIN="$(command -v minimappr 2>/dev/null || echo "$TOOL_BIN_DIR/minimappr")"

want_shortcut="$(prompt_yes_no 'Add a desktop shortcut?')"
if [ "$want_shortcut" = "y" ]; then
  case "$UNAME_S" in
    Linux)
      DESKTOP_DIR="$HOME/.local/share/applications"
      mkdir -p "$DESKTOP_DIR"
      DESKTOP_FILE="$DESKTOP_DIR/minimappr.desktop"
      cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=MinimapPR
Comment=Realtime environmental awareness: sound localization + classification + COP
Exec=sh -c '"$MINIMAPPR_BIN" & sleep 2 && xdg-open http://127.0.0.1:8080'
Icon=utilities-system-monitor
Terminal=true
Categories=Utility;
EOF
      chmod +x "$DESKTOP_FILE"
      info "Wrote $DESKTOP_FILE"
      ;;
    Darwin)
      APP_DIR="$HOME/Applications"
      mkdir -p "$APP_DIR"
      APP_FILE="$APP_DIR/MinimapPR.command"
      cat > "$APP_FILE" <<EOF
#!/bin/sh
"$MINIMAPPR_BIN" &
sleep 2
open http://127.0.0.1:8080
wait
EOF
      chmod +x "$APP_FILE"
      info "Wrote $APP_FILE"
      ;;
    *)
      echo "note: desktop shortcuts are only supported on Linux and macOS." >&2
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# 4. Optional launch now
# ---------------------------------------------------------------------------
want_launch="$(prompt_yes_no 'Launch MinimapPR now?')"
if [ "$want_launch" = "y" ]; then
  info "Starting MinimapPR..."
  "$MINIMAPPR_BIN" &
  sleep 2
  URL="http://127.0.0.1:8080"
  case "$UNAME_S" in
    Darwin) open "$URL" >/dev/null 2>&1 || true ;;
    Linux) xdg-open "$URL" >/dev/null 2>&1 || true ;;
  esac
  info "MinimapPR is running at $URL"
fi

# ---------------------------------------------------------------------------
# 5. Next steps
# ---------------------------------------------------------------------------
cat <<EOF

MinimapPR is installed.

  Run:        minimappr
  UI:         http://127.0.0.1:8080
  Uninstall:  uv tool uninstall minimappr

EOF
