#!/usr/bin/env bash
# Downloads an Eclipse Temurin JDK into field-app/.local-jdk (gitignored).
#
# Why: AGP 8.x / Gradle 8.14 cannot run on very new JDKs (the dev Mac only
# has Homebrew OpenJDK 27). Keeping a pinned major locally avoids touching
# the system JDK. scripts/gradlew_local.sh picks it up automatically.
set -euo pipefail

FIELD_APP_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_JDK_DIRECTORY="${FIELD_APP_DIRECTORY}/.local-jdk"
JDK_MAJOR_VERSION="${FIELD_APP_JDK_MAJOR_VERSION:-21}"

case "$(uname -s)" in
  Darwin) ADOPTIUM_OS=mac ;;
  Linux) ADOPTIUM_OS=linux ;;
  *) echo "unsupported host OS $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) ADOPTIUM_ARCHITECTURE=aarch64 ;;
  x86_64) ADOPTIUM_ARCHITECTURE=x64 ;;
  *) echo "unsupported host architecture $(uname -m)" >&2; exit 1 ;;
esac

if [[ -x "${LOCAL_JDK_DIRECTORY}/home/bin/java" ]]; then
  echo "local JDK already present: ${LOCAL_JDK_DIRECTORY}/home"
  exit 0
fi

ASSET_QUERY_URL="https://api.adoptium.net/v3/assets/latest/${JDK_MAJOR_VERSION}/hotspot?architecture=${ADOPTIUM_ARCHITECTURE}&image_type=jdk&os=${ADOPTIUM_OS}&vendor=eclipse"
read -r ARCHIVE_URL ARCHIVE_SHA256 < <(curl -fsSL "${ASSET_QUERY_URL}" | python3 -c \
  'import json,sys; package=json.load(sys.stdin)[0]["binary"]["package"]; print(package["link"], package["checksum"])')

DOWNLOAD_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${DOWNLOAD_DIRECTORY}"' EXIT
echo "downloading ${ARCHIVE_URL}"
curl -fsSL -o "${DOWNLOAD_DIRECTORY}/jdk.tar.gz" "${ARCHIVE_URL}"
echo "${ARCHIVE_SHA256}  ${DOWNLOAD_DIRECTORY}/jdk.tar.gz" | shasum -a 256 -c -

mkdir -p "${LOCAL_JDK_DIRECTORY}/extracted"
tar -xzf "${DOWNLOAD_DIRECTORY}/jdk.tar.gz" -C "${LOCAL_JDK_DIRECTORY}/extracted"
EXTRACTED_ROOT="$(find "${LOCAL_JDK_DIRECTORY}/extracted" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ -d "${EXTRACTED_ROOT}/Contents/Home" ]]; then
  ln -sfn "${EXTRACTED_ROOT}/Contents/Home" "${LOCAL_JDK_DIRECTORY}/home"
else
  ln -sfn "${EXTRACTED_ROOT}" "${LOCAL_JDK_DIRECTORY}/home"
fi
"${LOCAL_JDK_DIRECTORY}/home/bin/java" -version
