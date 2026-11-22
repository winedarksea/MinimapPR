#!/usr/bin/env bash
# Downloads a pinned Eclipse Temurin JDK into field-app/.local-jdk (gitignored).
#
# Why: AGP 8.x / Gradle 8.14 cannot run on very new JDKs (the dev Mac only
# has Homebrew OpenJDK 27). Keeping a pinned JDK locally avoids touching the
# system JDK. scripts/gradlew_local.sh picks it up automatically.
#
# Supply chain: the release and each archive's SHA-256 are pinned here (taken
# from the Adoptium API for this release), and the archive is verified before
# it is extracted or run. To bump: update TEMURIN_RELEASE_* and all four
# checksums together from
#   https://api.adoptium.net/v3/assets/release_name/eclipse/jdk-<release>
set -euo pipefail

FIELD_APP_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_JDK_DIRECTORY="${FIELD_APP_DIRECTORY}/.local-jdk"

TEMURIN_RELEASE_TAG="jdk-21.0.13.1+2"
TEMURIN_RELEASE_TAG_URL_ENCODED="jdk-21.0.13.1%2B2"
TEMURIN_ARCHIVE_VERSION_SUFFIX="21.0.13.1_2"

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

case "${ADOPTIUM_OS}-${ADOPTIUM_ARCHITECTURE}" in
  mac-aarch64) PINNED_ARCHIVE_SHA256=48067f2b7c958ed104a9d8a5749be7d25220b0a356b4d2d1bcfaab3e16ad0b14 ;;
  mac-x64) PINNED_ARCHIVE_SHA256=1b5eff1741bb83d32e2e19aade71452bb42e40f047c01d3e178d438304c9a26d ;;
  linux-aarch64) PINNED_ARCHIVE_SHA256=8c72816a6b308842d5ed1f13e5826b69c567da4784f7238d7e800efa71fe54b1 ;;
  linux-x64) PINNED_ARCHIVE_SHA256=bb8a93f9ae6e7a693b2b721e85dbeff26ae1a77447985a8d7afcdd9c7cf6222e ;;
esac

if [[ -x "${LOCAL_JDK_DIRECTORY}/home/bin/java" ]]; then
  echo "local JDK already present: ${LOCAL_JDK_DIRECTORY}/home"
  exit 0
fi

ARCHIVE_NAME="OpenJDK21U-jdk_${ADOPTIUM_ARCHITECTURE}_${ADOPTIUM_OS}_hotspot_${TEMURIN_ARCHIVE_VERSION_SUFFIX}.tar.gz"
ARCHIVE_URL="https://github.com/adoptium/temurin21-binaries/releases/download/${TEMURIN_RELEASE_TAG_URL_ENCODED}/${ARCHIVE_NAME}"

DOWNLOAD_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "${DOWNLOAD_DIRECTORY}"' EXIT
echo "downloading ${TEMURIN_RELEASE_TAG} from ${ARCHIVE_URL}"
curl -fsSL --proto '=https' --tlsv1.2 -o "${DOWNLOAD_DIRECTORY}/jdk.tar.gz" "${ARCHIVE_URL}"
echo "${PINNED_ARCHIVE_SHA256}  ${DOWNLOAD_DIRECTORY}/jdk.tar.gz" | shasum -a 256 -c -

# Extract into a fresh directory so a previous partial extraction can never
# be picked up as the JDK root.
rm -rf "${LOCAL_JDK_DIRECTORY}/extracted"
mkdir -p "${LOCAL_JDK_DIRECTORY}/extracted"
tar -xzf "${DOWNLOAD_DIRECTORY}/jdk.tar.gz" -C "${LOCAL_JDK_DIRECTORY}/extracted"
EXTRACTED_ROOT="${LOCAL_JDK_DIRECTORY}/extracted/${TEMURIN_RELEASE_TAG}"
if [[ -d "${EXTRACTED_ROOT}/Contents/Home" ]]; then
  ln -sfn "${EXTRACTED_ROOT}/Contents/Home" "${LOCAL_JDK_DIRECTORY}/home"
else
  ln -sfn "${EXTRACTED_ROOT}" "${LOCAL_JDK_DIRECTORY}/home"
fi
"${LOCAL_JDK_DIRECTORY}/home/bin/java" -version
