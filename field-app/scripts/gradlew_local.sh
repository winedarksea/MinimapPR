#!/usr/bin/env bash
# Runs ./gradlew with the local JDK (scripts/fetch_local_jdk.sh) and the
# default macOS Android SDK location, without editing user dotfiles.
# An explicit JAVA_HOME / ANDROID_HOME in the environment wins.
set -euo pipefail

FIELD_APP_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_JDK_HOME="${FIELD_APP_DIRECTORY}/.local-jdk/home"
DEFAULT_MACOS_ANDROID_SDK="${HOME}/Library/Android/sdk"

if [[ -z "${JAVA_HOME:-}" ]]; then
  if [[ ! -x "${LOCAL_JDK_HOME}/bin/java" ]]; then
    "${FIELD_APP_DIRECTORY}/scripts/fetch_local_jdk.sh"
  fi
  export JAVA_HOME="${LOCAL_JDK_HOME}"
fi
if [[ -z "${ANDROID_HOME:-}" && -d "${DEFAULT_MACOS_ANDROID_SDK}" ]]; then
  export ANDROID_HOME="${DEFAULT_MACOS_ANDROID_SDK}"
fi

cd "${FIELD_APP_DIRECTORY}"
exec ./gradlew "$@"
