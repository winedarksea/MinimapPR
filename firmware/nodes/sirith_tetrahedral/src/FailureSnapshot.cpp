#include "mmpr/FailureSnapshot.h"

#include <cstdio>

#include "hardware/structs/watchdog.h"
#include "hardware/watchdog.h"

namespace mmpr {
namespace {

constexpr uint32_t kFailureSnapshotMagic = 0x4d4d5052u;  // "MMPR"
constexpr uint8_t kFailureSnapshotVersion = 1;
constexpr uint32_t kWatchdogTimeoutMs = 16000;

constexpr size_t kMagicScratchIndex = 0;
constexpr size_t kMetadataScratchIndex = 1;
constexpr size_t kBootCountScratchIndex = 2;
constexpr size_t kProgressScratchIndex = 3;

bool hasValidSnapshotHeader() {
  if (watchdog_hw->scratch[kMagicScratchIndex] != kFailureSnapshotMagic) {
    return false;
  }

  const uint32_t metadata = watchdog_hw->scratch[kMetadataScratchIndex];
  return static_cast<uint8_t>(metadata >> 24) == kFailureSnapshotVersion;
}

const char* phaseName(FatalLifecyclePhase phase) {
  switch (phase) {
    case FatalLifecyclePhase::kBootStart:
      return "boot_start";
    case FatalLifecyclePhase::kWiFiInit:
      return "wifi_init";
    case FatalLifecyclePhase::kPeripheralInit:
      return "peripheral_init";
    case FatalLifecyclePhase::kStartupWiFiConnect:
      return "startup_wifi_connect";
    case FatalLifecyclePhase::kRunnerStart:
      return "runner_start";
    case FatalLifecyclePhase::kNtpStart:
      return "ntp_start";
    case FatalLifecyclePhase::kMainLoop:
      return "main_loop";
    default:
      return "unknown";
  }
}

void writeSnapshotWords(uint32_t metadata, uint32_t bootCount, uint32_t progressMarker) {
  watchdog_hw->scratch[kMagicScratchIndex] = kFailureSnapshotMagic;
  watchdog_hw->scratch[kMetadataScratchIndex] = metadata;
  watchdog_hw->scratch[kBootCountScratchIndex] = bootCount;
  watchdog_hw->scratch[kProgressScratchIndex] = progressMarker;
}

}  // namespace

PreviousFatalFailureSnapshot FailureSnapshot::initializeForBoot() {
  PreviousFatalFailureSnapshot previous = {};

  if (watchdog_enable_caused_reboot() && hasValidSnapshotHeader()) {
    previous.valid = true;
    previous.bootCount = watchdog_hw->scratch[kBootCountScratchIndex];
    previous.phase = unpackPhase(watchdog_hw->scratch[kMetadataScratchIndex]);
    previous.progressMarker = watchdog_hw->scratch[kProgressScratchIndex];
  }

  const uint32_t nextBootCount = hasValidSnapshotHeader()
      ? (watchdog_hw->scratch[kBootCountScratchIndex] + 1u)
      : 1u;
  writeSnapshotWords(packMetadata(FatalLifecyclePhase::kBootStart), nextBootCount, 0u);
  watchdog_enable(kWatchdogTimeoutMs, false);
  watchdog_update();

  if (previous.valid) {
    std::printf(
        "[sirith-pico] previous fatal failure boot=%lu phase=%s progress=%lu reset=watchdog_timeout\n",
        static_cast<unsigned long>(previous.bootCount),
        phaseName(previous.phase),
        static_cast<unsigned long>(previous.progressMarker));
  }

  std::printf(
      "[sirith-pico] failure snapshot armed boot=%lu watchdog_timeout_ms=%lu\n",
      static_cast<unsigned long>(nextBootCount),
      static_cast<unsigned long>(kWatchdogTimeoutMs));

  return previous;
}

void FailureSnapshot::updatePhase(FatalLifecyclePhase phase) {
  writeSnapshotWords(
      packMetadata(phase),
      watchdog_hw->scratch[kBootCountScratchIndex],
      watchdog_hw->scratch[kProgressScratchIndex]);
}

void FailureSnapshot::updateProgressMarker(uint32_t progressMarker) {
  watchdog_hw->scratch[kProgressScratchIndex] = progressMarker;
}

void FailureSnapshot::feedWatchdog() {
  watchdog_update();
}

uint32_t FailureSnapshot::packMetadata(FatalLifecyclePhase phase) {
  return (static_cast<uint32_t>(kFailureSnapshotVersion) << 24) |
         (static_cast<uint32_t>(phase) << 16);
}

FatalLifecyclePhase FailureSnapshot::unpackPhase(uint32_t metadata) {
  return static_cast<FatalLifecyclePhase>((metadata >> 16) & 0xffu);
}

}  // namespace mmpr
