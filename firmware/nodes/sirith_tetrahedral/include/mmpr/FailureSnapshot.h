#pragma once

#include <cstdint>

namespace mmpr {

enum class FatalLifecyclePhase : uint8_t {
  kBootStart = 1,
  kWiFiInit = 2,
  kPeripheralInit = 3,
  kStartupWiFiConnect = 4,
  kRunnerStart = 5,
  kNtpStart = 6,
  kMainLoop = 7,
};

struct PreviousFatalFailureSnapshot {
  bool valid = false;
  uint32_t bootCount = 0;
  FatalLifecyclePhase phase = FatalLifecyclePhase::kBootStart;
  uint32_t progressMarker = 0;
};

class FailureSnapshot {
 public:
  static PreviousFatalFailureSnapshot initializeForBoot();
  static void updatePhase(FatalLifecyclePhase phase);
  static void updateProgressMarker(uint32_t progressMarker);
  static void feedWatchdog();

 private:
  static uint32_t packMetadata(FatalLifecyclePhase phase);
  static FatalLifecyclePhase unpackPhase(uint32_t metadata);
};

}  // namespace mmpr
