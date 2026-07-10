#include "mmpr/AudioTransportQos.h"

#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;

void check(bool value, const char* label) {
  if (!value) {
    ++failures;
    std::printf("FAIL %s\n", label);
  }
}

mmpr::AuxiliaryTransportAdmissionInput healthyInput() {
  mmpr::AuxiliaryTransportAdmissionInput input;
  input.nowMs = 10'000;
  input.queueHighWaterPackets = 8;
  input.queueLowWaterPackets = 3;
  input.packetAgeLimitMs = 180;
  input.recoveryCooldownMs = 1'500;
  return input;
}

}  // namespace

int main() {
  auto input = healthyInput();
  check(!mmpr::shouldDeferAuxiliaryTransport(input), "healthy audio admits BLE report");

  input.audioPublishInProgress = true;
  check(mmpr::shouldDeferAuxiliaryTransport(input), "active audio publish defers BLE report");

  input = healthyInput();
  input.queuedAudioPackets = 8;
  check(mmpr::shouldDeferAuxiliaryTransport(input), "high-water queue defers BLE report");

  input = healthyInput();
  input.newestAudioPacketAgeUs = 180'000;
  check(mmpr::shouldDeferAuxiliaryTransport(input), "old audio packet defers BLE report");

  input = healthyInput();
  input.hasRecentAudioTransportFailure = true;
  input.lastAudioTransportFailureMs = 9'000;
  check(mmpr::shouldDeferAuxiliaryTransport(input), "recovery cooldown defers BLE report");

  input.nowMs = 11'000;
  check(!mmpr::shouldDeferAuxiliaryTransport(input), "recovered audio admits BLE report");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
