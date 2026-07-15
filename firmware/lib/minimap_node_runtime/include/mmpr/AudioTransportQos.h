#pragma once

#include <cstdint>

namespace mmpr {

struct AuxiliaryTransportAdmissionInput {
  bool audioPublishInProgress = false;
  uint32_t queuedAudioPackets = 0;
  uint64_t newestAudioPacketAgeUs = 0;
  bool hasRecentAudioTransportFailure = false;
  uint32_t lastAudioTransportFailureMs = 0;
  uint32_t nowMs = 0;
  uint32_t queueHighWaterPackets = 0;
  uint32_t queueLowWaterPackets = 0;
  uint32_t packetAgeLimitMs = 0;
  uint32_t recoveryCooldownMs = 0;
};

inline bool shouldDeferAuxiliaryTransport(const AuxiliaryTransportAdmissionInput& input) {
  const bool queuePressured = input.queueHighWaterPackets > 0 &&
      input.queuedAudioPackets >= input.queueHighWaterPackets;
  const bool packetIsOld = input.packetAgeLimitMs > 0 &&
      input.newestAudioPacketAgeUs >= static_cast<uint64_t>(input.packetAgeLimitMs) * 1000ULL;
  const bool recovering = input.hasRecentAudioTransportFailure &&
      static_cast<int32_t>(input.nowMs - input.lastAudioTransportFailureMs) <
          static_cast<int32_t>(input.recoveryCooldownMs);
  if (input.audioPublishInProgress || queuePressured || packetIsOld || recovering) {
    return true;
  }
  return input.queueHighWaterPackets > input.queueLowWaterPackets &&
      input.queuedAudioPackets > input.queueLowWaterPackets;
}

}  // namespace mmpr
