#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/GpsPpsTimerCapture.h"
#include "mmpr/Types.h"

namespace mmpr {

struct PacketTimingDiagnostics {
  bool hasGpsAnchor = false;
  uint32_t ppsEdgeCount = 0;
  uint32_t dmaRingSlotIndex = 0;
  int64_t ppsPhaseErrorNs = 0;
  double estimatedPpm = 0.0;
};

class NodeClock {
 public:
  void begin(
      uint32_t sampleRateHz,
      size_t captureBlockSamples,
      bool syncNtp,
      const char* ntpServer = "pool.ntp.org",
      long gmtOffsetSeconds = 0,
      int daylightOffsetSeconds = 0);

  void observeCapturedAudio(const AudioCaptureTimestamp& captureTimestamp);
  uint64_t utcAtSampleIndex(uint64_t sampleIndex) const;
  uint64_t sampleIndexAtUtcNs(uint64_t utcNs) const;
  uint64_t nowUtcNs() const;
  uint64_t utcAtMonotonicUs(uint64_t monotonicUs) const;
  TimeQuality timeQuality() const;
  double estimatedPpm() const { return filteredFrequencyPpm_; }
  bool currentPacketTimingDiagnostics(PacketTimingDiagnostics& outDiagnostics) const;

  void applyNtpObservation(uint64_t utcNs, uint64_t monotonicUs, uint64_t roundTripUs);
  void applyGpsPpsObservation(uint64_t utcSecondNs, const GpsPpsCaptureEvent& ppsEvent);

 private:
  static constexpr uint64_t kNsPerSecond = 1000000000ULL;
  static constexpr uint64_t kUsPerSecond = 1000000ULL;
  static constexpr uint64_t kGpsFreshUs = 1500000ULL;
  static constexpr uint64_t kGpsHoldoverUs = 60000000ULL;
  static constexpr uint64_t kNtpFreshUs = 7200000000ULL;
  static constexpr uint32_t kGpsStablePulseTarget = 3;

  double estimateSamplePositionAtMonotonicUs(uint64_t monotonicUs) const;
  uint64_t utcAtSamplePosition(double samplePosition) const;
  void setSampleAnchor(double samplePosition, uint64_t utcNs);
  void setWallReference(uint64_t utcNs, uint64_t monotonicUs);
  void updateEffectiveSamplePeriod(double localToUtcScale, bool preferImmediateLock);
  bool gpsDisciplineAvailable(uint64_t monotonicUs) const;

  uint32_t sampleRateHz_ = 0;
  size_t captureBlockSamples_ = 0;

  uint64_t bootMonotonicUs_ = 0;
  double nominalSamplePeriodNs_ = 0.0;
  double nominalSamplePeriodUs_ = 0.0;
  double effectiveSamplePeriodNs_ = 0.0;
  double localToUtcScale_ = 1.0;
  double filteredFrequencyPpm_ = 0.0;

  uint64_t wallReferenceUtcNs_ = 0;
  uint64_t wallReferenceMonotonicUs_ = 0;

  uint64_t lastObservedStartSampleIndex_ = 0;
  uint64_t lastObservedEndSampleIndex_ = 0;
  uint64_t lastObservedBlockStartMonotonicUs_ = 0;
  uint64_t lastObservedBlockEndMonotonicUs_ = 0;
  uint64_t lastObservedCompletedBlockCount_ = 0;
  uint32_t lastObservedDmaRingSlotIndex_ = 0;
  bool haveSampleObservation_ = false;

  uint64_t anchorWallNs_ = 0;
  double anchorSamplePosition_ = 0.0;
  bool haveSampleAnchor_ = false;

  bool haveNtpObservation_ = false;
  uint64_t previousNtpUtcNs_ = 0;
  uint64_t previousNtpMonotonicUs_ = 0;
  uint64_t lastNtpSyncMonotonicUs_ = 0;

  bool haveGpsInterval_ = false;
  uint64_t previousGpsMonotonicUs_ = 0;
  uint64_t lastGpsSyncMonotonicUs_ = 0;
  uint32_t gpsStablePulseCount_ = 0;
  uint32_t latestPpsEdgeCount_ = 0;
  int64_t latestPpsPhaseErrorNs_ = 0;
};

}  // namespace mmpr
