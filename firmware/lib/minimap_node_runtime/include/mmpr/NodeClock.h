#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/ClockHoldoverModel.h"
#include "mmpr/GpsPpsTimerCapture.h"
#include "mmpr/Types.h"

namespace mmpr {

struct PacketTimingDiagnostics {
  bool hasGpsAnchor = false;
  uint32_t ppsEdgeCount = 0;
  uint32_t dmaRingSlotIndex = 0;
  int64_t ppsPhaseErrorNs = 0;
  double estimatedPpm = 0.0;
  // Clock holdover diagnostics (telemetry only; see ClockHoldoverModel.h).
  bool holdoverActive = false;
  uint32_t holdoverAgeMs = 0;
  uint32_t predictedErrorNs = 0;
  bool ltValid = false;
  double ltPpm = 0.0;
  double ltSigmaPpm = 0.0;
  bool tempModelValid = false;
  bool tempCompApplied = false;
  double tempSlopePpmPerC = 0.0;
  double tempResidRmsPpm = 0.0;
};

// Holdover / crystal-model configuration. Defaults are legacy-equivalent: at
// 1 ppm drift uncertainty the 60000 ns error budget yields exactly the historic
// fixed 60 s window. NodeClock deliberately does NOT include node_config.h;
// the firmware main wires these from nodecfg constants via setHoldoverConfig().
struct NodeClockHoldoverConfig {
  bool enableTempComp = false;
  double driftUncertaintyPpm = 1.0;
  uint64_t errorBudgetNs = 60000ULL;
  uint64_t maxAgeUs = 900000000ULL;
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
  void applyNmeaUtcObservation(uint64_t utcNs, uint64_t monotonicUs);
  void applyGpsPpsObservation(uint64_t utcSecondNs, const GpsPpsCaptureEvent& ppsEvent);

  // Configure the holdover / crystal-frequency model. Legacy-equivalent by
  // default; the firmware main injects nodecfg values so NodeClock stays free
  // of node_config.h. Safe to call before begin().
  void setHoldoverConfig(const NodeClockHoldoverConfig& config) { holdoverConfig_ = config; }

  // Feed the latest on-node temperature sample (SHT45, ~2 s cadence). Cheap:
  // three field stores, no math. Consumed by the shadow temperature model at
  // the <=1 Hz LTFM median cadence.
  void observeTemperature(double temperatureC, uint64_t monotonicUs);

  // Slow-cadence maintenance run from refreshSlowTelemetry (NOT per loop). In
  // the common PPS-fresh case this fast-exits on a couple of integer compares.
  void maintain(uint64_t nowMonotonicUs);

 private:
  static constexpr uint64_t kNsPerSecond = 1000000000ULL;
  static constexpr uint64_t kUsPerSecond = 1000000ULL;
  static constexpr uint64_t kGpsFreshUs = 1500000ULL;
  static constexpr uint64_t kGpsHoldoverUs = 60000000ULL;
  static constexpr uint64_t kNtpFreshUs = 7200000000ULL;
  static constexpr uint32_t kGpsStablePulseTarget = 3;
  static constexpr uint64_t kTemperatureFreshUs = 10000000ULL;  // 10 s

  double estimateSamplePositionAtMonotonicUs(uint64_t monotonicUs) const;
  uint64_t utcAtSamplePosition(double samplePosition) const;
  void setSampleAnchor(double samplePosition, uint64_t utcNs);
  void setWallReference(uint64_t utcNs, uint64_t monotonicUs, const char* source = nullptr);
  void updateEffectiveSamplePeriod(double localToUtcScale, bool preferImmediateLock);
  bool gpsDisciplineAvailable(uint64_t monotonicUs) const;
  bool withinHoldoverBudget(uint64_t ageUs) const;
  bool ageWithinHoldover(uint64_t ageUs) const;
  uint64_t predictedHoldoverErrorNs(uint64_t ageUs) const;
  void enterHoldover(uint64_t nowMonotonicUs, uint64_t ageUs);
  void maintainActiveHoldover(uint64_t nowMonotonicUs);

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

  bool haveNmeaObservation_ = false;
  uint64_t previousNmeaUtcNs_ = 0;
  uint64_t previousNmeaMonotonicUs_ = 0;

  bool haveGpsInterval_ = false;
  uint64_t previousGpsMonotonicUs_ = 0;
  uint64_t previousGpsMonotonicNs_ = 0;
  uint64_t lastGpsSyncMonotonicUs_ = 0;
  uint64_t lastGpsSyncMonotonicNs_ = 0;
  uint64_t lastRejectedGpsEpochLogMonotonicUs_ = 0;
  uint64_t lastRejectedNmeaEpochLogMonotonicUs_ = 0;
  uint32_t gpsStablePulseCount_ = 0;
  uint32_t latestPpsEdgeCount_ = 0;
  int64_t latestPpsPhaseErrorNs_ = 0;
  uint64_t lastIgnoredSnapshotLogMonotonicUs_ = 0;
  uint32_t ppsPhaseStatCount_ = 0;
  double ppsPhaseSumSqNs_ = 0.0;
  int64_t ppsPhaseMaxAbsNs_ = 0;

  // --- Holdover / crystal-frequency model (telemetry-only in Phase 2) ---
  LongTermFrequencyModel longTermModel_;
  TempFrequencyModel tempModel_;
  NodeClockHoldoverConfig holdoverConfig_;
  bool holdoverActive_ = false;
  uint32_t reacquireRejectCount_ = 0;
  bool haveTemperature_ = false;
  double latestTemperatureC_ = 0.0;
  uint64_t latestTemperatureMonotonicUs_ = 0;
  double holdoverEntryTempC_ = 0.0;
  uint64_t lastHoldoverLogMonotonicUs_ = 0;
  double lastMedianPpm_ = 0.0;  // most recent accepted LTFM median (shadow log)
};

}  // namespace mmpr
