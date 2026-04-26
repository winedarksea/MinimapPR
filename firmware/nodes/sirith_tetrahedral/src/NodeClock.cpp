#include "mmpr/NodeClock.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "pico/time.h"

#include "mmpr/BuildTimestamp.h"

namespace mmpr {
namespace {

constexpr double kPpmScale = 1000000.0;
constexpr double kNtpSlewAlpha = 0.15;
constexpr double kGpsSlewAlpha = 0.35;
constexpr uint64_t kMaxAcceptedNtpRttUs = 200000ULL;
constexpr uint64_t kMinimumPpsIntervalUs = 900000ULL;
constexpr uint64_t kMaximumPpsIntervalUs = 1100000ULL;
constexpr int64_t kLargeNtpPhaseCorrectionNs = 2LL * 1000000000LL;
constexpr int64_t kGpsNtpSanityLimitNs = 2LL * 1000000000LL;
constexpr uint64_t kRejectedGpsEpochLogIntervalUs = 10000000ULL;

double clampScale(double scale) {
  return std::clamp(scale, 0.9995, 1.0005);
}

double nsToPpm(double nominalNs, double effectiveNs) {
  if (nominalNs <= 0.0 || effectiveNs <= 0.0) {
    return 0.0;
  }
  return ((effectiveNs / nominalNs) - 1.0) * kPpmScale;
}

uint64_t addRoundedDeltaNs(uint64_t referenceNs, double deltaNs) {
  if (!std::isfinite(deltaNs)) {
    return referenceNs;
  }
  if (deltaNs >= 0.0) {
    const uint64_t roundedDeltaNs = static_cast<uint64_t>(std::llround(deltaNs));
    return UINT64_MAX - referenceNs < roundedDeltaNs ? UINT64_MAX : referenceNs + roundedDeltaNs;
  }

  const uint64_t roundedDeltaNs = static_cast<uint64_t>(std::llround(-deltaNs));
  return referenceNs > roundedDeltaNs ? referenceNs - roundedDeltaNs : 0ULL;
}

}  // namespace

void NodeClock::begin(
    uint32_t sampleRateHz,
    size_t captureBlockSamples,
    bool syncNtp,
    const char* ntpServer,
    long gmtOffsetSeconds,
    int daylightOffsetSeconds) {
  (void)syncNtp;
  (void)ntpServer;
  (void)gmtOffsetSeconds;
  (void)daylightOffsetSeconds;

  sampleRateHz_ = sampleRateHz;
  captureBlockSamples_ = captureBlockSamples;
  bootMonotonicUs_ = time_us_64();
  nominalSamplePeriodNs_ = kNsPerSecond / static_cast<double>(sampleRateHz_);
  nominalSamplePeriodUs_ = kUsPerSecond / static_cast<double>(sampleRateHz_);
  effectiveSamplePeriodNs_ = nominalSamplePeriodNs_;
  localToUtcScale_ = 1.0;

  wallReferenceUtcNs_ = kBuildEpochNs;
  wallReferenceMonotonicUs_ = bootMonotonicUs_;
  anchorWallNs_ = kBuildEpochNs;
  anchorSamplePosition_ = 0.0;
  haveSampleAnchor_ = true;

  haveSampleObservation_ = false;
  haveNtpObservation_ = false;
  haveGpsInterval_ = false;
  lastObservedStartSampleIndex_ = 0;
  lastObservedEndSampleIndex_ = 0;
  lastObservedBlockStartMonotonicUs_ = bootMonotonicUs_;
  lastObservedBlockEndMonotonicUs_ = bootMonotonicUs_;
  lastObservedCompletedBlockCount_ = 0;
  lastObservedDmaRingSlotIndex_ = 0;
  previousNtpUtcNs_ = 0;
  previousNtpMonotonicUs_ = 0;
  lastNtpSyncMonotonicUs_ = 0;
  haveNmeaObservation_ = false;
  previousNmeaUtcNs_ = 0;
  previousNmeaMonotonicUs_ = 0;
  previousGpsMonotonicUs_ = 0;
  previousGpsMonotonicNs_ = 0;
  lastGpsSyncMonotonicUs_ = 0;
  lastGpsSyncMonotonicNs_ = 0;
  lastRejectedGpsEpochLogMonotonicUs_ = 0;
  lastRejectedNmeaEpochLogMonotonicUs_ = 0;
  gpsStablePulseCount_ = 0;
  latestPpsEdgeCount_ = 0;
  latestPpsPhaseErrorNs_ = 0;
  filteredFrequencyPpm_ = 0.0;

  std::printf(
      "[node] clock boot anchor %llu ns; awaiting NTP or GPS discipline\n",
      static_cast<unsigned long long>(kBuildEpochNs));
}

void NodeClock::observeCapturedAudio(const AudioCaptureTimestamp& captureTimestamp) {
  lastObservedStartSampleIndex_ = captureTimestamp.startSampleIndex;
  lastObservedEndSampleIndex_ = captureTimestamp.endSampleIndex;
  lastObservedBlockStartMonotonicUs_ = captureTimestamp.blockStartMonotonicUs;
  lastObservedBlockEndMonotonicUs_ = captureTimestamp.blockEndMonotonicUs;
  lastObservedCompletedBlockCount_ = captureTimestamp.completedBlockCount;
  lastObservedDmaRingSlotIndex_ = captureTimestamp.dmaRingSlotIndex;
  haveSampleObservation_ = true;
}

uint64_t NodeClock::utcAtSampleIndex(uint64_t sampleIndex) const {
  if (!haveSampleAnchor_) {
    return 0;
  }
  return utcAtSamplePosition(static_cast<double>(sampleIndex));
}

uint64_t NodeClock::sampleIndexAtUtcNs(uint64_t utcNs) const {
  if (!haveSampleAnchor_ || effectiveSamplePeriodNs_ <= 0.0) {
    return 0;
  }

  const double samplePosition =
      anchorSamplePosition_ + ((static_cast<double>(utcNs) - static_cast<double>(anchorWallNs_)) / effectiveSamplePeriodNs_);
  return samplePosition <= 0.0 ? 0ULL : static_cast<uint64_t>(std::llround(samplePosition));
}

uint64_t NodeClock::nowUtcNs() const {
  return utcAtMonotonicUs(time_us_64());
}

uint64_t NodeClock::utcAtMonotonicUs(uint64_t monotonicUs) const {
  if (wallReferenceUtcNs_ > 0 && wallReferenceMonotonicUs_ > 0) {
    const int64_t deltaUs =
        static_cast<int64_t>(monotonicUs) - static_cast<int64_t>(wallReferenceMonotonicUs_);
    const double deltaNs = static_cast<double>(deltaUs) * 1000.0 * localToUtcScale_;
    return addRoundedDeltaNs(wallReferenceUtcNs_, deltaNs);
  }
  return utcAtSamplePosition(estimateSamplePositionAtMonotonicUs(monotonicUs));
}

TimeQuality NodeClock::timeQuality() const {
  const uint64_t nowUs = time_us_64();
  if (gpsDisciplineAvailable(nowUs)) {
    const uint64_t gpsAgeUs = nowUs - lastGpsSyncMonotonicUs_;
    if (gpsAgeUs <= kGpsFreshUs) {
      return TimeQuality::kGpsLocked;
    }
    if (gpsAgeUs <= kGpsHoldoverUs) {
      return TimeQuality::kGpsHoldover;
    }
  }

  if (lastNtpSyncMonotonicUs_ > 0 && (nowUs - lastNtpSyncMonotonicUs_) <= kNtpFreshUs) {
    return TimeQuality::kNtpDisciplined;
  }

  return TimeQuality::kFreeRunning;
}

void NodeClock::applyNtpObservation(uint64_t utcNs, uint64_t monotonicUs, uint64_t roundTripUs) {
  const bool gpsPpsIsFresh =
      gpsStablePulseCount_ >= kGpsStablePulseTarget &&
      lastGpsSyncMonotonicUs_ > 0 &&
      monotonicUs >= lastGpsSyncMonotonicUs_ &&
      (monotonicUs - lastGpsSyncMonotonicUs_) <= kGpsFreshUs;
  if (roundTripUs > kMaxAcceptedNtpRttUs) {
    std::printf(
        "[node] ignored NTP observation with high RTT=%llu us\n",
        static_cast<unsigned long long>(roundTripUs));
    return;
  }

  if (!haveSampleAnchor_) {
    setSampleAnchor(estimateSamplePositionAtMonotonicUs(monotonicUs), utcNs);
  }

  if (gpsPpsIsFresh) {
    const double samplePositionAtObservation = estimateSamplePositionAtMonotonicUs(monotonicUs);
    const uint64_t estimatedUtcNs = utcAtSamplePosition(samplePositionAtObservation);
    const int64_t phaseErrorNs =
        static_cast<int64_t>(utcNs) - static_cast<int64_t>(estimatedUtcNs);
    if (std::llabs(phaseErrorNs) <= kGpsNtpSanityLimitNs) {
      return;
    }
    haveGpsInterval_ = false;
    gpsStablePulseCount_ = 0;
    lastGpsSyncMonotonicUs_ = 0;
    lastGpsSyncMonotonicNs_ = 0;
    latestPpsPhaseErrorNs_ = phaseErrorNs;
    std::printf(
        "[node] NTP sanity check rejected GPS PPS epoch error_ns=%lld; recovering with NTP\n",
        static_cast<long long>(phaseErrorNs));
  }

  if (!haveNtpObservation_) {
    haveNtpObservation_ = true;
    previousNtpUtcNs_ = utcNs;
    previousNtpMonotonicUs_ = monotonicUs;
    lastNtpSyncMonotonicUs_ = monotonicUs;
    setSampleAnchor(estimateSamplePositionAtMonotonicUs(monotonicUs), utcNs);
    setWallReference(utcNs, monotonicUs, "ntp-first");
    std::printf("[node] first NTP anchor established\n");
    return;
  }

  if (monotonicUs <= previousNtpMonotonicUs_ || utcNs <= previousNtpUtcNs_) {
    return;
  }

  const double localElapsedNs = static_cast<double>(monotonicUs - previousNtpMonotonicUs_) * 1000.0;
  const double utcElapsedNs = static_cast<double>(utcNs - previousNtpUtcNs_);
  if (localElapsedNs <= 0.0 || utcElapsedNs <= 0.0) {
    return;
  }

  const double localToUtcScale = clampScale(utcElapsedNs / localElapsedNs);
  updateEffectiveSamplePeriod(localToUtcScale, false);
  previousNtpUtcNs_ = utcNs;
  previousNtpMonotonicUs_ = monotonicUs;
  lastNtpSyncMonotonicUs_ = monotonicUs;
  // Most refreshes should only discipline rate, but if the current phase is
  // still off by seconds then the node is effectively stuck on its boot/build
  // anchor. In that state, preserving continuity is less important than
  // converging to real UTC so the server can treat incoming audio as recent.
  const double samplePositionAtObservation = estimateSamplePositionAtMonotonicUs(monotonicUs);
  const uint64_t estimatedUtcNs = utcAtSamplePosition(samplePositionAtObservation);
  const int64_t phaseErrorNs =
      static_cast<int64_t>(utcNs) - static_cast<int64_t>(estimatedUtcNs);
  if (std::llabs(phaseErrorNs) >= kLargeNtpPhaseCorrectionNs) {
    setSampleAnchor(samplePositionAtObservation, utcNs);
    std::printf(
        "[node] applied large NTP phase correction error_ns=%lld\n",
        static_cast<long long>(phaseErrorNs));
  }
  setWallReference(utcNs, monotonicUs, "ntp");
}

void NodeClock::applyNmeaUtcObservation(uint64_t utcNs, uint64_t monotonicUs) {
  if (gpsDisciplineAvailable(monotonicUs)) {
    return;
  }

  const double samplePositionAtObservation = estimateSamplePositionAtMonotonicUs(monotonicUs);
  const uint64_t estimatedUtcNs = utcAtSamplePosition(samplePositionAtObservation);
  const int64_t phaseErrorNs =
      static_cast<int64_t>(utcNs) - static_cast<int64_t>(estimatedUtcNs);
  const bool ntpEpochIsFresh =
      lastNtpSyncMonotonicUs_ > 0 &&
      monotonicUs >= lastNtpSyncMonotonicUs_ &&
      (monotonicUs - lastNtpSyncMonotonicUs_) <= kNtpFreshUs;
  if (ntpEpochIsFresh && std::llabs(phaseErrorNs) > kGpsNtpSanityLimitNs) {
    if (lastRejectedNmeaEpochLogMonotonicUs_ == 0 ||
        monotonicUs < lastRejectedNmeaEpochLogMonotonicUs_ ||
        (monotonicUs - lastRejectedNmeaEpochLogMonotonicUs_) >= kRejectedGpsEpochLogIntervalUs) {
      lastRejectedNmeaEpochLogMonotonicUs_ = monotonicUs;
      std::printf(
          "[node] rejected NMEA UTC fallback error_ns=%lld while NTP epoch is fresh\n",
          static_cast<long long>(phaseErrorNs));
    }
    return;
  }

  if (!haveNmeaObservation_) {
    haveNmeaObservation_ = true;
    previousNmeaUtcNs_ = utcNs;
    previousNmeaMonotonicUs_ = monotonicUs;
    setSampleAnchor(samplePositionAtObservation, utcNs);
    setWallReference(utcNs, monotonicUs, "nmea-first");
    std::printf("[node] first NMEA UTC fallback anchor established\n");
    return;
  }

  if (monotonicUs <= previousNmeaMonotonicUs_ || utcNs <= previousNmeaUtcNs_) {
    return;
  }

  const double localElapsedNs = static_cast<double>(monotonicUs - previousNmeaMonotonicUs_) * 1000.0;
  const double utcElapsedNs = static_cast<double>(utcNs - previousNmeaUtcNs_);
  if (localElapsedNs <= 0.0 || utcElapsedNs <= 0.0) {
    return;
  }

  const double localToUtcScale = clampScale(utcElapsedNs / localElapsedNs);
  updateEffectiveSamplePeriod(localToUtcScale, false);
  previousNmeaUtcNs_ = utcNs;
  previousNmeaMonotonicUs_ = monotonicUs;
  // NMEA is a coarse fallback source with serial-delivery jitter. After the
  // first fallback anchor, keep phase continuity for small corrections. If
  // NTP has left the clock seconds away from GPS time, GPS must be allowed to
  // re-anchor so PPS can take over with the correct UTC label.
  if (std::llabs(phaseErrorNs) >= kLargeNtpPhaseCorrectionNs) {
    setSampleAnchor(samplePositionAtObservation, utcNs);
    std::printf(
        "[node] applied large NMEA UTC phase correction error_ns=%lld\n",
        static_cast<long long>(phaseErrorNs));
  }
  setWallReference(utcNs, monotonicUs, "nmea");
}

void NodeClock::applyGpsPpsObservation(uint64_t utcSecondNs, const GpsPpsCaptureEvent& ppsEvent) {
  const uint64_t monotonicUs = ppsEvent.monotonicUs;
  const uint64_t monotonicNs =
      ppsEvent.monotonicNs > 0 ? ppsEvent.monotonicNs : (monotonicUs * 1000ULL);
  const bool missedPpsEdges =
      latestPpsEdgeCount_ > 0 &&
      ppsEvent.edgeCount > latestPpsEdgeCount_ &&
      ppsEvent.edgeCount != (latestPpsEdgeCount_ + 1u);
  if (missedPpsEdges) {
    haveGpsInterval_ = false;
    gpsStablePulseCount_ = 0;
    latestPpsPhaseErrorNs_ = 0;
    std::printf(
        "[timing] PPS edge jump detected previous=%u current=%u; restarting GPS lock acquisition\n",
        static_cast<unsigned>(latestPpsEdgeCount_),
        static_cast<unsigned>(ppsEvent.edgeCount));
  }

  double samplePosition = estimateSamplePositionAtMonotonicUs(monotonicUs);
  if (ppsEvent.audioProducerSnapshot.valid) {
    const double samplePeriodNs =
        effectiveSamplePeriodNs_ > 0.0 ? effectiveSamplePeriodNs_ : nominalSamplePeriodNs_;
    const uint64_t snapshotMonotonicNs = ppsEvent.audioProducerSnapshot.capturedMonotonicUs * 1000ULL;
    const uint64_t snapshotDelayUs =
        snapshotMonotonicNs > monotonicNs
            ? ((snapshotMonotonicNs - monotonicNs) / 1000ULL)
            : 0ULL;
    const uint64_t snapshotDelayNs =
        snapshotMonotonicNs > monotonicNs ? (snapshotMonotonicNs - monotonicNs) : 0ULL;
    const double delaySamples = samplePeriodNs > 0.0
        ? (static_cast<double>(snapshotDelayNs) / samplePeriodNs)
        : 0.0;
    (void)snapshotDelayUs;
    samplePosition = std::max<double>(0.0, ppsEvent.audioProducerSnapshot.samplePosition - delaySamples);
    lastObservedCompletedBlockCount_ = ppsEvent.audioProducerSnapshot.completedBlockCount;
    lastObservedDmaRingSlotIndex_ = ppsEvent.audioProducerSnapshot.dmaRingSlotIndex;
  }
  const uint64_t gpsAgeUs =
      (lastGpsSyncMonotonicUs_ > 0 && monotonicUs >= lastGpsSyncMonotonicUs_)
          ? (monotonicUs - lastGpsSyncMonotonicUs_)
          : UINT64_MAX;
  const bool gpsWasFresh = gpsAgeUs <= kGpsHoldoverUs;

  if (haveGpsInterval_ && monotonicUs > previousGpsMonotonicUs_) {
    const uint64_t localIntervalNs =
        (monotonicNs > previousGpsMonotonicNs_ && previousGpsMonotonicNs_ > 0)
            ? (monotonicNs - previousGpsMonotonicNs_)
            : ((monotonicUs - previousGpsMonotonicUs_) * 1000ULL);
    const uint64_t localIntervalUs = localIntervalNs / 1000ULL;
    if (localIntervalUs < kMinimumPpsIntervalUs || localIntervalUs > kMaximumPpsIntervalUs) {
      haveGpsInterval_ = true;
      previousGpsMonotonicUs_ = monotonicUs;
      previousGpsMonotonicNs_ = monotonicNs;
      gpsStablePulseCount_ = 1;
      latestPpsEdgeCount_ = ppsEvent.edgeCount;
      latestPpsPhaseErrorNs_ = 0;
      std::printf(
          "[timing] PPS interval out of range interval_us=%llu edge=%u; waiting for stable 1 Hz PPS\n",
          static_cast<unsigned long long>(localIntervalUs),
          static_cast<unsigned>(ppsEvent.edgeCount));
      return;
    }

    const double localToUtcScale = clampScale(static_cast<double>(kNsPerSecond) / static_cast<double>(localIntervalNs));
    updateEffectiveSamplePeriod(localToUtcScale, true);
    gpsStablePulseCount_ = std::min<uint32_t>(gpsStablePulseCount_ + 1U, kGpsStablePulseTarget);
  } else {
    haveGpsInterval_ = true;
    gpsStablePulseCount_ = 1;
  }

  previousGpsMonotonicUs_ = monotonicUs;
  previousGpsMonotonicNs_ = monotonicNs;
  const uint64_t estimatedUtcAtPpsNs = utcAtSamplePosition(samplePosition);
  const int64_t gpsEpochErrorNs =
      static_cast<int64_t>(utcSecondNs) - static_cast<int64_t>(estimatedUtcAtPpsNs);
  const bool ntpEpochIsFresh =
      lastNtpSyncMonotonicUs_ > 0 &&
      monotonicUs >= lastNtpSyncMonotonicUs_ &&
      (monotonicUs - lastNtpSyncMonotonicUs_) <= kNtpFreshUs;
  if (ntpEpochIsFresh && std::llabs(gpsEpochErrorNs) > kGpsNtpSanityLimitNs) {
    latestPpsEdgeCount_ = ppsEvent.edgeCount;
    latestPpsPhaseErrorNs_ = gpsEpochErrorNs;
    lastGpsSyncMonotonicUs_ = 0;
    lastGpsSyncMonotonicNs_ = 0;
    if (lastRejectedGpsEpochLogMonotonicUs_ == 0 ||
        monotonicUs < lastRejectedGpsEpochLogMonotonicUs_ ||
        (monotonicUs - lastRejectedGpsEpochLogMonotonicUs_) >= kRejectedGpsEpochLogIntervalUs) {
      lastRejectedGpsEpochLogMonotonicUs_ = monotonicUs;
      std::printf(
          "[node] rejected GPS PPS UTC label error_ns=%lld while NTP epoch is fresh; keeping PPS cadence only\n",
          static_cast<long long>(gpsEpochErrorNs));
    }
    return;
  }

  lastGpsSyncMonotonicUs_ = monotonicUs;
  lastGpsSyncMonotonicNs_ = monotonicNs;
  const uint64_t anchorSampleIndex =
      samplePosition <= 0.0 ? 0ULL : static_cast<uint64_t>(std::llround(samplePosition));
  if (!gpsWasFresh || !haveSampleAnchor_ || gpsStablePulseCount_ == 1) {
    latestPpsPhaseErrorNs_ = 0;
    setSampleAnchor(samplePosition, utcSecondNs);
  } else {
    latestPpsPhaseErrorNs_ =
        static_cast<int64_t>(utcSecondNs) - static_cast<int64_t>(utcAtSampleIndex(anchorSampleIndex));
    setSampleAnchor(samplePosition, utcSecondNs);
  }
  latestPpsEdgeCount_ = ppsEvent.edgeCount;
  setWallReference(utcSecondNs, monotonicUs);
}

double NodeClock::estimateSamplePositionAtMonotonicUs(uint64_t monotonicUs) const {
  if (!haveSampleObservation_ || nominalSamplePeriodUs_ <= 0.0) {
    if (nominalSamplePeriodUs_ <= 0.0) {
      return 0.0;
    }
    const double elapsedUs = static_cast<double>(monotonicUs - bootMonotonicUs_);
    return elapsedUs / nominalSamplePeriodUs_;
  }

  const double deltaUs =
      static_cast<double>(static_cast<int64_t>(monotonicUs) - static_cast<int64_t>(lastObservedBlockStartMonotonicUs_));
  return static_cast<double>(lastObservedStartSampleIndex_) + (deltaUs / nominalSamplePeriodUs_);
}

uint64_t NodeClock::utcAtSamplePosition(double samplePosition) const {
  const double sampleDelta = samplePosition - anchorSamplePosition_;
  return addRoundedDeltaNs(anchorWallNs_, sampleDelta * effectiveSamplePeriodNs_);
}

void NodeClock::setSampleAnchor(double samplePosition, uint64_t utcNs) {
  anchorSamplePosition_ = samplePosition;
  anchorWallNs_ = utcNs;
  haveSampleAnchor_ = true;
}

void NodeClock::setWallReference(uint64_t utcNs, uint64_t monotonicUs, const char* source) {
  wallReferenceUtcNs_ = utcNs;
  wallReferenceMonotonicUs_ = monotonicUs;
  if (source != nullptr) {
    const uint64_t nowUs = time_us_64();
    std::printf(
        "[node] wall reference source=%s utc=%llu mono=%llu now_us=%llu scale=%.9f now_utc=%llu\n",
        source,
        static_cast<unsigned long long>(utcNs),
        static_cast<unsigned long long>(monotonicUs),
        static_cast<unsigned long long>(nowUs),
        static_cast<double>(localToUtcScale_),
        static_cast<unsigned long long>(utcAtMonotonicUs(nowUs)));
  }
}

void NodeClock::updateEffectiveSamplePeriod(double localToUtcScale, bool preferImmediateLock) {
  const double alpha = preferImmediateLock ? kGpsSlewAlpha : kNtpSlewAlpha;
  localToUtcScale_ = (localToUtcScale_ * (1.0 - alpha)) + (localToUtcScale * alpha);
  const double targetSamplePeriodNs = nominalSamplePeriodNs_ * localToUtcScale_;
  effectiveSamplePeriodNs_ =
      (effectiveSamplePeriodNs_ * (1.0 - alpha)) + (targetSamplePeriodNs * alpha);
  filteredFrequencyPpm_ = nsToPpm(nominalSamplePeriodNs_, effectiveSamplePeriodNs_);
}

bool NodeClock::gpsDisciplineAvailable(uint64_t monotonicUs) const {
  return gpsStablePulseCount_ >= kGpsStablePulseTarget &&
         lastGpsSyncMonotonicUs_ > 0 &&
         monotonicUs >= lastGpsSyncMonotonicUs_ &&
         (monotonicUs - lastGpsSyncMonotonicUs_) <= kGpsHoldoverUs;
}

bool NodeClock::currentPacketTimingDiagnostics(PacketTimingDiagnostics& outDiagnostics) const {
  outDiagnostics.hasGpsAnchor = gpsDisciplineAvailable(time_us_64());
  outDiagnostics.ppsEdgeCount = latestPpsEdgeCount_;
  outDiagnostics.dmaRingSlotIndex = lastObservedDmaRingSlotIndex_;
  outDiagnostics.ppsPhaseErrorNs = latestPpsPhaseErrorNs_;
  outDiagnostics.estimatedPpm = filteredFrequencyPpm_;
  return outDiagnostics.hasGpsAnchor || latestPpsEdgeCount_ > 0 || filteredFrequencyPpm_ != 0.0;
}

}  // namespace mmpr
