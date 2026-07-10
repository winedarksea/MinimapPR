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
constexpr uint64_t kMaxSnapshotDelayNs = 50000000ULL;
constexpr uint32_t kPpsPhaseStatReportInterval = 60;
constexpr uint64_t kHoldoverEnterAgeUs = 2500000ULL;      // coast after 2.5 s of no PPS
constexpr uint64_t kHoldoverLogIntervalUs = 10000000ULL;  // 10 s holdover status cadence
constexpr uint32_t kReacquireRejectLimit = 10;            // accept override after N rejects
constexpr uint64_t kReacquireMinAllowedNs = 2000ULL;      // 2 us re-acquisition floor
constexpr double kHoldoverTempCompReanchorPpm = 0.02;     // re-anchor when |Δppm| exceeds this

const char* timeQualityName(TimeQuality quality) {
  switch (quality) {
    case TimeQuality::kGpsLocked:
      return "gps_locked";
    case TimeQuality::kGpsHoldover:
      return "gps_holdover";
    case TimeQuality::kNtpDisciplined:
      return "ntp_disciplined";
    case TimeQuality::kFreeRunning:
    default:
      return "free_running";
  }
}

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
  lastIgnoredSnapshotLogMonotonicUs_ = 0;
  ppsPhaseStatCount_ = 0;
  ppsPhaseSumSqNs_ = 0.0;
  ppsPhaseMaxAbsNs_ = 0;
  filteredFrequencyPpm_ = 0.0;
  // Holdover scalar state resets. The frequency/temperature models are
  // intentionally NOT wiped here — begin() runs once at boot when they are
  // already default-constructed, and their learned state must survive dropouts.
  holdoverActive_ = false;
  reacquireRejectCount_ = 0;
  haveTemperature_ = false;
  latestTemperatureC_ = 0.0;
  latestTemperatureMonotonicUs_ = 0;
  holdoverEntryTempC_ = 0.0;
  lastHoldoverLogMonotonicUs_ = 0;
  lastMedianPpm_ = 0.0;
  longTermModel_.resetWindow();

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
    // gpsDisciplineAvailable already enforced the (budget or legacy) holdover
    // window, so anything past "fresh" but still available is holdover.
    return TimeQuality::kGpsHoldover;
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

  // NTP interplay during a budgeted GPS holdover: RTT-noisy NTP must not
  // discipline rate or phase — that would corrupt a precise multi-minute coast.
  // Keep the >2 s sanity rescue (a gross disagreement still breaks a bad
  // holdover), update NTP bookkeeping so freshness/fallback keep working, and
  // otherwise defer before any rate/phase discipline.
  if (holdoverActive_ && gpsDisciplineAvailable(monotonicUs)) {
    const double samplePositionAtObservation = estimateSamplePositionAtMonotonicUs(monotonicUs);
    const uint64_t estimatedUtcNs = utcAtSamplePosition(samplePositionAtObservation);
    const int64_t phaseErrorNs =
        static_cast<int64_t>(utcNs) - static_cast<int64_t>(estimatedUtcNs);
    if (std::llabs(phaseErrorNs) > kGpsNtpSanityLimitNs) {
      holdoverActive_ = false;
      reacquireRejectCount_ = 0;
      haveGpsInterval_ = false;
      gpsStablePulseCount_ = 0;
      lastGpsSyncMonotonicUs_ = 0;
      lastGpsSyncMonotonicNs_ = 0;
      std::printf(
          "[node] NTP sanity broke GPS holdover error_ns=%lld; recovering with NTP\n",
          static_cast<long long>(phaseErrorNs));
      // Fall through to normal NTP discipline below.
    } else {
      haveNtpObservation_ = true;
      previousNtpUtcNs_ = utcNs;
      previousNtpMonotonicUs_ = monotonicUs;
      lastNtpSyncMonotonicUs_ = monotonicUs;
      std::printf("[node] NTP rate/phase deferred during GPS holdover\n");
      return;
    }
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
  if (missedPpsEdges || ppsEvent.rebased) {
    haveGpsInterval_ = false;
    gpsStablePulseCount_ = 0;
    latestPpsPhaseErrorNs_ = 0;
    ppsPhaseStatCount_ = 0;
    ppsPhaseSumSqNs_ = 0.0;
    ppsPhaseMaxAbsNs_ = 0;
    // Discard only the partial median window; the learned long-term frequency
    // must survive re-acquisition (this is the fix for the legacy reset defect).
    longTermModel_.resetWindow();
    if (ppsEvent.rebased) {
      std::printf(
          "[timing] PPS timebase rebased at edge=%u; restarting GPS lock acquisition\n",
          static_cast<unsigned>(ppsEvent.edgeCount));
    } else {
      std::printf(
          "[timing] PPS edge jump detected previous=%u current=%u; restarting GPS lock acquisition\n",
          static_cast<unsigned>(latestPpsEdgeCount_),
          static_cast<unsigned>(ppsEvent.edgeCount));
    }
  }

  double samplePosition = estimateSamplePositionAtMonotonicUs(monotonicUs);
  if (ppsEvent.audioProducerSnapshot.valid) {
    const double samplePeriodNs =
        effectiveSamplePeriodNs_ > 0.0 ? effectiveSamplePeriodNs_ : nominalSamplePeriodNs_;
    const uint64_t snapshotMonotonicNs = ppsEvent.audioProducerSnapshot.capturedMonotonicUs * 1000ULL;
    const uint64_t snapshotDelayNs =
        snapshotMonotonicNs > monotonicNs ? (snapshotMonotonicNs - monotonicNs) : 0ULL;
    if (snapshotDelayNs > kMaxSnapshotDelayNs) {
      // A snapshot this far behind the PPS edge means the tick and timer
      // domains disagree beyond any plausible IRQ latency; keep the
      // monotonic-time estimate instead of a wildly over-corrected position.
      if (lastIgnoredSnapshotLogMonotonicUs_ == 0 ||
          monotonicUs < lastIgnoredSnapshotLogMonotonicUs_ ||
          (monotonicUs - lastIgnoredSnapshotLogMonotonicUs_) >= kRejectedGpsEpochLogIntervalUs) {
        lastIgnoredSnapshotLogMonotonicUs_ = monotonicUs;
        std::printf(
            "[timing] ignored audio snapshot with delay_ns=%llu at PPS edge=%u\n",
            static_cast<unsigned long long>(snapshotDelayNs),
            static_cast<unsigned>(ppsEvent.edgeCount));
      }
    } else {
      const double delaySamples = samplePeriodNs > 0.0
          ? (static_cast<double>(snapshotDelayNs) / samplePeriodNs)
          : 0.0;
      samplePosition = std::max<double>(0.0, ppsEvent.audioProducerSnapshot.samplePosition - delaySamples);
      lastObservedCompletedBlockCount_ = ppsEvent.audioProducerSnapshot.completedBlockCount;
      lastObservedDmaRingSlotIndex_ = ppsEvent.audioProducerSnapshot.dmaRingSlotIndex;
    }
  }
  const uint64_t gpsAgeUs =
      (lastGpsSyncMonotonicUs_ > 0 && monotonicUs >= lastGpsSyncMonotonicUs_)
          ? (monotonicUs - lastGpsSyncMonotonicUs_)
          : UINT64_MAX;
  const bool gpsWasFresh = ageWithinHoldover(gpsAgeUs);

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
      longTermModel_.resetWindow();  // keep learned frequency; drop partial window only
      std::printf(
          "[timing] PPS interval out of range interval_us=%llu edge=%u; waiting for stable 1 Hz PPS\n",
          static_cast<unsigned long long>(localIntervalUs),
          static_cast<unsigned>(ppsEvent.edgeCount));
      return;
    }

    const double localToUtcScale = clampScale(static_cast<double>(kNsPerSecond) / static_cast<double>(localIntervalNs));
    updateEffectiveSamplePeriod(localToUtcScale, true);
    gpsStablePulseCount_ = std::min<uint32_t>(gpsStablePulseCount_ + 1U, kGpsStablePulseTarget);

    // Long-term crystal frequency learning: only fully-locked pulses feed the
    // model. The median-of-9 prefilter rejects degraded pulses at jamming onset.
    if (gpsStablePulseCount_ >= kGpsStablePulseTarget) {
      const double ppmInst = (localToUtcScale - 1.0) * kPpmScale;
      const LtSampleResult lt = longTermModel_.addLockedSample(ppmInst);
      if (lt.accepted) {
        lastMedianPpm_ = lt.medianPpm;
        // Shadow temperature model: feed the accepted median against a fresh
        // (< 10 s) on-node temperature sample when one exists.
        if (haveTemperature_ && monotonicUs >= latestTemperatureMonotonicUs_ &&
            (monotonicUs - latestTemperatureMonotonicUs_) <= kTemperatureFreshUs) {
          tempModel_.addSample(latestTemperatureC_, lt.medianPpm);
        }
        if (lt.reseeded) {
          std::printf(
              "[timing] lt-freq reseed lt_ppm=%.3f; sustained frequency shift detected\n",
              longTermModel_.ltPpm());
        }
      }
    }
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

  // Re-acquisition sanity: while coasting in a budgeted holdover, don't hard
  // re-anchor on a returning PPS UTC label that contradicts the held-over clock
  // beyond a few multiples of the predicted holdover error. A dropped wrap pulse
  // can leave the GPS source's edge label stale, so a mismatched first label is
  // treated as suspect rather than trusted. After a bounded number of rejects we
  // accept anyway (the clock genuinely shifted).
  if (holdoverActive_ && longTermModel_.valid() && withinHoldoverBudget(gpsAgeUs)) {
    const uint64_t predErrNs = predictedHoldoverErrorNs(gpsAgeUs);
    const uint64_t allowedNs = std::max<uint64_t>(kReacquireMinAllowedNs, 4ULL * predErrNs);
    if (static_cast<uint64_t>(std::llabs(gpsEpochErrorNs)) > allowedNs) {
      ++reacquireRejectCount_;
      latestPpsEdgeCount_ = ppsEvent.edgeCount;  // preserve edge continuity
      std::printf(
          "[timing] re-acquisition label mismatch error_ns=%lld allowed_ns=%llu reject=%u; holding over\n",
          static_cast<long long>(gpsEpochErrorNs),
          static_cast<unsigned long long>(allowedNs),
          static_cast<unsigned>(reacquireRejectCount_));
      if (reacquireRejectCount_ < kReacquireRejectLimit) {
        // Holdover continues; lastGpsSyncMonotonicUs_ deliberately untouched.
        return;
      }
      std::printf(
          "[timing] re-acquisition override after %u rejects; accepting GPS anchor\n",
          static_cast<unsigned>(reacquireRejectCount_));
    }
  }

  lastGpsSyncMonotonicUs_ = monotonicUs;
  lastGpsSyncMonotonicNs_ = monotonicNs;
  // Any accepted anchor ends holdover coasting and clears the reject counter.
  holdoverActive_ = false;
  reacquireRejectCount_ = 0;
  const uint64_t anchorSampleIndex =
      samplePosition <= 0.0 ? 0ULL : static_cast<uint64_t>(std::llround(samplePosition));
  if (!gpsWasFresh || !haveSampleAnchor_ || gpsStablePulseCount_ == 1) {
    latestPpsPhaseErrorNs_ = 0;
    setSampleAnchor(samplePosition, utcSecondNs);
  } else {
    latestPpsPhaseErrorNs_ =
        static_cast<int64_t>(utcSecondNs) - static_cast<int64_t>(utcAtSampleIndex(anchorSampleIndex));
    setSampleAnchor(samplePosition, utcSecondNs);
    ++ppsPhaseStatCount_;
    const double phaseErrorNs = static_cast<double>(latestPpsPhaseErrorNs_);
    ppsPhaseSumSqNs_ += phaseErrorNs * phaseErrorNs;
    ppsPhaseMaxAbsNs_ = std::max<int64_t>(ppsPhaseMaxAbsNs_, std::llabs(latestPpsPhaseErrorNs_));
    if (ppsPhaseStatCount_ >= kPpsPhaseStatReportInterval) {
      std::printf(
          "[timing] pps phase stats n=%u rms_ns=%.0f max_ns=%lld ppm=%.3f\n",
          static_cast<unsigned>(ppsPhaseStatCount_),
          std::sqrt(ppsPhaseSumSqNs_ / static_cast<double>(ppsPhaseStatCount_)),
          static_cast<long long>(ppsPhaseMaxAbsNs_),
          filteredFrequencyPpm_);
      ppsPhaseStatCount_ = 0;
      ppsPhaseSumSqNs_ = 0.0;
      ppsPhaseMaxAbsNs_ = 0;
      // Shadow-mode model telemetry at the same 60-pulse cadence as the phase
      // stats line above (no new logging cadence introduced).
      std::printf(
          "[timing] lt-freq lt_ppm=%.3f sigma_ppm=%.3f n=%u ewma_ppm=%.3f\n",
          longTermModel_.ltPpm(),
          longTermModel_.sigmaPpm(),
          static_cast<unsigned>(longTermModel_.updateCount()),
          filteredFrequencyPpm_);
      if (haveTemperature_) {
        std::printf(
            "[timing] tempmodel T=%.2fC meas_ppm=%.3f pred_ppm=%.3f slope_ppm_c=%.4f "
            "n_eff=%.0f rms_ppm=%.4f valid=%d\n",
            latestTemperatureC_,
            lastMedianPpm_,
            tempModel_.predictPpm(latestTemperatureC_),
            tempModel_.slopePpmPerC(),
            tempModel_.nEff(),
            tempModel_.residRmsPpm(),
            tempModel_.valid() ? 1 : 0);
      }
    }
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

bool NodeClock::ageWithinHoldover(uint64_t ageUs) const {
  // Error-budget window once the long-term model is characterized; legacy fixed
  // 60 s until then (boot behavior identical, model-less nodes unchanged).
  return longTermModel_.valid() ? withinHoldoverBudget(ageUs) : (ageUs <= kGpsHoldoverUs);
}

bool NodeClock::gpsDisciplineAvailable(uint64_t monotonicUs) const {
  if (gpsStablePulseCount_ < kGpsStablePulseTarget || lastGpsSyncMonotonicUs_ == 0 ||
      monotonicUs < lastGpsSyncMonotonicUs_) {
    return false;
  }
  return ageWithinHoldover(monotonicUs - lastGpsSyncMonotonicUs_);
}

bool NodeClock::currentPacketTimingDiagnostics(PacketTimingDiagnostics& outDiagnostics) const {
  const uint64_t nowUs = time_us_64();
  outDiagnostics.hasGpsAnchor = gpsDisciplineAvailable(nowUs);
  outDiagnostics.ppsEdgeCount = latestPpsEdgeCount_;
  outDiagnostics.dmaRingSlotIndex = lastObservedDmaRingSlotIndex_;
  outDiagnostics.ppsPhaseErrorNs = latestPpsPhaseErrorNs_;
  outDiagnostics.estimatedPpm = filteredFrequencyPpm_;

  const uint64_t gpsAgeUs =
      (lastGpsSyncMonotonicUs_ > 0 && nowUs >= lastGpsSyncMonotonicUs_)
          ? (nowUs - lastGpsSyncMonotonicUs_)
          : 0ULL;
  outDiagnostics.holdoverActive = holdoverActive_;
  outDiagnostics.holdoverAgeMs =
      holdoverActive_ ? static_cast<uint32_t>(std::min<uint64_t>(gpsAgeUs / 1000ULL, UINT32_MAX)) : 0;
  outDiagnostics.predictedErrorNs =
      holdoverActive_
          ? static_cast<uint32_t>(std::min<uint64_t>(predictedHoldoverErrorNs(gpsAgeUs), UINT32_MAX))
          : 0;
  outDiagnostics.ltValid = longTermModel_.valid();
  outDiagnostics.ltPpm = longTermModel_.ltPpm();
  outDiagnostics.ltSigmaPpm = longTermModel_.sigmaPpm();
  outDiagnostics.tempModelValid = tempModel_.valid();
  outDiagnostics.tempCompApplied =
      holdoverActive_ && holdoverConfig_.enableTempComp && tempModel_.valid();
  outDiagnostics.tempSlopePpmPerC = tempModel_.slopePpmPerC();
  outDiagnostics.tempResidRmsPpm = tempModel_.residRmsPpm();

  return outDiagnostics.hasGpsAnchor || latestPpsEdgeCount_ > 0 || filteredFrequencyPpm_ != 0.0 ||
         holdoverActive_ || longTermModel_.seeded();
}

void NodeClock::observeTemperature(double temperatureC, uint64_t monotonicUs) {
  latestTemperatureC_ = temperatureC;
  latestTemperatureMonotonicUs_ = monotonicUs;
  haveTemperature_ = true;
}

uint64_t NodeClock::predictedHoldoverErrorNs(uint64_t ageUs) const {
  const double drift =
      effectiveHoldoverDriftPpm(holdoverConfig_.driftUncertaintyPpm, longTermModel_.sigmaPpm());
  return mmpr::predictedHoldoverErrorNs(ageUs, drift);
}

bool NodeClock::withinHoldoverBudget(uint64_t ageUs) const {
  return predictedHoldoverErrorNs(ageUs) <= holdoverConfig_.errorBudgetNs &&
         ageUs <= holdoverConfig_.maxAgeUs;
}

void NodeClock::maintain(uint64_t nowMonotonicUs) {
  if (!holdoverActive_) {
    // Common case (PPS fresh): a couple of integer comparisons and out. No
    // model math, no allocation, no blocking.
    if (lastGpsSyncMonotonicUs_ == 0 || nowMonotonicUs < lastGpsSyncMonotonicUs_) {
      return;
    }
    const uint64_t ageUs = nowMonotonicUs - lastGpsSyncMonotonicUs_;
    if (ageUs <= kHoldoverEnterAgeUs) {
      return;
    }
    // Beyond here only during an actual PPS gap. Require a characterized model
    // and a prior stable lock; otherwise fall back to legacy passive coasting.
    if (!longTermModel_.valid() || gpsStablePulseCount_ < kGpsStablePulseTarget) {
      return;
    }
    enterHoldover(nowMonotonicUs, ageUs);
    return;
  }
  maintainActiveHoldover(nowMonotonicUs);
}

void NodeClock::enterHoldover(uint64_t nowMonotonicUs, uint64_t ageUs) {
  const double ewmaPpmBefore = filteredFrequencyPpm_;

  // 1. Continuity re-anchor with the OLD slope: pin both timebases to the
  //    current instant so the timestamp value at "now" is unchanged.
  const double samplePos = estimateSamplePositionAtMonotonicUs(nowMonotonicUs);
  const uint64_t utcNow = utcAtMonotonicUs(nowMonotonicUs);
  setSampleAnchor(samplePos, utcNow);
  setWallReference(utcNow, nowMonotonicUs);

  // 2. Swap the slope to the learned long-term frequency. Temperature comp (if
  //    enabled) is relative to entry temperature, so its correction is zero at
  //    entry; it is applied incrementally in maintainActiveHoldover().
  holdoverEntryTempC_ = latestTemperatureC_;
  const bool tempComp =
      holdoverConfig_.enableTempComp && tempModel_.valid() && haveTemperature_;
  const double holdoverPpm = longTermModel_.ltPpm();
  const double newScale = clampScale(1.0 + holdoverPpm / kPpmScale);
  localToUtcScale_ = newScale;
  effectiveSamplePeriodNs_ = nominalSamplePeriodNs_ * newScale;
  filteredFrequencyPpm_ = nsToPpm(nominalSamplePeriodNs_, effectiveSamplePeriodNs_);

  holdoverActive_ = true;
  lastHoldoverLogMonotonicUs_ = nowMonotonicUs;
  std::printf(
      "[timing] holdover enter age_ms=%llu ewma_ppm=%.3f lt_ppm=%.3f sigma_ppm=%.3f temp_comp=%d\n",
      static_cast<unsigned long long>(ageUs / 1000ULL),
      ewmaPpmBefore,
      holdoverPpm,
      longTermModel_.sigmaPpm(),
      tempComp ? 1 : 0);
}

void NodeClock::maintainActiveHoldover(uint64_t nowMonotonicUs) {
  const uint64_t ageUs =
      (lastGpsSyncMonotonicUs_ > 0 && nowMonotonicUs >= lastGpsSyncMonotonicUs_)
          ? (nowMonotonicUs - lastGpsSyncMonotonicUs_)
          : 0ULL;

  // Relative temperature prediction (computed for logging regardless of the
  // flag; only APPLIED when temp-comp is enabled).
  double tempPredPpm = 0.0;
  if (tempModel_.valid() && haveTemperature_) {
    tempPredPpm = tempModel_.predictRelativePpm(latestTemperatureC_ - holdoverEntryTempC_);
  }
  if (holdoverConfig_.enableTempComp && tempModel_.valid() && haveTemperature_) {
    const double targetPpm = longTermModel_.ltPpm() + tempPredPpm;
    if (std::fabs(targetPpm - filteredFrequencyPpm_) > kHoldoverTempCompReanchorPpm) {
      // Continuity re-anchor, then swap to the temperature-compensated slope.
      const double samplePos = estimateSamplePositionAtMonotonicUs(nowMonotonicUs);
      const uint64_t utcNow = utcAtMonotonicUs(nowMonotonicUs);
      setSampleAnchor(samplePos, utcNow);
      setWallReference(utcNow, nowMonotonicUs);
      const double newScale = clampScale(1.0 + targetPpm / kPpmScale);
      localToUtcScale_ = newScale;
      effectiveSamplePeriodNs_ = nominalSamplePeriodNs_ * newScale;
      filteredFrequencyPpm_ = nsToPpm(nominalSamplePeriodNs_, effectiveSamplePeriodNs_);
    }
  }

  if (lastHoldoverLogMonotonicUs_ == 0 || nowMonotonicUs < lastHoldoverLogMonotonicUs_ ||
      (nowMonotonicUs - lastHoldoverLogMonotonicUs_) >= kHoldoverLogIntervalUs) {
    lastHoldoverLogMonotonicUs_ = nowMonotonicUs;
    std::printf(
        "[timing] holdover age_ms=%llu pred_err_ns=%llu lt_ppm=%.3f temp_pred_ppm=%.3f "
        "applied_ppm=%.3f quality=%s\n",
        static_cast<unsigned long long>(ageUs / 1000ULL),
        static_cast<unsigned long long>(predictedHoldoverErrorNs(ageUs)),
        longTermModel_.ltPpm(),
        tempPredPpm,
        filteredFrequencyPpm_,
        timeQualityName(timeQuality()));
  }
}

}  // namespace mmpr
