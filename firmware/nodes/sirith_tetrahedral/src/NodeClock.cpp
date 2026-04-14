#include "mmpr/NodeClock.h"

#include <cstdio>

#include "pico/time.h"

#include "mmpr/BuildTimestamp.h"

namespace mmpr {

void NodeClock::begin(
    uint32_t sampleRateHz,
    size_t frameSamples,
    bool syncNtp,
    const char* ntpServer,
    long gmtOffsetSeconds,
    int daylightOffsetSeconds) {
  (void)ntpServer;
  (void)gmtOffsetSeconds;
  (void)daylightOffsetSeconds;

  // NTP is handled externally by NtpClient; the syncNtp parameter is retained
  // for API compatibility but has no effect here.
  (void)syncNtp;

  // Anchor the freerunning clock to the compile-time timestamp so frame
  // timestamps are approximately correct even without GPS or NTP.  The wall
  // clock will be upgraded to kNtpSync or kGpsLocked if a better source
  // becomes available later.
  streamStartMonotonicUs_ = time_us_64();
  setUtcNs(kBuildEpochNs, TimeQuality::kBuildTimestamp);
  std::printf(
      "[node] clock anchored to build timestamp %llu ns (build_timestamp quality)\n",
      static_cast<unsigned long long>(kBuildEpochNs));
  frameDurationNs_ =
      static_cast<uint64_t>((static_cast<double>(frameSamples) * 1000000000.0) / static_cast<double>(sampleRateHz));
  frameIndex_ = 0;
}

uint64_t NodeClock::nextFrameStartNs() {
  const uint64_t out = streamStartNs_ + (frameIndex_ * frameDurationNs_);
  ++frameIndex_;
  return out;
}

uint64_t NodeClock::nowUtcNs() const {
  return utcAtMonotonicUs(time_us_64());
}

uint64_t NodeClock::utcAtMonotonicUs(uint64_t monotonicUs) const {
  if (hasWallClock_) {
    return wallClockNs_ + (monotonicUs - wallClockSetUs_) * 1000ULL;
  }
  return monotonicUs * 1000ULL;
}

void NodeClock::setUtcNs(uint64_t utcNs, TimeQuality quality) {
  setUtcAtMonotonicUs(utcNs, time_us_64(), quality);
}

void NodeClock::setUtcAtMonotonicUs(uint64_t utcNs, uint64_t monotonicUs, TimeQuality quality) {
  // Enforce priority: GPS > NTP > build_timestamp > freerunning.
  // Lower enum value = better quality.  Equal quality is allowed so that GPS
  // can keep refreshing its own anchor after the initial lock.
  if (hasWallClock_ && static_cast<int>(quality) > static_cast<int>(timeQuality_)) {
    return;
  }

  wallClockNs_ = utcNs;
  wallClockSetUs_ = monotonicUs;
  hasWallClock_ = true;
  timeQuality_ = quality;

  // Preserve frame cadence against the monotonic capture timeline instead of
  // the frame counter so delayed sync handling still lands on the true stream.
  const uint64_t elapsedSinceStreamStartNs =
      (monotonicUs >= streamStartMonotonicUs_) ? (monotonicUs - streamStartMonotonicUs_) * 1000ULL : 0ULL;
  streamStartNs_ = (utcNs >= elapsedSinceStreamStartNs) ? (utcNs - elapsedSinceStreamStartNs) : 0;
}

}  // namespace mmpr
