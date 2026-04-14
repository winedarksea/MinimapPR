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
  if (hasWallClock_) {
    return wallClockNs_ + (time_us_64() - wallClockSetUs_) * 1000ULL;
  }
  return static_cast<uint64_t>(time_us_64()) * 1000ULL;
}

void NodeClock::setUtcNs(uint64_t utcNs, TimeQuality quality) {
  // Enforce priority: GPS > NTP > build_timestamp > freerunning.
  // Lower enum value = better quality.  Equal quality is allowed so that GPS
  // can keep refreshing its own anchor after the initial lock.
  if (hasWallClock_ && static_cast<int>(quality) > static_cast<int>(timeQuality_)) {
    return;
  }

  wallClockNs_ = utcNs;
  wallClockSetUs_ = time_us_64();
  hasWallClock_ = true;
  timeQuality_ = quality;

  // Preserve frame cadence when wall-clock lock arrives after streaming starts.
  const uint64_t elapsedFrameNs = frameIndex_ * frameDurationNs_;
  streamStartNs_ = (utcNs >= elapsedFrameNs) ? (utcNs - elapsedFrameNs) : 0;
}

}  // namespace mmpr
