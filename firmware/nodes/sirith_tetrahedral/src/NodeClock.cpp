#include "mmpr/NodeClock.h"

#include <cstdio>

#include "pico/time.h"

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

  if (syncNtp) {
    std::printf("[node] NTP sync requested but not enabled in this no-OS build; using monotonic clock\n");
  }

  hasWallClock_ = false;
  timeQuality_ = TimeQuality::kFreerunning;
  streamStartNs_ = nowUtcNs();
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
  wallClockNs_ = utcNs;
  wallClockSetUs_ = time_us_64();
  hasWallClock_ = true;
  timeQuality_ = quality;

  // Preserve frame cadence when wall-clock lock arrives after streaming starts.
  const uint64_t elapsedFrameNs = frameIndex_ * frameDurationNs_;
  streamStartNs_ = (utcNs >= elapsedFrameNs) ? (utcNs - elapsedFrameNs) : 0;
}

}  // namespace mmpr
