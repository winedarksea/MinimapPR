#include "mmpr/NodeClock.h"

#include <Arduino.h>
#include <sys/time.h>
#include <time.h>

namespace mmpr {

void NodeClock::begin(
    uint32_t sampleRateHz,
    size_t frameSamples,
    bool syncNtp,
    const char* ntpServer,
    long gmtOffsetSeconds,
    int daylightOffsetSeconds) {
#if defined(ESP32) || defined(ESP8266)
  if (syncNtp) {
    configTime(gmtOffsetSeconds, daylightOffsetSeconds, ntpServer);
    const uint32_t waitMs = 3000;
    const uint32_t startMs = millis();
    while ((millis() - startMs) < waitMs) {
      if (time(nullptr) > 1600000000) {
        hasWallClock_ = true;
        break;
      }
      delay(50);
    }
  }
#else
  (void)syncNtp;
  (void)ntpServer;
  (void)gmtOffsetSeconds;
  (void)daylightOffsetSeconds;
#endif

  if (!hasWallClock_) {
    hasWallClock_ = (time(nullptr) > 1600000000);
  }

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
  if (time(nullptr) > 1600000000) {
    timeval tv = {};
    gettimeofday(&tv, nullptr);
    return (static_cast<uint64_t>(tv.tv_sec) * 1000000000ULL) + (static_cast<uint64_t>(tv.tv_usec) * 1000ULL);
  }

  return static_cast<uint64_t>(millis()) * 1000000ULL;
}

void NodeClock::setUtcNs(uint64_t utcNs) {
  streamStartNs_ = utcNs;
  frameIndex_ = 0;
  hasWallClock_ = true;
}

}  // namespace mmpr
