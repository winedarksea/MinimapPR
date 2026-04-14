#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/Types.h"

namespace mmpr {

class NodeClock {
 public:
  void begin(
      uint32_t sampleRateHz,
      size_t frameSamples,
      bool syncNtp,
      const char* ntpServer = "pool.ntp.org",
      long gmtOffsetSeconds = 0,
      int daylightOffsetSeconds = 0);

  uint64_t nextFrameStartNs();
  uint64_t nowUtcNs() const;
  uint64_t utcAtMonotonicUs(uint64_t monotonicUs) const;
  bool hasWallClock() const { return hasWallClock_; }
  TimeQuality timeQuality() const { return timeQuality_; }

  void setUtcNs(uint64_t utcNs, TimeQuality quality = TimeQuality::kNtpSync);
  void setUtcAtMonotonicUs(uint64_t utcNs, uint64_t monotonicUs, TimeQuality quality = TimeQuality::kNtpSync);

 private:
  uint64_t streamStartNs_ = 0;
  uint64_t streamStartMonotonicUs_ = 0;
  uint64_t frameDurationNs_ = 0;
  uint64_t frameIndex_ = 0;

  bool hasWallClock_ = false;
  TimeQuality timeQuality_ = TimeQuality::kFreerunning;
  uint64_t wallClockNs_ = 0;
  uint64_t wallClockSetUs_ = 0;
};

}  // namespace mmpr
