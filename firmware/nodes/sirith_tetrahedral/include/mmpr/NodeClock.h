#pragma once

#include <cstddef>
#include <cstdint>

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
  bool hasWallClock() const { return hasWallClock_; }

  void setUtcNs(uint64_t utcNs);

 private:
  uint64_t streamStartNs_ = 0;
  uint64_t frameDurationNs_ = 0;
  uint64_t frameIndex_ = 0;

  bool hasWallClock_ = false;
  uint64_t wallClockNs_ = 0;
  uint64_t wallClockSetUs_ = 0;
};

}  // namespace mmpr
