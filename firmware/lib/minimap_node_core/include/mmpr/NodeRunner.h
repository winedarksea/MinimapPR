#pragma once

#include <cstddef>
#include <cstdint>

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/IAudioSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

struct RunnerStats {
  uint64_t framesCaptured = 0;
  uint64_t framesPublished = 0;
  uint64_t framesDropped = 0;
  uint64_t publishErrors = 0;
};

class NodeRunner {
 public:
  NodeRunner(
      const NodeDescriptor& descriptor,
      IAudioSource& audioSource,
      HttpFramePublisher& publisher,
      NodeClock& clock,
      uint32_t logEveryFrames = 100);

  bool begin(
      bool syncNtp,
      const char* ntpServer = "pool.ntp.org",
      long gmtOffsetSeconds = 0,
      int daylightOffsetSeconds = 0);

  void loopOnce();

  const RunnerStats& stats() const { return stats_; }

 private:
  const NodeDescriptor& descriptor_;
  IAudioSource& audioSource_;
  HttpFramePublisher& publisher_;
  NodeClock& clock_;

  int16_t* frameBuffer_ = nullptr;
  size_t frameBufferSamples_ = 0;
  uint64_t sequence_ = 0;
  uint32_t logEveryFrames_ = 100;

  RunnerStats stats_;
};

}  // namespace mmpr
