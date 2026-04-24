#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mmpr/HttpFramePublisher.h"
#include "mmpr/IAudioSource.h"
#include "mmpr/IEnvironmentalSource.h"
#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

struct RunnerStats {
  uint64_t framesCaptured = 0;
  uint64_t framesPublished = 0;
  uint64_t framesDropped = 0;
  uint64_t publishErrors = 0;
  uint64_t packetContinuityViolations = 0;
};

class NodeRunner {
 public:
  NodeRunner(
      const NodeDescriptor& descriptor,
      IAudioSource& audioSource,
      HttpFramePublisher& publisher,
      NodeClock& clock,
      uint32_t logEveryFrames = 100,
      IEnvironmentalSource* environmentalSource = nullptr);

  bool begin(
      bool syncNtp,
      const char* ntpServer = "pool.ntp.org",
      long gmtOffsetSeconds = 0,
      int daylightOffsetSeconds = 0);

  void loopOnce();

  const RunnerStats& stats() const { return stats_; }

 private:
  bool publishCurrentPacket(
      uint64_t packetEndSampleIndex,
      uint64_t packetEndUtcNs,
      const EnvironmentalSample* environmentalSample,
      int& lastPublishStatus);

  const NodeDescriptor& descriptor_;
  IAudioSource& audioSource_;
  HttpFramePublisher& publisher_;
  IEnvironmentalSource* environmentalSource_ = nullptr;
  bool environmentalSourceReady_ = false;
  NodeClock& clock_;

  int16_t* frameBuffer_ = nullptr;
  size_t frameBufferSamples_ = 0;
  std::vector<int16_t> packetInterleavedSamples_;
  bool packetOpen_ = false;
  uint64_t packetStartSampleIndex_ = 0;
  uint64_t packetStartUtcNs_ = 0;
  uint64_t packetTargetEndUtcNs_ = 0;
  bool haveExpectedNextSampleIndex_ = false;
  uint64_t expectedNextSampleIndex_ = 0;
  uint64_t sequence_ = 0;
  uint32_t logEveryFrames_ = 100;

  RunnerStats stats_ = {};
};

}  // namespace mmpr
