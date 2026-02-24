#include "mmpr/NodeRunner.h"

#include <cstdio>
#include <inttypes.h>
#include <new>

#include "pico/time.h"

namespace mmpr {

NodeRunner::NodeRunner(
    const NodeDescriptor& descriptor,
    IAudioSource& audioSource,
    HttpFramePublisher& publisher,
    NodeClock& clock,
    uint32_t logEveryFrames,
    IEnvironmentalSource* environmentalSource)
    : descriptor_(descriptor),
      audioSource_(audioSource),
      publisher_(publisher),
      environmentalSource_(environmentalSource),
      clock_(clock),
      logEveryFrames_(logEveryFrames) {}

bool NodeRunner::begin(bool syncNtp, const char* ntpServer, long gmtOffsetSeconds, int daylightOffsetSeconds) {
  if (!audioSource_.begin()) {
    std::printf("[node] audio source init failed\n");
    return false;
  }

  frameBufferSamples_ = audioSource_.frameSamples() * static_cast<size_t>(audioSource_.channels());
  frameBuffer_ = new (std::nothrow) int16_t[frameBufferSamples_];
  if (frameBuffer_ == nullptr) {
    std::printf("[node] unable to allocate frame buffer\n");
    return false;
  }

  clock_.begin(
      audioSource_.sampleRateHz(),
      audioSource_.frameSamples(),
      syncNtp,
      ntpServer,
      gmtOffsetSeconds,
      daylightOffsetSeconds);

  if (environmentalSource_ != nullptr) {
    environmentalSourceReady_ = environmentalSource_->begin();
    if (!environmentalSourceReady_) {
      std::printf("[node] environmental source unavailable; continuing without telemetry\n");
    } else {
      std::printf("[node] environmental source enabled\n");
    }
  }

  std::printf(
      "[node] started id=%s channels=%u sample_rate=%lu frame_samples=%u endpoint=%s\n",
      descriptor_.id,
      static_cast<unsigned>(audioSource_.channels()),
      static_cast<unsigned long>(audioSource_.sampleRateHz()),
      static_cast<unsigned>(audioSource_.frameSamples()),
      publisher_.endpointUrl().c_str());

  return true;
}

void NodeRunner::loopOnce() {
  if (frameBuffer_ == nullptr) {
    return;
  }

  const bool frameOk = audioSource_.readFrame(frameBuffer_, audioSource_.frameSamples());
  if (!frameOk) {
    ++stats_.framesDropped;
    sleep_ms(1);
    return;
  }

  AudioFrame frame = {
      clock_.nextFrameStartNs(),
      audioSource_.sampleRateHz(),
      audioSource_.channels(),
      sequence_,
      frameBuffer_,
      audioSource_.frameSamples(),
  };

  EnvironmentalSample environmental = {};
  const EnvironmentalSample* environmentalPtr = nullptr;
  if (environmentalSourceReady_ && environmentalSource_ != nullptr) {
    if (environmentalSource_->read(environmental)) {
      environmentalPtr = &environmental;
    }
  }

  const PublishResult result = publisher_.publish(descriptor_, frame, environmentalPtr, false);

  ++stats_.framesCaptured;
  if (result.ok) {
    ++stats_.framesPublished;
  } else {
    ++stats_.publishErrors;
  }

  ++sequence_;

  if (logEveryFrames_ > 0 && (stats_.framesCaptured % logEveryFrames_) == 0) {
    std::printf(
        "[node] frames=%" PRIu64 " published=%" PRIu64 " dropped=%" PRIu64
        " errors=%" PRIu64 " last_status=%d\n",
        stats_.framesCaptured,
        stats_.framesPublished,
        stats_.framesDropped,
        stats_.publishErrors,
        result.statusCode);
  }
}

}  // namespace mmpr
