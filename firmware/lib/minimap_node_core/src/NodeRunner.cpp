#include "mmpr/NodeRunner.h"

#include <Arduino.h>

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
    Serial.println("[node] audio source init failed");
    return false;
  }

  frameBufferSamples_ = audioSource_.frameSamples() * static_cast<size_t>(audioSource_.channels());
  frameBuffer_ = new int16_t[frameBufferSamples_];
  if (frameBuffer_ == nullptr) {
    Serial.println("[node] unable to allocate frame buffer");
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
      Serial.println("[node] environmental source unavailable; continuing without environmental telemetry");
    } else {
      Serial.println("[node] environmental source enabled");
    }
  }

  Serial.printf(
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
    delay(1);
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
    Serial.printf(
        "[node] frames=%llu published=%llu dropped=%llu errors=%llu last_status=%d\n",
        static_cast<unsigned long long>(stats_.framesCaptured),
        static_cast<unsigned long long>(stats_.framesPublished),
        static_cast<unsigned long long>(stats_.framesDropped),
        static_cast<unsigned long long>(stats_.publishErrors),
        result.statusCode);
  }
}

}  // namespace mmpr
