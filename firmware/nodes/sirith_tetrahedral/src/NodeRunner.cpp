#include "mmpr/NodeRunner.h"

#include <cstdio>
#include <inttypes.h>
#include <new>
#include <algorithm>
#include <vector>

#include "pico/time.h"

namespace mmpr {
namespace {

constexpr uint64_t kNsPerSecond = 1000000000ULL;

uint64_t nextUtcSecondBoundary(uint64_t utcNs) {
  return ((utcNs / kNsPerSecond) + 1ULL) * kNsPerSecond;
}

}  // namespace

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

  packetInterleavedSamples_.clear();
  packetInterleavedSamples_.reserve(
      audioSource_.sampleRateHz() * static_cast<size_t>(audioSource_.channels()));

  std::printf(
      "[node] started id=%s channels=%u sample_rate=%lu frame_samples=%u endpoint=%s\n",
      descriptor_.id,
      static_cast<unsigned>(audioSource_.channels()),
      static_cast<unsigned long>(audioSource_.sampleRateHz()),
      static_cast<unsigned>(audioSource_.frameSamples()),
      publisher_.endpointUrl().c_str());

  return true;
}

bool NodeRunner::publishCurrentPacket(
    uint64_t packetEndSampleIndex,
    uint64_t packetEndUtcNs,
    const EnvironmentalSample* environmentalSample,
    int& lastPublishStatus) {
  if (!packetOpen_ || packetInterleavedSamples_.empty() || packetEndSampleIndex <= packetStartSampleIndex_) {
    packetOpen_ = false;
    packetInterleavedSamples_.clear();
    packetTargetEndUtcNs_ = 0;
    return false;
  }

  const uint64_t receiptNs = clock_.utcAtMonotonicUs(time_us_64());
  PacketTimingDiagnostics timingDiagnostics = {};
  const bool haveTimingDiagnostics = clock_.currentPacketTimingDiagnostics(timingDiagnostics);
  const AudioFrame frame = {
      packetStartUtcNs_,
      packetEndUtcNs,
      packetStartSampleIndex_,
      packetEndSampleIndex,
      audioSource_.sampleRateHz(),
      audioSource_.channels(),
      sequence_,
      packetStartUtcNs_,
      receiptNs,
      clock_.timeQuality(),
      haveTimingDiagnostics,
      timingDiagnostics.ppsEdgeCount,
      timingDiagnostics.dmaRingSlotIndex,
      timingDiagnostics.ppsPhaseErrorNs,
      timingDiagnostics.estimatedPpm,
      packetInterleavedSamples_.data(),
      packetInterleavedSamples_.size() / static_cast<size_t>(audioSource_.channels()),
  };
  const PublishResult result = publisher_.publish(descriptor_, frame, environmentalSample, false);
  lastPublishStatus = result.statusCode;
  if (result.ok) {
    ++stats_.framesPublished;
  } else {
    ++stats_.publishErrors;
  }
  ++sequence_;
  packetOpen_ = false;
  packetInterleavedSamples_.clear();
  packetTargetEndUtcNs_ = 0;
  return result.ok;
}

void NodeRunner::loopOnce() {
  if (frameBuffer_ == nullptr) {
    return;
  }

  AudioCaptureTimestamp captureTimestamp = {};
  const bool frameOk =
      audioSource_.readFrame(frameBuffer_, audioSource_.frameSamples(), &captureTimestamp);
  if (!frameOk) {
    ++stats_.framesDropped;
    sleep_ms(1);
    return;
  }

  clock_.observeCapturedAudio(captureTimestamp);

  ++stats_.framesCaptured;
  stats_.framesDropped += captureTimestamp.droppedBlocksBeforeCapture;

  EnvironmentalSample environmental = {};
  const EnvironmentalSample* environmentalPtr = nullptr;
  if (environmentalSourceReady_ && environmentalSource_ != nullptr && environmentalSource_->read(environmental)) {
    environmentalPtr = &environmental;
  }

  const uint8_t channels = audioSource_.channels();
  int lastPublishStatus = -1;
  if (haveExpectedNextSampleIndex_ && captureTimestamp.startSampleIndex != expectedNextSampleIndex_) {
    ++stats_.packetContinuityViolations;
    std::printf(
        "[timing] audio continuity gap expected=%" PRIu64 " actual=%" PRIu64 " dropped_blocks=%u\n",
        expectedNextSampleIndex_,
        captureTimestamp.startSampleIndex,
        static_cast<unsigned>(captureTimestamp.droppedBlocksBeforeCapture));

    if (packetOpen_ && !packetInterleavedSamples_.empty()) {
      const uint64_t packetEndSampleIndex =
          packetStartSampleIndex_ + (packetInterleavedSamples_.size() / static_cast<size_t>(channels));
      const uint64_t packetEndUtcNs = clock_.utcAtSampleIndex(packetEndSampleIndex);
      publishCurrentPacket(packetEndSampleIndex, packetEndUtcNs, environmentalPtr, lastPublishStatus);
    } else {
      packetOpen_ = false;
      packetInterleavedSamples_.clear();
      packetTargetEndUtcNs_ = 0;
    }
  }
  haveExpectedNextSampleIndex_ = true;
  expectedNextSampleIndex_ = captureTimestamp.endSampleIndex;

  size_t frameSampleOffset = 0;
  while (frameSampleOffset < audioSource_.frameSamples()) {
    const uint64_t chunkStartSampleIndex = captureTimestamp.startSampleIndex + frameSampleOffset;
    if (!packetOpen_) {
      packetOpen_ = true;
      packetStartSampleIndex_ = chunkStartSampleIndex;
      packetStartUtcNs_ = clock_.utcAtSampleIndex(packetStartSampleIndex_);
      packetTargetEndUtcNs_ = nextUtcSecondBoundary(packetStartUtcNs_);
      packetInterleavedSamples_.clear();
    }

    uint64_t packetEndSampleIndex = clock_.sampleIndexAtUtcNs(packetTargetEndUtcNs_);
    if (packetEndSampleIndex <= chunkStartSampleIndex) {
      packetEndSampleIndex = chunkStartSampleIndex + 1ULL;
    }

    const uint64_t chunkSamplesUntilBoundary = packetEndSampleIndex - chunkStartSampleIndex;
    const size_t frameSamplesRemaining = audioSource_.frameSamples() - frameSampleOffset;
    const size_t chunkSamples =
        static_cast<size_t>(std::min<uint64_t>(chunkSamplesUntilBoundary, frameSamplesRemaining));
    const int16_t* chunkStart = frameBuffer_ + (frameSampleOffset * static_cast<size_t>(channels));
    packetInterleavedSamples_.insert(
        packetInterleavedSamples_.end(),
        chunkStart,
        chunkStart + (chunkSamples * static_cast<size_t>(channels)));
    frameSampleOffset += chunkSamples;

    if ((packetStartSampleIndex_ + (packetInterleavedSamples_.size() / static_cast<size_t>(channels))) >=
        packetEndSampleIndex) {
      const uint64_t packetEndUtcNs = packetTargetEndUtcNs_;
      publishCurrentPacket(packetEndSampleIndex, packetEndUtcNs, environmentalPtr, lastPublishStatus);
      packetOpen_ = false;
      packetStartSampleIndex_ = packetEndSampleIndex;
      packetStartUtcNs_ = packetEndUtcNs;
      packetTargetEndUtcNs_ = 0;
    }
  }

  if (logEveryFrames_ > 0 && (stats_.framesCaptured % logEveryFrames_) == 0) {
    std::printf(
        "[node] blocks=%" PRIu64 " published=%" PRIu64 " dropped=%" PRIu64
        " continuity=%" PRIu64 " errors=%" PRIu64 " last_status=%d\n",
        stats_.framesCaptured,
        stats_.framesPublished,
        stats_.framesDropped,
        stats_.packetContinuityViolations,
        stats_.publishErrors,
        lastPublishStatus);
  }
}

}  // namespace mmpr
