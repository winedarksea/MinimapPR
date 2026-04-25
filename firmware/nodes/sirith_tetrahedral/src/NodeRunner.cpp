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
constexpr int kPublishSkippedForBackoffStatus = -5;

uint64_t nextUtcSecondBoundary(uint64_t utcNs) {
  return ((utcNs / kNsPerSecond) + 1ULL) * kNsPerSecond;
}

bool deadlineReached(uint32_t nowMs, uint32_t deadlineMs) {
  return static_cast<int32_t>(nowMs - deadlineMs) >= 0;
}

}  // namespace

NodeRunner::NodeRunner(
    const NodeDescriptor& descriptor,
    IAudioSource& audioSource,
    HttpFramePublisher& publisher,
    NodeClock& clock,
    uint32_t logEveryFrames,
    IEnvironmentalSource* environmentalSource,
    size_t maxPacketSamplesPerChannel,
    uint32_t publishFailureBackoffMs)
    : descriptor_(descriptor),
      audioSource_(audioSource),
      publisher_(publisher),
      environmentalSource_(environmentalSource),
      clock_(clock),
      logEveryFrames_(logEveryFrames),
      maxPacketSamplesPerChannel_(maxPacketSamplesPerChannel),
      publishFailureBackoffMs_(publishFailureBackoffMs) {}

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

  // Bound the packet vector reservation. The previous version reserved a full
  // second of interleaved audio (sampleRate * channels * 2 bytes), which on a
  // 4-channel 16 kHz config is 128 KB. On the RP2350 (520 KB SRAM) that
  // competes with the 256 KB DMA ring + CYW43/lwIP heap and either throws
  // bad_alloc (terminate, then watchdog reset) or fragments the heap so the
  // first publish allocation fails. Cap at maxPacketSamplesPerChannel_ when
  // configured; fall back to a 1-second reserve only if no cap was provided.
  packetInterleavedSamples_.clear();
  {
    const size_t maxSamplesPerChannel =
        (maxPacketSamplesPerChannel_ > 0) ? maxPacketSamplesPerChannel_
                                          : audioSource_.sampleRateHz();
    packetInterleavedSamples_.reserve(
        maxSamplesPerChannel * static_cast<size_t>(audioSource_.channels()));
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

bool NodeRunner::publishCurrentPacket(
    uint64_t packetEndSampleIndex,
    const EnvironmentalSample* environmentalSample,
    int& lastPublishStatus) {
  if (!packetOpen_ || packetInterleavedSamples_.empty() || packetEndSampleIndex <= packetStartSampleIndex_) {
    packetOpen_ = false;
    packetInterleavedSamples_.clear();
    packetTargetEndSampleIndex_ = 0;
    return false;
  }

  // Stamp both endpoints from the current clock anchor. If a discipline update
  // still yields a non-monotonic interval, keep the packet valid using its
  // sample-count duration; the server rejects frames where end < start.
  uint64_t utcStartNs = clock_.utcAtSampleIndex(packetStartSampleIndex_);
  uint64_t utcEndNs = clock_.utcAtSampleIndex(packetEndSampleIndex);
  if (utcEndNs < utcStartNs) {
    const uint64_t elapsedSamples = packetEndSampleIndex - packetStartSampleIndex_;
    const uint64_t durationNs =
        (elapsedSamples * kNsPerSecond) / static_cast<uint64_t>(audioSource_.sampleRateHz());
    utcEndNs = utcStartNs + durationNs;
  }
  const uint64_t receiptNs = clock_.utcAtMonotonicUs(time_us_64());
  PacketTimingDiagnostics timingDiagnostics = {};
  const bool haveTimingDiagnostics = clock_.currentPacketTimingDiagnostics(timingDiagnostics);
  const AudioFrame frame = {
      utcStartNs,
      utcEndNs,
      packetStartSampleIndex_,
      packetEndSampleIndex,
      audioSource_.sampleRateHz(),
      audioSource_.channels(),
      sequence_,
      utcStartNs,
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

  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  if (publishFailureBackoffMs_ > 0 && !deadlineReached(nowMs, nextPublishAttemptMs_)) {
    lastPublishStatus = kPublishSkippedForBackoffStatus;
    ++stats_.publishErrors;
    ++sequence_;
    packetOpen_ = false;
    packetInterleavedSamples_.clear();
    packetTargetEndSampleIndex_ = 0;
    return false;
  }

  const PublishResult result = publisher_.publish(descriptor_, frame, environmentalSample, false);
  lastPublishStatus = result.statusCode;
  if (result.ok) {
    ++stats_.framesPublished;
    nextPublishAttemptMs_ = 0;
  } else {
    ++stats_.publishErrors;
    if (publishFailureBackoffMs_ > 0) {
      nextPublishAttemptMs_ = nowMs + publishFailureBackoffMs_;
    }
    std::printf(
        "[node] publish failed status=%d seq=%" PRIu64 " samples=%lu utc_start=%" PRIu64 " utc_end=%" PRIu64 "\n",
        result.statusCode,
        sequence_,
        static_cast<unsigned long>(frame.samplesPerChannel),
        utcStartNs,
        utcEndNs);
  }
  ++sequence_;
  packetOpen_ = false;
  packetInterleavedSamples_.clear();
  packetTargetEndSampleIndex_ = 0;
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
      publishCurrentPacket(packetEndSampleIndex, environmentalPtr, lastPublishStatus);
    } else {
      packetOpen_ = false;
      packetInterleavedSamples_.clear();
      packetTargetEndSampleIndex_ = 0;
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
      // Compute the target end-sample-index ONCE at packet open from the
      // current anchor. Storing it as a sample index (not a UTC ns value)
      // makes packet length stable across NTP/GPS anchor jumps. The previous
      // design recomputed sampleIndexAtUtcNs() every iteration, which yielded
      // wildly different lengths after first lock and could cause runaway
      // accumulation or never-closing packets.
      const uint64_t startUtcNs = clock_.utcAtSampleIndex(packetStartSampleIndex_);
      packetTargetEndSampleIndex_ = clock_.sampleIndexAtUtcNs(nextUtcSecondBoundary(startUtcNs));
      if (packetTargetEndSampleIndex_ <= packetStartSampleIndex_) {
        // Degenerate clock state (e.g. anchor not initialized): fall back to a
        // one-second packet sized by nominal sample rate so the runner still
        // makes forward progress and the LED heartbeat stays healthy.
        packetTargetEndSampleIndex_ =
            packetStartSampleIndex_ + static_cast<uint64_t>(audioSource_.sampleRateHz());
      }
      // Cap the packet length so packetInterleavedSamples_ never grows past
      // the reservation made in begin(). UTC-second alignment is preserved on
      // the common path (most seconds will fit under the cap); when a target
      // would overflow, we publish an early-truncated packet and the next
      // packet realigns to the following UTC second on its first chunk.
      if (maxPacketSamplesPerChannel_ > 0) {
        const uint64_t cappedEnd =
            packetStartSampleIndex_ + static_cast<uint64_t>(maxPacketSamplesPerChannel_);
        if (packetTargetEndSampleIndex_ > cappedEnd) {
          packetTargetEndSampleIndex_ = cappedEnd;
        }
      }
      packetInterleavedSamples_.clear();
    }

    uint64_t packetEndSampleIndex = packetTargetEndSampleIndex_;
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
      publishCurrentPacket(packetEndSampleIndex, environmentalPtr, lastPublishStatus);
      packetOpen_ = false;
      packetStartSampleIndex_ = packetEndSampleIndex;
      packetTargetEndSampleIndex_ = 0;
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
