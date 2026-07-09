#include "mmpr/NodeRunner.h"

#include <cstdio>
#include <inttypes.h>
#include <new>
#include <algorithm>
#include <vector>

#include "pico/cyw43_arch.h"
#include "pico/time.h"

namespace mmpr {
namespace {

constexpr uint64_t kNsPerSecond = 1000000000ULL;
constexpr int kPublishSkippedForBackoffStatus = -5;
constexpr size_t kDefaultQueuedPacketSlots = 40;
constexpr size_t kDefaultPublishBatchByteBudget = 20480;
constexpr uint32_t kQueueDepthBypassThreshold = 32;
constexpr uint32_t kSlowTelemetryRefreshMs = 1000;

uint64_t nextUtcSecondBoundary(uint64_t utcNs) {
  return ((utcNs / kNsPerSecond) + 1ULL) * kNsPerSecond;
}

bool deadlineReached(uint32_t nowMs, uint32_t deadlineMs) {
  return static_cast<int32_t>(nowMs - deadlineMs) >= 0;
}

bool isWiFiConnected() {
  return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA) == CYW43_LINK_UP;
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
    uint32_t publishFailureBackoffMs,
    size_t publishBatchByteBudget,
    size_t queueSlots)
    : descriptor_(descriptor),
      audioSource_(audioSource),
      publisher_(publisher),
      environmentalSource_(environmentalSource),
      clock_(clock),
      logEveryFrames_(logEveryFrames),
      maxPacketSamplesPerChannel_(maxPacketSamplesPerChannel),
      publishFailureBackoffMs_(publishFailureBackoffMs),
      publishBatchByteBudget_(publishBatchByteBudget > 0 ? publishBatchByteBudget : kDefaultPublishBatchByteBudget) {
  stats_.queueSlotsCapacity = static_cast<uint32_t>(queueSlots > 0 ? queueSlots : kDefaultQueuedPacketSlots);
}

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
  stats_.bootId = static_cast<uint32_t>(time_us_64() ^ (reinterpret_cast<uintptr_t>(this) >> 2));

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
    const size_t queueSlots = std::max<size_t>(1, stats_.queueSlotsCapacity);
    queuedPackets_.resize(queueSlots);
    for (QueuedAudioPacket& packet : queuedPackets_) {
      packet.interleavedSamples.reserve(
          maxSamplesPerChannel * static_cast<size_t>(audioSource_.channels()));
    }
    const size_t packetBytes =
        maxSamplesPerChannel * static_cast<size_t>(audioSource_.channels()) * sizeof(int16_t);
    const size_t maxActivePacketsByBudget =
        packetBytes > 0 ? std::max<size_t>(1, publishBatchByteBudget_ / packetBytes) : 1;
    activePublishPackets_.resize(std::min(queueSlots, maxActivePacketsByBudget));
    for (QueuedAudioPacket& packet : activePublishPackets_) {
      packet.interleavedSamples.reserve(
          maxSamplesPerChannel * static_cast<size_t>(audioSource_.channels()));
    }
  }
  stats_.queueSlotsCapacity = static_cast<uint32_t>(queuedPackets_.size());
  stats_.ringFramesCapacity = audioSource_.ringFramesCapacity();
  stats_.ringFramesHighWater = audioSource_.ringFramesHighWater();

  std::printf(
      "[node] started id=%s channels=%u sample_rate=%lu frame_samples=%u endpoint=%s\n",
      descriptor_.id,
      static_cast<unsigned>(audioSource_.channels()),
      static_cast<unsigned long>(audioSource_.sampleRateHz()),
      static_cast<unsigned>(audioSource_.frameSamples()),
      publisher_.endpointUrl().c_str());

  return true;
}

bool NodeRunner::enqueueCurrentPacket(
    uint64_t packetEndSampleIndex,
    const EnvironmentalSample* environmentalSample) {
  if (!packetOpen_ || packetInterleavedSamples_.empty() || packetEndSampleIndex <= packetStartSampleIndex_) {
    packetOpen_ = false;
    packetInterleavedSamples_.clear();
    packetTargetEndSampleIndex_ = 0;
    packetStartMonotonicUs_ = 0;
    return false;
  }

  if (queuedPackets_.empty()) {
    packetOpen_ = false;
    packetInterleavedSamples_.clear();
    packetTargetEndSampleIndex_ = 0;
    packetStartMonotonicUs_ = 0;
    return false;
  }
  if (queueDepth_ >= queuedPackets_.size()) {
    dropOldestQueuedPacket();
  }
  const size_t writeIndex = (queueHead_ + queueDepth_) % queuedPackets_.size();
  QueuedAudioPacket& packet = queuedPackets_[writeIndex];
  packet.interleavedSamples.assign(packetInterleavedSamples_.begin(), packetInterleavedSamples_.end());
  packet.startSampleIndex = packetStartSampleIndex_;
  packet.endSampleIndex = packetEndSampleIndex;
  packet.startMonotonicUs = packetStartMonotonicUs_;
  packet.sequence = sequence_++;
  packet.hasEnvironmentalSample = false;
  if (environmentalSample != nullptr) {
    packet.hasEnvironmentalSample = true;
    packet.environmentalSample = *environmentalSample;
  }
  ++queueDepth_;
  stats_.queueDepth = effectiveQueueDepth();
  if (stats_.queueDepth > stats_.queueSlotsHighWater) {
    stats_.queueSlotsHighWater = stats_.queueDepth;
  }

  packetOpen_ = false;
  packetInterleavedSamples_.clear();
  packetTargetEndSampleIndex_ = 0;
  packetStartMonotonicUs_ = 0;
  return true;
}

void NodeRunner::dropOldestQueuedPacket() {
  if (queueDepth_ == 0 || queuedPackets_.empty()) {
    return;
  }
  queueHead_ = (queueHead_ + 1u) % queuedPackets_.size();
  --queueDepth_;
  ++stats_.queueOverflows;
}

bool NodeRunner::popQueuedPacket(QueuedAudioPacket& packet) {
  if (queueDepth_ == 0 || queuedPackets_.empty()) {
    return false;
  }
  packet = queuedPackets_[queueHead_];
  queuedPackets_[queueHead_].interleavedSamples.clear();
  queuedPackets_[queueHead_].hasEnvironmentalSample = false;
  queueHead_ = (queueHead_ + 1u) % queuedPackets_.size();
  --queueDepth_;
  stats_.queueDepth = effectiveQueueDepth();
  return true;
}

uint32_t NodeRunner::effectiveQueueDepth() const {
  return static_cast<uint32_t>(queueDepth_ + (activePublishBatchValid_ ? activePublishBatchSize_ : 0u));
}

AudioFrame NodeRunner::buildFrameForPacket(const QueuedAudioPacket& packet, uint64_t publishMonotonicUs) const {
  // Stamp TOA from the current UTC epoch plus the DMA-captured monotonic age of
  // the packet. This keeps recent audio tied to the clock correction that is
  // valid at publish time while preserving the precise capture offset.
  const uint64_t elapsedSamples = packet.endSampleIndex - packet.startSampleIndex;
  const uint64_t durationNs =
      (elapsedSamples * kNsPerSecond) / static_cast<uint64_t>(audioSource_.sampleRateHz());
  const uint64_t receiptNs = clock_.utcAtMonotonicUs(publishMonotonicUs);
  uint64_t packetAgeNs = 0;
  uint64_t utcStartNs = 0;
  uint64_t utcEndNs = 0;
  if (packet.startMonotonicUs > 0 && publishMonotonicUs >= packet.startMonotonicUs) {
    packetAgeNs = (publishMonotonicUs - packet.startMonotonicUs) * 1000ULL;
    utcStartNs = receiptNs > packetAgeNs ? receiptNs - packetAgeNs : 0;
    utcEndNs = utcStartNs + durationNs;
  } else {
    utcStartNs = clock_.utcAtSampleIndex(packet.startSampleIndex);
    utcEndNs = clock_.utcAtSampleIndex(packet.endSampleIndex);
    if (utcEndNs < utcStartNs) {
      utcEndNs = utcStartNs + durationNs;
    }
  }
  PacketTimingDiagnostics timingDiagnostics = {};
  const bool haveTimingDiagnostics = clock_.currentPacketTimingDiagnostics(timingDiagnostics);
  const AudioFrame frame = {
      utcStartNs,
      utcEndNs,
      packet.startSampleIndex,
      packet.endSampleIndex,
      audioSource_.sampleRateHz(),
      audioSource_.channels(),
      audioSource_.sourceType(),
      packet.sequence,
      utcStartNs,
      receiptNs,
      clock_.timeQuality(),
      haveTimingDiagnostics,
      timingDiagnostics.hasGpsAnchor,
      timingDiagnostics.ppsEdgeCount,
      timingDiagnostics.dmaRingSlotIndex,
      timingDiagnostics.ppsPhaseErrorNs,
      timingDiagnostics.estimatedPpm,
      packet.interleavedSamples.data(),
      packet.interleavedSamples.size() / static_cast<size_t>(audioSource_.channels()),
      stats_.framesCaptured,
      stats_.framesDropped,
      stats_.packetContinuityViolations,
      stats_.publishErrors,
      effectiveQueueDepth(),
      stats_.queueOverflows,
      stats_.lastPublishStatus,
      packetAgeNs / 1000ULL,
      stats_.lastPublishFailureStage,
      stats_.lastPublishLwipError,
      stats_.consecutivePublishFailures,
      stats_.publishTimeoutFailures,
      stats_.publishConnectOrResetFailures,
      stats_.publishDnsFailures,
      stats_.publishWifiDownFailures,
      audioSource_.ringFramesHighWater(),
      audioSource_.ringFramesCapacity(),
      stats_.queueSlotsHighWater,
      stats_.queueSlotsCapacity,
      stats_.publishLatencyLastMs,
      stats_.publishLatencyEwmaMs,
      stats_.publishLatencyMaxMs,
      stats_.wifiRssiDbm,
      stats_.heapFreeBytes,
      stats_.bootId,
  };
  return frame;
}

bool NodeRunner::fillActivePublishBatch() {
  if (activePublishBatchValid_) {
    return true;
  }
  if (queueDepth_ == 0) {
    activePublishBatchSize_ = 0;
    return false;
  }

  activePublishBatchSize_ = 0;
  size_t accumulatedAudioBytes = 0;
  while (queueDepth_ > 0 && activePublishBatchSize_ < activePublishPackets_.size()) {
    QueuedAudioPacket& packet = activePublishPackets_[activePublishBatchSize_];
    if (!popQueuedPacket(packet)) {
      break;
    }
    const size_t packetBytes = packet.interleavedSamples.size() * sizeof(int16_t);
    if (activePublishBatchSize_ > 0 &&
        accumulatedAudioBytes + packetBytes > publishBatchByteBudget_) {
      const size_t writeIndex = (queueHead_ + queuedPackets_.size() - 1u) % queuedPackets_.size();
      queuedPackets_[writeIndex] = packet;
      queueHead_ = writeIndex;
      ++queueDepth_;
      break;
    }
    accumulatedAudioBytes += packetBytes;
    ++activePublishBatchSize_;
  }
  activePublishBatchValid_ = activePublishBatchSize_ > 0;
  stats_.queueDepth = effectiveQueueDepth();
  return activePublishBatchValid_;
}

void NodeRunner::clearActivePublishBatch() {
  for (size_t i = 0; i < activePublishBatchSize_ && i < activePublishPackets_.size(); ++i) {
    activePublishPackets_[i].interleavedSamples.clear();
    activePublishPackets_[i].hasEnvironmentalSample = false;
  }
  activePublishBatchSize_ = 0;
  activePublishBatchValid_ = false;
  stats_.queueDepth = effectiveQueueDepth();
}

uint32_t NodeRunner::adaptivePublishBackoffMs() const {
  const uint32_t failures = stats_.consecutivePublishFailures;
  if (failures <= 1) {
    return 0;
  }
  if (failures == 2) {
    return 25;
  }
  if (failures == 3) {
    return 50;
  }
  if (failures == 4) {
    return 100;
  }
  return 200;
}

bool NodeRunner::shouldBypassAdaptiveBackoff() const {
  // Never bypass backoff for HTTP 4xx responses. The server is explicitly
  // rejecting the request (e.g., 410 Gone when direct ingest is disabled),
  // so retrying immediately just saturates CPU and Wi-Fi without making
  // forward progress. Transient transport failures still bypass when the
  // queue is deep so we can drain before audio is dropped.
  if (stats_.lastPublishStatus >= 400 && stats_.lastPublishStatus < 500) {
    return false;
  }
  return effectiveQueueDepth() >= kQueueDepthBypassThreshold;
}

void NodeRunner::refreshSlowTelemetry(uint32_t nowMs) {
  stats_.ringFramesCapacity = audioSource_.ringFramesCapacity();
  stats_.ringFramesHighWater = audioSource_.ringFramesHighWater();

  if (lastTelemetryRefreshMs_ != 0 &&
      !deadlineReached(nowMs, lastTelemetryRefreshMs_ + kSlowTelemetryRefreshMs)) {
    return;
  }
  lastTelemetryRefreshMs_ = nowMs;

  if (!isWiFiConnected()) {
    stats_.wifiRssiDbm = 0;
    return;
  }

  int32_t rssiDbm = 0;
  if (cyw43_wifi_get_rssi(&cyw43_state, &rssiDbm) == 0) {
    if (rssiDbm < -128) {
      rssiDbm = -128;
    } else if (rssiDbm > 127) {
      rssiDbm = 127;
    }
    stats_.wifiRssiDbm = static_cast<int8_t>(rssiDbm);
  }
}

void NodeRunner::onPublishSuccess() {
  stats_.framesPublished += activePublishBatchSize_;
  stats_.consecutivePublishFailures = 0;
  stats_.lastPublishFailureStage = PublishFailureStage::kNone;
  stats_.lastPublishLwipError = 0;
  nextPublishAttemptMs_ = 0;
  clearActivePublishBatch();
}

void NodeRunner::onPublishFailure(const PublishResult& result, uint32_t nowMs) {
  ++stats_.publishErrors;
  ++stats_.consecutivePublishFailures;
  stats_.lastPublishFailureStage = result.failureStage;
  stats_.lastPublishLwipError = result.lwipError;

  switch (result.failureStage) {
    case PublishFailureStage::kDns:
      ++stats_.publishDnsFailures;
      break;
    case PublishFailureStage::kTimeout:
      ++stats_.publishTimeoutFailures;
      break;
    case PublishFailureStage::kWiFiDisconnected:
      ++stats_.publishWifiDownFailures;
      break;
    case PublishFailureStage::kConnect:
    case PublishFailureStage::kRecv:
    case PublishFailureStage::kSend:
      ++stats_.publishConnectOrResetFailures;
      break;
    default:
      break;
  }

  uint32_t backoffMs = adaptivePublishBackoffMs();
  if (publishFailureBackoffMs_ > 0 && backoffMs < publishFailureBackoffMs_) {
    backoffMs = publishFailureBackoffMs_;
  }
  if (shouldBypassAdaptiveBackoff()) {
    backoffMs = 0;
  }
  nextPublishAttemptMs_ = backoffMs > 0 ? (nowMs + backoffMs) : 0;

  const uint64_t firstSequence =
      activePublishBatchSize_ > 0 ? activePublishPackets_[0].sequence : 0;
  std::printf(
      "[node] publish failed status=%d stage=%u lwip=%d seq=%" PRIu64
      " batch=%lu queue=%u backoff_ms=%lu\n",
      result.statusCode,
      static_cast<unsigned>(result.failureStage),
      static_cast<int>(result.lwipError),
      firstSequence,
      static_cast<unsigned long>(activePublishBatchSize_),
      static_cast<unsigned>(effectiveQueueDepth()),
      static_cast<unsigned long>(backoffMs));
  stats_.queueDepth = effectiveQueueDepth();
}

void NodeRunner::pumpPublisher() {
  PublishResult result = {};
  if (publisher_.pollPublish(result)) {
    stats_.lastPublishStatus = result.statusCode;
    stats_.publishLatencyLastMs = result.latencyMs;
    if (result.latencyMs > stats_.publishLatencyMaxMs) {
      stats_.publishLatencyMaxMs = result.latencyMs;
    }
    if (stats_.publishLatencyEwmaMs == 0) {
      stats_.publishLatencyEwmaMs = result.latencyMs;
    } else {
      stats_.publishLatencyEwmaMs =
          ((stats_.publishLatencyEwmaMs * 7u) + result.latencyMs + 4u) / 8u;
    }
    if (result.ok) {
      onPublishSuccess();
    } else {
      onPublishFailure(result, to_ms_since_boot(get_absolute_time()));
    }
  }
  startNextPublishIfPossible();
}

void NodeRunner::startNextPublishIfPossible() {
  if (publisher_.publishInProgress()) {
    return;
  }
  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  if (nextPublishAttemptMs_ > 0 && !deadlineReached(nowMs, nextPublishAttemptMs_)) {
    stats_.lastPublishStatus = kPublishSkippedForBackoffStatus;
    stats_.queueDepth = effectiveQueueDepth();
    return;
  }

  if (!fillActivePublishBatch()) {
    stats_.queueDepth = effectiveQueueDepth();
    return;
  }
  const uint64_t publishMonotonicUs = time_us_64();
  std::vector<AudioFrame> frames;
  std::vector<const EnvironmentalSample*> environments;
  try {
    frames.reserve(activePublishBatchSize_);
    environments.reserve(activePublishBatchSize_);
    for (size_t i = 0; i < activePublishBatchSize_; ++i) {
      frames.push_back(buildFrameForPacket(activePublishPackets_[i], publishMonotonicUs));
      environments.push_back(
          activePublishPackets_[i].hasEnvironmentalSample
              ? &activePublishPackets_[i].environmentalSample
              : nullptr);
    }
  } catch (const std::bad_alloc&) {
    PublishResult allocFailure = {};
    allocFailure.ok = false;
    allocFailure.statusCode = -6;
    allocFailure.failureStage = PublishFailureStage::kResponseParse;
    allocFailure.lwipError = 0;
    stats_.lastPublishStatus = allocFailure.statusCode;
    onPublishFailure(allocFailure, nowMs);
    return;
  }
  PublishResult immediateResult = {};
  if (publisher_.beginBinaryStoreForwardPublish(descriptor_, frames, environments, false, false, immediateResult)) {
    stats_.queueDepth = effectiveQueueDepth();
    return;
  }

  stats_.lastPublishStatus = immediateResult.statusCode;
  if (immediateResult.ok) {
    onPublishSuccess();
  } else {
    onPublishFailure(immediateResult, nowMs);
  }
  stats_.queueDepth = effectiveQueueDepth();
  stats_.ringFramesCapacity = audioSource_.ringFramesCapacity();
  stats_.ringFramesHighWater = audioSource_.ringFramesHighWater();
}

void NodeRunner::processCapturedFrame(
    const AudioCaptureTimestamp& captureTimestamp,
    const EnvironmentalSample* environmentalPtr) {
  const uint8_t channels = audioSource_.channels();
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
      enqueueCurrentPacket(packetEndSampleIndex, environmentalPtr);
    } else {
      packetOpen_ = false;
      packetInterleavedSamples_.clear();
      packetTargetEndSampleIndex_ = 0;
      packetStartMonotonicUs_ = 0;
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
      packetStartMonotonicUs_ =
          captureTimestamp.blockStartMonotonicUs +
          ((frameSampleOffset * 1000000ULL) / static_cast<uint64_t>(audioSource_.sampleRateHz()));
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
      enqueueCurrentPacket(packetEndSampleIndex, environmentalPtr);
      packetOpen_ = false;
      packetStartSampleIndex_ = packetEndSampleIndex;
      packetStartMonotonicUs_ = 0;
      packetTargetEndSampleIndex_ = 0;
    }
  }
}

void NodeRunner::drainAvailableAudioFrames() {
  if (frameBuffer_ == nullptr) {
    return;
  }

  bool drainedAnyFrame = false;
  while (true) {
    AudioCaptureTimestamp captureTimestamp = {};
    const bool frameOk =
        audioSource_.readFrameNonblocking(frameBuffer_, audioSource_.frameSamples(), &captureTimestamp);
    if (!frameOk) {
      break;
    }

    drainedAnyFrame = true;
    clock_.observeCapturedAudio(captureTimestamp);
    ++stats_.framesCaptured;
    stats_.framesDropped += captureTimestamp.droppedBlocksBeforeCapture;

    EnvironmentalSample environmental = {};
    const EnvironmentalSample* environmentalPtr = nullptr;
    if (environmentalSourceReady_ && environmentalSource_ != nullptr && environmentalSource_->read(environmental)) {
      environmentalPtr = &environmental;
    }
    processCapturedFrame(captureTimestamp, environmentalPtr);
  }

  if (!drainedAnyFrame) {
    sleep_ms(1);
  }
}

void NodeRunner::loopOnce() {
  if (frameBuffer_ == nullptr) {
    return;
  }

  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  refreshSlowTelemetry(nowMs);
  pumpPublisher();
  drainAvailableAudioFrames();
  pumpPublisher();

  if (logEveryFrames_ > 0 &&
      stats_.framesCaptured != lastLoggedFrameCount_ &&
      (stats_.framesCaptured % logEveryFrames_) == 0) {
    lastLoggedFrameCount_ = stats_.framesCaptured;
    std::printf(
        "[node] blocks=%" PRIu64 " published=%" PRIu64 " dropped=%" PRIu64
        " continuity=%" PRIu64 " errors=%" PRIu64 " queue=%u overflows=%" PRIu64
        " last_status=%d\n",
        stats_.framesCaptured,
        stats_.framesPublished,
        stats_.framesDropped,
        stats_.packetContinuityViolations,
        stats_.publishErrors,
        static_cast<unsigned>(stats_.queueDepth),
        stats_.queueOverflows,
        stats_.lastPublishStatus);
  }
}

}  // namespace mmpr
