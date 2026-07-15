#include "mmpr/EspC5Publisher.h"

#include "pico/time.h"

#include "mmpr/NodeProtocol.h"

namespace mmpr {

namespace {
// Read a chunk at a time while draining the link; small enough to avoid
// starving other loopOnce() work if the C5 trickles bytes, large enough that
// a typical POST_STATUS frame (sync+header+5-byte payload+crc = 16 bytes)
// arrives in one or two reads.
constexpr size_t kResponseReadChunkBytes = 64;
// A POST_STATUS frame's minimum wire size: sync(1) + header(6) + payload(5) + crc(4).
constexpr size_t kMinPostStatusFrameBytes = 1 + kEspC5FrameHeaderSize + 5 + kEspC5FrameCrcSize;
}  // namespace

EspC5Publisher::EspC5Publisher(const char* ingestPath, SpiHostLink& link, uint32_t timeoutMs)
    : link_(link),
      endpointUrl_(std::string("spi://esp32c5") + (ingestPath != nullptr ? ingestPath : "")),
      ingestPath_(ingestPath != nullptr ? ingestPath : ""),
      timeoutMs_(timeoutMs) {}

uint16_t EspC5Publisher::allocateSeq() {
  const uint16_t seq = nextSeq_;
  // Skip 0: reserved as "no sequence" in case a future frame type wants that.
  nextSeq_ = static_cast<uint16_t>(nextSeq_ + 1);
  if (nextSeq_ == 0) {
    nextSeq_ = 1;
  }
  return seq;
}

bool EspC5Publisher::beginBinaryStoreForwardPublish(
    const NodeDescriptor& node,
    const std::vector<AudioFrame>& frames,
    const std::vector<const EnvironmentalSample*>& environments,
    bool sortByToa,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;
  immediateResult.failureStage = PublishFailureStage::kNone;
  immediateResult.lwipError = 0;

  if (state_ != State::kIdle) {
    immediateResult.statusCode = -7;  // publish already in flight
    return false;
  }
  if (frames.empty()) {
    immediateResult.statusCode = -2;
    return false;
  }
  // Note: linkUp_ defaults to false because no LINK_STATUS producer exists
  // yet (see EspC5Frame.h) -- we deliberately don't gate publish on it, so an
  // "unknown" link state behaves as "assume up"; a real down-link is caught
  // by the response timeout in tryDecodeStatusResponse/pollPublish instead.

  std::vector<IngestPayloadParts> payloadParts;
  std::vector<const EnvironmentalSample*> environmentPtrs = environments;
  if (environmentPtrs.size() < frames.size()) {
    environmentPtrs.resize(frames.size(), nullptr);
  }

  std::string body;
  try {
    if (!buildBinaryStoreForwardPayloadParts(
            node,
            frames.data(),
            environmentPtrs.data(),
            frames.size(),
            sortByToa,
            payloadParts)) {
      immediateResult.statusCode = -2;
      return false;
    }

    size_t bodySize = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
      bodySize += payloadParts[i].prefix.size() + payloadParts[i].encodedAudioBytes + payloadParts[i].suffix.size();
    }
    body.reserve(bodySize);
    for (size_t i = 0; i < frames.size(); ++i) {
      const IngestPayloadParts& part = payloadParts[i];
      body.append(part.prefix);
      if (part.encodedAudioBytes > 0) {
        if (frames[i].interleavedSamples == nullptr) {
          immediateResult.statusCode = -2;
          return false;
        }
        body.append(reinterpret_cast<const char*>(frames[i].interleavedSamples), part.encodedAudioBytes);
      }
      body.append(part.suffix);
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  const uint16_t seq = allocateSeq();
  std::vector<uint8_t> encoded;
  try {
    encodeDataPostFrame(seq, ingestPath_, body, encoded);
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  link_.sendFrame(encoded);

  inFlightSeq_ = seq;
  keepResponseBody_ = keepResponseBody;
  rxAccumulator_.clear();
  requestStartedMs_ = to_ms_since_boot(get_absolute_time());
  state_ = State::kAwaitingStatus;
  return true;  // started; caller polls via pollPublish
}

bool EspC5Publisher::tryDecodeStatusResponse(PublishResult& result) {
  // Drain whatever bytes the link has ready right now. Real SPI response
  // timing (how many bytes arrive per read, how promptly wakePin asserts) is
  // hardware-dependent and unverified without a bench; this loop is written
  // to tolerate an arbitrary drip-feed rather than assuming a whole frame
  // lands in one shot.
  while (link_.responseReady() && rxAccumulator_.size() < kEspC5FrameMaxPayloadBytes) {
    const size_t before = rxAccumulator_.size();
    link_.readResponse(rxAccumulator_, kResponseReadChunkBytes);
    if (rxAccumulator_.size() == before) {
      break;  // link had nothing more to give this poll despite wakePin
    }
    if (rxAccumulator_.size() >= kMinPostStatusFrameBytes) {
      break;  // enough for a decode attempt; don't over-read past this frame
    }
  }

  while (!rxAccumulator_.empty()) {
    EspC5Frame frame;
    size_t consumed = 0;
    const EspC5DecodeStatus status = decodeEspC5Frame(rxAccumulator_.data(), rxAccumulator_.size(), frame, consumed);

    switch (status) {
      case EspC5DecodeStatus::kBadSync:
        // Resync: drop one byte of noise and keep looking within this buffer.
        rxAccumulator_.erase(rxAccumulator_.begin());
        continue;

      case EspC5DecodeStatus::kCrcMismatch: {
        // A full-length frame arrived but failed its integrity check. Drop it
        // and report failure rather than silently retrying forever.
        result.ok = false;
        result.statusCode = -5;
        result.failureStage = PublishFailureStage::kResponseParse;
        result.latencyMs = to_ms_since_boot(get_absolute_time()) - requestStartedMs_;
        return true;
      }

      case EspC5DecodeStatus::kBadLength: {
        result.ok = false;
        result.statusCode = -5;
        result.failureStage = PublishFailureStage::kResponseParse;
        result.latencyMs = to_ms_since_boot(get_absolute_time()) - requestStartedMs_;
        return true;
      }

      case EspC5DecodeStatus::kIncomplete:
        return false;  // wait for more bytes / another poll

      case EspC5DecodeStatus::kOk: {
        rxAccumulator_.erase(rxAccumulator_.begin(), rxAccumulator_.begin() + static_cast<long>(consumed));

        if (frame.type == EspC5FrameType::kLinkStatus) {
          LinkStatusPayload linkStatus;
          if (decodeLinkStatusPayload(frame.payload, linkStatus)) {
            applyLinkStatus(linkStatus);
          }
          continue;  // not the response we're waiting on; keep draining
        }

        if (frame.type != EspC5FrameType::kPostStatus || frame.seq != inFlightSeq_) {
          continue;  // stale/foreign frame; discard and keep looking
        }

        bool ok = false;
        int32_t statusCode = -1;
        if (!decodePostStatusPayload(frame.payload, ok, statusCode)) {
          result.ok = false;
          result.statusCode = -5;
          result.failureStage = PublishFailureStage::kResponseParse;
          result.latencyMs = to_ms_since_boot(get_absolute_time()) - requestStartedMs_;
          return true;
        }

        result.ok = ok;
        result.statusCode = statusCode;
        result.failureStage = ok ? PublishFailureStage::kNone : PublishFailureStage::kRecv;
        result.latencyMs = to_ms_since_boot(get_absolute_time()) - requestStartedMs_;
        return true;
      }
    }
  }

  return false;
}

bool EspC5Publisher::pollPublish(PublishResult& result) {
  result = {};
  result.ok = false;
  result.statusCode = -1;
  result.failureStage = PublishFailureStage::kNone;
  result.lwipError = 0;

  if (state_ != State::kAwaitingStatus) {
    return false;
  }

  if (backgroundPollCallback_ != nullptr) {
    backgroundPollCallback_(backgroundPollContext_);
  }

  if (tryDecodeStatusResponse(result)) {
    if (!keepResponseBody_) {
      result.responseBody.clear();
    }
    state_ = State::kIdle;
    rxAccumulator_.clear();
    return true;
  }

  const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
  if (nowMs - requestStartedMs_ >= timeoutMs_) {
    result.ok = false;
    result.statusCode = -4;
    result.failureStage = PublishFailureStage::kTimeout;
    result.latencyMs = nowMs - requestStartedMs_;
    state_ = State::kIdle;
    rxAccumulator_.clear();
    return true;
  }

  return false;
}

bool EspC5Publisher::publishInProgress() const {
  return state_ != State::kIdle;
}

void EspC5Publisher::cancelPublish() {
  state_ = State::kIdle;
  rxAccumulator_.clear();
}

void EspC5Publisher::setBackgroundPollCallback(BackgroundPollCallback callback, void* context) {
  backgroundPollCallback_ = callback;
  backgroundPollContext_ = context;
}

void EspC5Publisher::applyLinkStatus(const LinkStatusPayload& status) {
  linkUp_ = status.linkUp;
  lastRssiDbm_ = status.rssiDbm;
}

}  // namespace mmpr
