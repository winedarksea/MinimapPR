#include "mmpr/BleReportPublisher.h"

#include "mmpr/NodeRunner.h"

#include <algorithm>

#include "pico/time.h"

namespace mmpr {
namespace {

bool deadlineReached(uint32_t nowMs, uint32_t deadlineMs) {
  return static_cast<int32_t>(nowMs - deadlineMs) >= 0;
}

}  // namespace

BleReportPublisher::BleReportPublisher(
    HttpFramePublisher& publisher,
    BleRssiScanner& scanner,
    const NodeDescriptor& node,
    uint32_t reportIntervalMs,
    size_t maxObservations,
    const NodeRunner* audioRunner,
    uint32_t audioQueueHighWaterPackets,
    uint32_t audioQueueLowWaterPackets,
    uint32_t audioPacketAgeLimitMs,
    uint32_t audioRecoveryCooldownMs)
    : publisher_(publisher),
      scanner_(scanner),
      node_(node),
      reportIntervalMs_(reportIntervalMs),
      maxObservations_(std::min<size_t>(maxObservations, 48)),
      audioRunner_(audioRunner),
      audioQueueHighWaterPackets_(audioQueueHighWaterPackets),
      audioQueueLowWaterPackets_(audioQueueLowWaterPackets),
      audioPacketAgeLimitMs_(audioPacketAgeLimitMs),
      audioRecoveryCooldownMs_(audioRecoveryCooldownMs) {}

void BleReportPublisher::poll(uint32_t nowMs, uint32_t bootId) {
  const uint64_t startedUs = time_us_64();
  scanner_.poll();

  PublishResult result = {};
  if (publisher_.pollPublish(result)) {
    publishInFlight_ = false;
    stats_.lastReportStatus = static_cast<uint32_t>(result.statusCode);
    if (result.ok) {
      ++stats_.reportsSent;
    } else {
      ++stats_.reportPublishErrors;
    }
  }

  if (publishInFlight_ || publisher_.publishInProgress()) {
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }
  if (audioRunner_ != nullptr && audioRunner_->shouldDeferAuxiliaryTransport(
          nowMs,
          audioQueueHighWaterPackets_,
          audioQueueLowWaterPackets_,
          audioPacketAgeLimitMs_,
          audioRecoveryCooldownMs_)) {
    ++stats_.reportsDeferredForAudio;
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }
  if (nextReportMs_ != 0 && !deadlineReached(nowMs, nextReportMs_)) {
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }
  nextReportMs_ = nowMs + reportIntervalMs_;

  std::string json;
  uint32_t observationCount = 0;
  if (!buildReportJson(nowMs, bootId, json, observationCount)) {
    ++stats_.reportsDropped;
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }
  stats_.lastReportObservationCount = observationCount;
  if (observationCount == 0) {
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }

  PublishResult immediateResult = {};
  if (publisher_.beginJsonPost(json, false, immediateResult)) {
    publishInFlight_ = true;
    stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
    return;
  }
  stats_.lastReportStatus = static_cast<uint32_t>(immediateResult.statusCode);
  if (!immediateResult.ok) {
    ++stats_.reportPublishErrors;
  }
  stats_.lastPollDurationUs = static_cast<uint32_t>(time_us_64() - startedUs);
}

bool BleReportPublisher::buildReportJson(
    uint32_t nowMs,
    uint32_t bootId,
    std::string& outJson,
    uint32_t& observationCount) {
  BleAdvertiserSnapshot snapshots[48] = {};
  const size_t maxSnapshots = maxObservations_ > 0 ? maxObservations_ : 48;
  const size_t snapshotCount = scanner_.snapshotAndClear(snapshots, maxSnapshots, nowMs);
  observationCount = static_cast<uint32_t>(snapshotCount);
  try {
    outJson.clear();
    outJson.reserve(256 + (snapshotCount * 160));
    outJson += "{\"node_id\":";
    appendEscapedJsonString(outJson, node_.id);
    outJson += ",\"boot_count\":";
    outJson += std::to_string(node_.bootCount);
    outJson += ",\"boot_id\":";
    outJson += std::to_string(bootId);
    outJson += ",\"monotonic_ms\":";
    outJson += std::to_string(nowMs);
    outJson += ",\"observations\":[";
    for (size_t i = 0; i < snapshotCount; ++i) {
      if (i > 0) {
        outJson += ',';
      }
      const BleAdvertiserSnapshot& snapshot = snapshots[i];
      outJson += "{\"mac\":\"";
      appendMac(outJson, snapshot.address);
      outJson += "\",\"addr_type\":";
      outJson += std::to_string(snapshot.addressType);
      outJson += ",\"rssi_last\":";
      outJson += std::to_string(snapshot.rssiLast);
      outJson += ",\"rssi_ewma\":";
      outJson += std::to_string(static_cast<float>(snapshot.rssiEwmaQ8) / 256.0f);
      outJson += ",\"rssi_min\":";
      outJson += std::to_string(snapshot.rssiMin);
      outJson += ",\"rssi_max\":";
      outJson += std::to_string(snapshot.rssiMax);
      outJson += ",\"count\":";
      outJson += std::to_string(snapshot.count);
      outJson += ",\"age_ms\":";
      outJson += std::to_string(nowMs - snapshot.lastSeenMs);
      outJson += ",\"adv_type\":";
      outJson += std::to_string(snapshot.advType);
      if (snapshot.nameHint[0] != '\0') {
        outJson += ",\"name\":";
        appendEscapedJsonString(outJson, snapshot.nameHint);
      }
      outJson += '}';
    }
    outJson += "]}";
    return true;
  } catch (const std::bad_alloc&) {
    outJson.clear();
    observationCount = 0;
    return false;
  }
}

void BleReportPublisher::appendEscapedJsonString(std::string& out, const char* value) const {
  out += '"';
  if (value != nullptr) {
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
      const char ch = *cursor;
      if (ch == '"' || ch == '\\') {
        out += '\\';
        out += ch;
      } else if (static_cast<unsigned char>(ch) >= 0x20) {
        out += ch;
      }
    }
  }
  out += '"';
}

void BleReportPublisher::appendMac(std::string& out, const uint8_t* address) const {
  static constexpr char kHex[] = "0123456789abcdef";
  for (size_t i = 0; i < 6; ++i) {
    if (i > 0) {
      out += ':';
    }
    const uint8_t byte = address[i];
    out += kHex[(byte >> 4) & 0x0f];
    out += kHex[byte & 0x0f];
  }
}

}  // namespace mmpr
