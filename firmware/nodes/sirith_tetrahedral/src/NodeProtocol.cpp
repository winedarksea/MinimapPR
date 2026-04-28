#include "mmpr/NodeProtocol.h"

#include <inttypes.h>
#include <cstdio>
#include <cstring>
#include <utility>

#include "mmpr/Base64.h"

namespace mmpr {
namespace {

void appendEscapedString(std::string& out, const char* text) {
  if (text == nullptr) {
    return;
  }

  while (*text != '\0') {
    const char c = *text;
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
    ++text;
  }
}

void appendQuoted(std::string& out, const char* text) {
  out += '"';
  appendEscapedString(out, text);
  out += '"';
}

void appendFloat(std::string& out, float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
  out += buffer;
}

void appendUint64(std::string& out, uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%" PRIu64, value);
  out += buffer;
}

void appendUint32(std::string& out, uint32_t value) {
  char buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%" PRIu32, value);
  out += buffer;
}

void appendInt64(std::string& out, int64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%" PRId64, value);
  out += buffer;
}

void appendLeU8(std::string& out, uint8_t value) {
  out.push_back(static_cast<char>(value));
}

void appendLeU16(std::string& out, uint16_t value) {
  out.push_back(static_cast<char>(value & 0xffu));
  out.push_back(static_cast<char>((value >> 8) & 0xffu));
}

void appendLeU32(std::string& out, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<char>((value >> shift) & 0xffu));
  }
}

void appendLeU64(std::string& out, uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<char>((value >> shift) & 0xffu));
  }
}

void appendLeI32(std::string& out, int32_t value) {
  appendLeU32(out, static_cast<uint32_t>(value));
}

void appendLeI64(std::string& out, int64_t value) {
  appendLeU64(out, static_cast<uint64_t>(value));
}

void appendLeF32(std::string& out, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32 bits");
  std::memcpy(&bits, &value, sizeof(bits));
  appendLeU32(out, bits);
}

void appendLeF64(std::string& out, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double must be 64 bits");
  std::memcpy(&bits, &value, sizeof(bits));
  appendLeU64(out, bits);
}

bool appendBinaryString(std::string& out, const char* value) {
  const char* safeValue = value != nullptr ? value : "";
  const size_t length = std::strlen(safeValue);
  if (length > 255) {
    return false;
  }
  appendLeU8(out, static_cast<uint8_t>(length));
  out.append(safeValue, length);
  return true;
}

uint8_t nodeTypeToBinary(NodeType type) {
  switch (type) {
    case NodeType::kSirithTetra:
      return 1;
    case NodeType::kPoint:
    case NodeType::kUnknown:
    default:
      return 0;
  }
}

uint8_t timeQualityToBinary(TimeQuality quality) {
  switch (quality) {
    case TimeQuality::kGpsLocked:
      return 0;
    case TimeQuality::kGpsHoldover:
      return 1;
    case TimeQuality::kNtpDisciplined:
      return 2;
    case TimeQuality::kFreeRunning:
    default:
      return 3;
  }
}

bool appendBinaryNode(std::string& out, const NodeDescriptor& node) {
  if (node.id == nullptr || node.sensorOffsetsM == nullptr || node.sensorCount == 0 ||
      node.sensorCount > 255 || node.capabilityCount > 255) {
    return false;
  }

  if (!appendBinaryString(out, node.id)) {
    return false;
  }
  appendLeU8(out, nodeTypeToBinary(node.type));
  appendLeF32(out, node.positionM.x);
  appendLeF32(out, node.positionM.y);
  appendLeF32(out, node.positionM.z);

  appendLeU8(out, node.hasGeoPosition ? 1 : 0);
  if (node.hasGeoPosition) {
    appendLeF32(out, node.geoPosition.lat);
    appendLeF32(out, node.geoPosition.lon);
    appendLeF32(out, node.geoPosition.altM);
  }

  appendLeU8(out, static_cast<uint8_t>(node.sensorCount));
  for (size_t i = 0; i < node.sensorCount; ++i) {
    appendLeF32(out, node.sensorOffsetsM[i].x);
    appendLeF32(out, node.sensorOffsetsM[i].y);
    appendLeF32(out, node.sensorOffsetsM[i].z);
  }

  appendLeU8(out, static_cast<uint8_t>(node.capabilityCount));
  for (size_t i = 0; i < node.capabilityCount; ++i) {
    if (!appendBinaryString(out, node.capabilities[i])) {
      return false;
    }
  }
  return appendBinaryString(out, node.hardwareName != nullptr ? node.hardwareName : "unknown") &&
         appendBinaryString(out, node.firmwareVersion != nullptr ? node.firmwareVersion : "dev") &&
         appendBinaryString(out, node.gpsSignalStatus != nullptr ? node.gpsSignalStatus : "") &&
         appendBinaryString(out, node.positionSource != nullptr ? node.positionSource : "") &&
         (appendLeU32(out, node.bootCount), true);
}

bool appendBinaryFrameHeader(
    std::string& out,
    const AudioFrame& frame,
    const EnvironmentalSample* environment) {
  if (frame.interleavedSamples == nullptr || frame.channels == 0 ||
      frame.samplesPerChannel == 0 || frame.samplesPerChannel > UINT32_MAX) {
    return false;
  }

  appendLeU64(out, frame.startTimeNs);
  appendLeU64(out, frame.endTimeNs);
  appendLeU64(out, frame.startSampleIndex);
  appendLeU64(out, frame.endSampleIndex);
  appendLeU32(out, frame.sampleRateHz);
  appendLeU8(out, frame.channels);
  appendLeU64(out, frame.sequence);
  appendLeU64(out, frame.toaNs);
  appendLeU64(out, frame.torNs);
  appendLeU8(out, timeQualityToBinary(frame.timeQuality));

  appendLeU8(out, frame.hasTimingDiagnostics ? 1 : 0);
  if (frame.hasTimingDiagnostics) {
    appendLeU8(out, frame.timingHasGpsAnchor ? 1 : 0);
    appendLeU32(out, frame.ppsEdgeCount);
    appendLeU32(out, frame.dmaRingSlotIndex);
    appendLeI64(out, frame.ppsPhaseErrorNs);
    appendLeF64(out, frame.estimatedPpm);
    appendLeU64(out, frame.runnerFramesCaptured);
    appendLeU64(out, frame.runnerFramesDropped);
    appendLeU64(out, frame.runnerContinuityViolations);
    appendLeU64(out, frame.runnerPublishErrors);
    appendLeU32(out, frame.runnerQueueDepth);
    appendLeU64(out, frame.runnerQueueOverflows);
    appendLeI32(out, static_cast<int32_t>(frame.runnerLastPublishStatus));
    appendLeU64(out, frame.packetAgeUs);
    appendLeU8(out, static_cast<uint8_t>(frame.runnerLastPublishFailureStage));
    appendLeI32(out, frame.runnerLastPublishLwipError);
    appendLeU32(out, frame.runnerConsecutivePublishFailures);
    appendLeU64(out, frame.runnerPublishTimeoutFailures);
    appendLeU64(out, frame.runnerPublishConnectOrResetFailures);
    appendLeU64(out, frame.runnerPublishDnsFailures);
    appendLeU64(out, frame.runnerPublishWifiDownFailures);
  }

  uint8_t environmentFlags = 0;
  if (environment != nullptr) {
    environmentFlags |= environment->hasTemperatureC ? 0x01 : 0;
    environmentFlags |= environment->hasHumidityFraction ? 0x02 : 0;
    environmentFlags |= environment->temperatureSource != nullptr ? 0x04 : 0;
  }
  appendLeU8(out, environmentFlags);
  if ((environmentFlags & 0x01) != 0) {
    appendLeF32(out, environment->temperatureC);
  }
  if ((environmentFlags & 0x02) != 0) {
    appendLeF32(out, environment->humidityFraction);
  }
  if ((environmentFlags & 0x04) != 0 && !appendBinaryString(out, environment->temperatureSource)) {
    return false;
  }

  appendLeU32(out, static_cast<uint32_t>(frame.samplesPerChannel));
  return true;
}

}  // namespace

const char* nodeTypeToWire(NodeType type) {
  switch (type) {
    case NodeType::kPoint:
      return "point";
    case NodeType::kSirithTetra:
      return "sirith_tetra";
    default:
      return "unknown";
  }
}

const char* timeQualityToWire(TimeQuality quality) {
  switch (quality) {
    case TimeQuality::kGpsLocked:
      return "gps_locked";
    case TimeQuality::kGpsHoldover:
      return "gps_holdover";
    case TimeQuality::kNtpDisciplined:
      return "ntp_disciplined";
    case TimeQuality::kFreeRunning:
    default:
      return "free_running";
  }
}

const char* publishFailureStageToWire(PublishFailureStage stage) {
  switch (stage) {
    case PublishFailureStage::kDns:
      return "dns";
    case PublishFailureStage::kConnect:
      return "connect";
    case PublishFailureStage::kSend:
      return "send";
    case PublishFailureStage::kRecv:
      return "recv";
    case PublishFailureStage::kResponseParse:
      return "response_parse";
    case PublishFailureStage::kTimeout:
      return "timeout";
    case PublishFailureStage::kWiFiDisconnected:
      return "wifi_disconnected";
    case PublishFailureStage::kNone:
    default:
      return "none";
  }
}

bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, std::string& outPayload) {
  return buildIngestPayload(node, frame, nullptr, outPayload);
}

bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    std::string& outPayload) {
  IngestPayloadParts parts;
  if (!buildIngestPayloadParts(node, frame, environment, parts)) {
    return false;
  }

  outPayload.clear();
  outPayload.reserve(parts.prefix.size() + parts.encodedAudioBytes + parts.suffix.size());
  outPayload += parts.prefix;
  appendBase64(
      outPayload,
      reinterpret_cast<const uint8_t*>(frame.interleavedSamples),
      parts.rawAudioBytes);
  outPayload += parts.suffix;
  return true;
}

bool buildNodePayloadJson(const NodeDescriptor& node, std::string& outPayload) {
  if (node.id == nullptr || node.sensorOffsetsM == nullptr || node.sensorCount == 0) {
    return false;
  }

  outPayload.clear();
  outPayload.reserve(640);

  outPayload += '{';

  outPayload += "\"id\":";
  appendQuoted(outPayload, node.id);

  outPayload += ",\"node_type\":";
  appendQuoted(outPayload, nodeTypeToWire(node.type));

  outPayload += ",\"position_m\":[";
  appendFloat(outPayload, node.positionM.x);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.y);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.z);
  outPayload += ']';

  if (node.hasGeoPosition) {
    outPayload += ",\"position_geo\":{";
    outPayload += "\"lat\":";
    appendFloat(outPayload, node.geoPosition.lat);
    outPayload += ",\"lon\":";
    appendFloat(outPayload, node.geoPosition.lon);
    outPayload += ",\"alt_m\":";
    appendFloat(outPayload, node.geoPosition.altM);
    outPayload += '}';
  }

  outPayload += ",\"sensor_offsets_m\":[";
  for (size_t i = 0; i < node.sensorCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    outPayload += '[';
    appendFloat(outPayload, node.sensorOffsetsM[i].x);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].y);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].z);
    outPayload += ']';
  }
  outPayload += ']';

  outPayload += ",\"capabilities\":[";
  for (size_t i = 0; i < node.capabilityCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    appendQuoted(outPayload, node.capabilities[i]);
  }
  outPayload += ']';

  outPayload += ",\"metadata\":{";
  outPayload += "\"hardware\":";
  appendQuoted(outPayload, node.hardwareName != nullptr ? node.hardwareName : "unknown");
  outPayload += ",\"firmware\":";
  appendQuoted(outPayload, node.firmwareVersion != nullptr ? node.firmwareVersion : "dev");
  outPayload += ",\"boot_count\":";
  appendUint32(outPayload, node.bootCount);
  if (node.gpsSignalStatus != nullptr || node.positionSource != nullptr) {
    outPayload += ",\"gps\":{";
    bool needComma = false;
    if (node.gpsSignalStatus != nullptr) {
      outPayload += "\"signal\":";
      appendQuoted(outPayload, node.gpsSignalStatus);
      needComma = true;
    }
    if (node.positionSource != nullptr) {
      if (needComma) {
        outPayload += ',';
      }
      outPayload += "\"position_source\":";
      appendQuoted(outPayload, node.positionSource);
    }
    outPayload += '}';
  }
  outPayload += "}}";

  return true;
}

bool buildFramePayloadParts(
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    IngestPayloadParts& outParts) {
  if (frame.interleavedSamples == nullptr) {
    return false;
  }

  const size_t bytes = frame.samplesPerChannel * static_cast<size_t>(frame.channels) * sizeof(int16_t);
  const size_t encodedBytes = 4 * ((bytes + 2) / 3);

  outParts.prefix.clear();
  outParts.suffix.clear();
  outParts.rawAudioBytes = bytes;
  outParts.encodedAudioBytes = encodedBytes;

  std::string& outPayload = outParts.prefix;
  outPayload.reserve(512);

  outPayload += "{\"frame\":{";

  outPayload += "\"start_time_ns\":";
  appendUint64(outPayload, frame.startTimeNs);

  // Keep the legacy field until all server deployments consume utc_start_ns as
  // the canonical name. Both currently carry the same UTC anchor.
  outPayload += ",\"utc_start_ns\":";
  appendUint64(outPayload, frame.startTimeNs);

  outPayload += ",\"utc_end_ns\":";
  appendUint64(outPayload, frame.endTimeNs);

  outPayload += ",\"start_sample_index\":";
  appendUint64(outPayload, frame.startSampleIndex);

  outPayload += ",\"end_sample_index\":";
  appendUint64(outPayload, frame.endSampleIndex);

  outPayload += ",\"sample_rate_hz\":";
  appendUint32(outPayload, frame.sampleRateHz);

  outPayload += ",\"channels\":";
  outPayload += std::to_string(static_cast<unsigned>(frame.channels));

  outPayload += ",\"encoding\":\"pcm16le\"";

  outPayload += ",\"samples_per_channel\":";
  appendUint64(outPayload, static_cast<uint64_t>(frame.samplesPerChannel));

  outPayload += ",\"samples_b64\":\"";

  std::string& suffix = outParts.suffix;
  suffix.reserve(512);
  suffix += '"';

  suffix += ",\"sequence\":";
  appendUint64(suffix, frame.sequence);

  suffix += ",\"toa_ns\":";
  appendUint64(suffix, frame.toaNs);

  suffix += ",\"tor_ns\":";
  appendUint64(suffix, frame.torNs);

  suffix += ",\"time_quality\":";
  appendQuoted(suffix, timeQualityToWire(frame.timeQuality));

  if (frame.hasTimingDiagnostics) {
    suffix += ",\"timing_diagnostics\":{";
    suffix += "\"gps_anchor\":";
    suffix += frame.timingHasGpsAnchor ? "true" : "false";
    suffix += ",\"pps_edge_count\":";
    appendUint32(suffix, frame.ppsEdgeCount);
    suffix += ",\"dma_ring_slot_index\":";
    appendUint32(suffix, frame.dmaRingSlotIndex);
    suffix += ",\"pps_phase_error_ns\":";
    appendInt64(suffix, frame.ppsPhaseErrorNs);
    suffix += ",\"estimated_ppm\":";
    char ppmBuffer[32];
    std::snprintf(ppmBuffer, sizeof(ppmBuffer), "%.6f", frame.estimatedPpm);
    suffix += ppmBuffer;
    suffix += ",\"runner_frames_captured\":";
    appendUint64(suffix, frame.runnerFramesCaptured);
    suffix += ",\"runner_frames_dropped\":";
    appendUint64(suffix, frame.runnerFramesDropped);
    suffix += ",\"runner_continuity_violations\":";
    appendUint64(suffix, frame.runnerContinuityViolations);
    suffix += ",\"runner_publish_errors\":";
    appendUint64(suffix, frame.runnerPublishErrors);
    suffix += ",\"runner_queue_depth\":";
    appendUint32(suffix, frame.runnerQueueDepth);
    suffix += ",\"runner_queue_overflows\":";
    appendUint64(suffix, frame.runnerQueueOverflows);
    suffix += ",\"runner_last_publish_status\":";
    suffix += std::to_string(frame.runnerLastPublishStatus);
    suffix += ",\"packet_age_us\":";
    appendUint64(suffix, frame.packetAgeUs);
    suffix += ",\"runner_last_publish_failure_stage\":";
    appendQuoted(suffix, publishFailureStageToWire(frame.runnerLastPublishFailureStage));
    suffix += ",\"runner_last_publish_lwip_error\":";
    suffix += std::to_string(frame.runnerLastPublishLwipError);
    suffix += ",\"runner_consecutive_publish_failures\":";
    appendUint32(suffix, frame.runnerConsecutivePublishFailures);
    suffix += ",\"runner_publish_timeout_failures\":";
    appendUint64(suffix, frame.runnerPublishTimeoutFailures);
    suffix += ",\"runner_publish_connect_or_reset_failures\":";
    appendUint64(suffix, frame.runnerPublishConnectOrResetFailures);
    suffix += ",\"runner_publish_dns_failures\":";
    appendUint64(suffix, frame.runnerPublishDnsFailures);
    suffix += ",\"runner_publish_wifi_down_failures\":";
    appendUint64(suffix, frame.runnerPublishWifiDownFailures);
    suffix += "}";
  }

  suffix += "}";

  if (environment != nullptr && (environment->hasTemperatureC || environment->hasHumidityFraction)) {
    suffix += ",\"environment\":{";
    bool needComma = false;
    if (environment->hasTemperatureC) {
      suffix += "\"temperature_c\":";
      appendFloat(suffix, environment->temperatureC);
      needComma = true;
    }
    if (environment->hasHumidityFraction) {
      if (needComma) {
        suffix += ',';
      }
      suffix += "\"humidity_fraction\":";
      appendFloat(suffix, environment->humidityFraction);
      needComma = true;
    }
    if (environment->temperatureSource != nullptr) {
      if (needComma) {
        suffix += ',';
      }
      suffix += "\"source\":";
      appendQuoted(suffix, environment->temperatureSource);
    }
    suffix += "}";
  }

  suffix += "}";

  return true;
}

bool buildIngestPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    IngestPayloadParts& outParts) {
  std::string nodeJson;
  IngestPayloadParts frameParts;
  if (!buildNodePayloadJson(node, nodeJson) ||
      !buildFramePayloadParts(frame, environment, frameParts)) {
    return false;
  }

  outParts.rawAudioBytes = frameParts.rawAudioBytes;
  outParts.encodedAudioBytes = frameParts.encodedAudioBytes;
  outParts.prefix.clear();
  outParts.suffix.clear();
  outParts.prefix.reserve(16 + nodeJson.size() + frameParts.prefix.size());
  outParts.prefix += "{\"node\":";
  outParts.prefix += nodeJson;
  outParts.prefix += ',';
  outParts.prefix += frameParts.prefix.substr(1);
  outParts.suffix = std::move(frameParts.suffix);
  return true;
}

bool buildStoreForwardPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame* frames,
    const EnvironmentalSample* const* environments,
    size_t frameCount,
    bool sortByToa,
    std::vector<IngestPayloadParts>& outParts) {
  if (frames == nullptr || frameCount == 0) {
    return false;
  }

  std::string nodeJson;
  if (!buildNodePayloadJson(node, nodeJson)) {
    return false;
  }

  outParts.clear();
  outParts.resize(frameCount);

  for (size_t i = 0; i < frameCount; ++i) {
    const EnvironmentalSample* environment =
        environments != nullptr ? environments[i] : nullptr;
    IngestPayloadParts frameParts;
    if (!buildFramePayloadParts(frames[i], environment, frameParts)) {
      outParts.clear();
      return false;
    }

    if (i == 0) {
      outParts[i].prefix.reserve(nodeJson.size() + frameParts.prefix.size() + 96);
      outParts[i].prefix += "{\"node\":";
      outParts[i].prefix += nodeJson;
      outParts[i].prefix += ",\"sort_by_toa\":";
      outParts[i].prefix += sortByToa ? "true" : "false";
      outParts[i].prefix += ",\"buffered_frames\":[";
    }
    outParts[i].prefix += frameParts.prefix;
    outParts[i].suffix = std::move(frameParts.suffix);
    if (i + 1 < frameCount) {
      outParts[i].suffix += ',';
    } else {
      outParts[i].suffix += "]}";
    }
    outParts[i].rawAudioBytes = frameParts.rawAudioBytes;
    outParts[i].encodedAudioBytes = frameParts.encodedAudioBytes;
  }
  return true;
}

bool buildBinaryStoreForwardPayloadParts(
    const NodeDescriptor& node,
    const AudioFrame* frames,
    const EnvironmentalSample* const* environments,
    size_t frameCount,
    bool sortByToa,
    std::vector<IngestPayloadParts>& outParts) {
  if (frames == nullptr || frameCount == 0 || frameCount > UINT16_MAX) {
    return false;
  }

  outParts.clear();
  outParts.resize(frameCount);

  for (size_t i = 0; i < frameCount; ++i) {
    const EnvironmentalSample* environment =
        environments != nullptr ? environments[i] : nullptr;
    IngestPayloadParts& part = outParts[i];
    part.prefix.clear();
    part.suffix.clear();
    part.rawAudioBytes =
        frames[i].samplesPerChannel * static_cast<size_t>(frames[i].channels) * sizeof(int16_t);
    part.encodedAudioBytes = part.rawAudioBytes;

    if (i == 0) {
      part.prefix.reserve(256);
      part.prefix.append("MMB1", 4);
      appendLeU8(part.prefix, 1);
      appendLeU8(part.prefix, sortByToa ? 1 : 0);
      appendLeU16(part.prefix, static_cast<uint16_t>(frameCount));
      if (!appendBinaryNode(part.prefix, node)) {
        outParts.clear();
        return false;
      }
    } else {
      part.prefix.reserve(128);
    }
    if (!appendBinaryFrameHeader(part.prefix, frames[i], environment)) {
      outParts.clear();
      return false;
    }
  }
  return true;
}

}  // namespace mmpr
