#include "mmpr/NodeProtocol.h"

#include <inttypes.h>
#include <cstdio>

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

bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, std::string& outPayload) {
  return buildIngestPayload(node, frame, nullptr, outPayload);
}

bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    std::string& outPayload) {
  if (node.id == nullptr || node.sensorOffsetsM == nullptr || node.sensorCount == 0 ||
      frame.interleavedSamples == nullptr) {
    return false;
  }

  const size_t bytes = frame.samplesPerChannel * static_cast<size_t>(frame.channels) * sizeof(int16_t);
  const size_t encodedBytes = 4 * ((bytes + 2) / 3);

  outPayload.clear();
  outPayload.reserve(640 + encodedBytes);

  outPayload += "{\"node\":{";

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

  outPayload += ",\"frame\":{";

  outPayload += "\"start_time_ns\":";
  appendUint64(outPayload, frame.startTimeNs);

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
  appendBase64(outPayload, reinterpret_cast<const uint8_t*>(frame.interleavedSamples), bytes);
  outPayload += '"';

  outPayload += ",\"sequence\":";
  appendUint64(outPayload, frame.sequence);

  outPayload += ",\"toa_ns\":";
  appendUint64(outPayload, frame.toaNs);

  outPayload += ",\"tor_ns\":";
  appendUint64(outPayload, frame.torNs);

  outPayload += ",\"time_quality\":";
  appendQuoted(outPayload, timeQualityToWire(frame.timeQuality));

  if (frame.hasTimingDiagnostics) {
    outPayload += ",\"timing_diagnostics\":{";
    outPayload += "\"pps_edge_count\":";
    appendUint32(outPayload, frame.ppsEdgeCount);
    outPayload += ",\"dma_ring_slot_index\":";
    appendUint32(outPayload, frame.dmaRingSlotIndex);
    outPayload += ",\"pps_phase_error_ns\":";
    appendInt64(outPayload, frame.ppsPhaseErrorNs);
    outPayload += ",\"estimated_ppm\":";
    char ppmBuffer[32];
    std::snprintf(ppmBuffer, sizeof(ppmBuffer), "%.6f", frame.estimatedPpm);
    outPayload += ppmBuffer;
    outPayload += "}";
  }

  outPayload += "}";

  if (environment != nullptr && (environment->hasTemperatureC || environment->hasHumidityFraction)) {
    outPayload += ",\"environment\":{";
    bool needComma = false;
    if (environment->hasTemperatureC) {
      outPayload += "\"temperature_c\":";
      appendFloat(outPayload, environment->temperatureC);
      needComma = true;
    }
    if (environment->hasHumidityFraction) {
      if (needComma) {
        outPayload += ',';
      }
      outPayload += "\"humidity_fraction\":";
      appendFloat(outPayload, environment->humidityFraction);
      needComma = true;
    }
    if (environment->temperatureSource != nullptr) {
      if (needComma) {
        outPayload += ',';
      }
      outPayload += "\"source\":";
      appendQuoted(outPayload, environment->temperatureSource);
    }
    outPayload += "}";
  }

  outPayload += "}";

  return true;
}

}  // namespace mmpr
