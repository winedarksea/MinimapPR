#pragma once

#include <Arduino.h>

#include "mmpr/Types.h"

namespace mmpr {

const char* nodeTypeToWire(NodeType type);

// Builds payload for POST /api/v1/ingest/frame.
bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, String& outPayload);
bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    String& outPayload);

}  // namespace mmpr
