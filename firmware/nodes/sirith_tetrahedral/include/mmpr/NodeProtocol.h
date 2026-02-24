#pragma once

#include <string>

#include "mmpr/Types.h"

namespace mmpr {

const char* nodeTypeToWire(NodeType type);

bool buildIngestPayload(const NodeDescriptor& node, const AudioFrame& frame, std::string& outPayload);
bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    std::string& outPayload);

}  // namespace mmpr
