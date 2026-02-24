#pragma once

#include <Arduino.h>

#include "mmpr/Types.h"

namespace mmpr {

class HttpFramePublisher {
 public:
  HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs);

  PublishResult publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody = false);

  const String& endpointUrl() const { return endpointUrl_; }

 private:
  bool parseEndpoint();

  String endpointUrl_;
  String host_;
  String path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;
  uint32_t timeoutMs_;
};

}  // namespace mmpr
