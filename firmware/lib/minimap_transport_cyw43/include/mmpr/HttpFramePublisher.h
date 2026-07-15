#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mmpr/IUplinkTransport.h"
#include "mmpr/Types.h"

namespace mmpr {

class HttpFramePublisher : public IUplinkTransport {
 public:
  struct TransportState;
  using BackgroundPollCallback = IUplinkTransport::BackgroundPollCallback;

  HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs);
  ~HttpFramePublisher();

  HttpFramePublisher(const HttpFramePublisher&) = delete;
  HttpFramePublisher& operator=(const HttpFramePublisher&) = delete;

  PublishResult publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody = false);
  PublishResult publish(
      const NodeDescriptor& node,
      const AudioFrame& frame,
      const EnvironmentalSample* environment,
      bool keepResponseBody = false);

  bool beginPublish(
      const NodeDescriptor& node,
      const AudioFrame& frame,
      const EnvironmentalSample* environment,
      bool keepResponseBody,
      PublishResult& immediateResult);
  bool beginStoreForwardPublish(
      const NodeDescriptor& node,
      const std::vector<AudioFrame>& frames,
      const std::vector<const EnvironmentalSample*>& environments,
      bool sortByToa,
      bool keepResponseBody,
      PublishResult& immediateResult);
  bool beginBinaryStoreForwardPublish(
      const NodeDescriptor& node,
      const std::vector<AudioFrame>& frames,
      const std::vector<const EnvironmentalSample*>& environments,
      bool sortByToa,
      bool keepResponseBody,
      PublishResult& immediateResult) override;
  bool beginJsonPost(
      const std::string& jsonBody,
      bool keepResponseBody,
      PublishResult& immediateResult);
  bool pollPublish(PublishResult& result) override;
  bool publishInProgress() const override;
  void cancelPublish() override;
    bool setEndpointPort(uint16_t port);

  const std::string& endpointUrl() const override { return endpointUrl_; }
    uint16_t endpointPort() const { return port_; }
    bool endpointValid() const { return endpointValid_; }
  void setBackgroundPollCallback(BackgroundPollCallback callback, void* context) override;

 private:
  bool parseEndpoint();
  static void trimAsciiWhitespace(std::string& s);

  std::string endpointUrl_;
  std::string host_;
  std::string path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;
  uint32_t timeoutMs_ = 0;
  TransportState* transportState_ = nullptr;
};

}  // namespace mmpr
