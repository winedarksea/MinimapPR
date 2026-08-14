#pragma once

// IUplinkTransport over blocking BSD sockets, executed on a dedicated FreeRTOS
// task.
//
// The CYW43 node's HttpFramePublisher is a 1480-line hand-rolled lwIP raw-TCP
// state machine because the RP2350 has no RTOS and could not afford to block
// the single main loop on the network. With FreeRTOS there is no reason to
// reproduce that: the publisher hands the request to a publisher task, which
// blocks on connect/send/recv while the runner task keeps draining audio.
//
// Wire format is byte-identical to the Pico node's:
//   POST <ingestPath> HTTP/1.1
//   Host: <host>[:<port>]
//   Connection: keep-alive
//   Content-Type: application/octet-stream
//   Content-Length: <n>
// followed by the MMB3 body assembled by the shared
// buildBinaryStoreForwardPayloadParts().
//
// ZERO-COPY / LIFETIME CONTRACT: the audio bytes are not copied. Each
// AudioFrame::interleavedSamples points into NodeRunner's activePublishPackets_,
// which NodeRunner only clears after pollPublish() reports completion, so the
// publisher task may reference them directly for the life of the request. The
// small prefix/suffix strings ARE copied, since they are built per-publish.
//
// PORTABILITY: no ESP32-S3 peripherals here -- see EspWiFiSupport.h for why
// this library must stay linkable by a future ESP32-C5 bridge.

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "mmpr/IUplinkTransport.h"
#include "mmpr/NodeProtocol.h"
#include "mmpr/Types.h"

namespace mmpr {

struct EspHttpPublisherConfig {
  // Full base URL including scheme, host and port, e.g. "http://192.168.8.103:8081".
  const char* serverBaseUrl = nullptr;
  const char* ingestPath = "/api/v1/ingest/binary";
  // Bounds a single publish: applied as SO_SNDTIMEO/SO_RCVTIMEO and as the
  // connect() timeout, so a black-holed link fails within roughly this and the
  // next fresh batch goes out instead of the queue backing up.
  uint32_t timeoutMs = 500;
  // Publisher task placement. Core 0 keeps the socket work off the core running
  // capture + runner (see the node main's task table).
  int taskCoreId = 0;
  uint32_t taskPriority = 5;
  uint32_t taskStackBytes = 6144;
};

class EspHttpFramePublisher final : public IUplinkTransport {
 public:
  explicit EspHttpFramePublisher(const EspHttpPublisherConfig& config);
  ~EspHttpFramePublisher() override;

  EspHttpFramePublisher(const EspHttpFramePublisher&) = delete;
  EspHttpFramePublisher& operator=(const EspHttpFramePublisher&) = delete;

  // Starts the publisher task. Must be called after Wi-Fi is up and before the
  // runner starts publishing.
  bool begin();

  bool beginBinaryStoreForwardPublish(
      const NodeDescriptor& node,
      const std::vector<AudioFrame>& frames,
      const std::vector<const EnvironmentalSample*>& environments,
      bool sortByToa,
      bool keepResponseBody,
      PublishResult& immediateResult) override;

  bool pollPublish(PublishResult& result) override;
  bool publishInProgress() const override;
  void cancelPublish() override;
  void setBackgroundPollCallback(BackgroundPollCallback callback, void* context) override;
  const std::string& endpointUrl() const override { return endpointUrl_; }
  int8_t linkRssiDbm() const override;

  uint16_t endpointPort() const { return port_; }
  bool endpointValid() const { return endpointValid_; }
  // Socket reuses across batches (keep-alive). A climbing value means the
  // server or the link is dropping the connection between publishes.
  uint32_t socketOpenCount() const { return socketOpenCount_.load(std::memory_order_relaxed); }

 private:
  struct AudioSegment {
    const uint8_t* data = nullptr;
    size_t bytes = 0;
  };

  static void taskEntry(void* context);
  void taskLoop();
  void runOneRequest();

  bool parseEndpoint();
  bool ensureSocket(PublishResult& result);
  void closeSocket();
  bool sendAll(const void* data, size_t bytes, PublishResult& result);
  bool sendRequest(PublishResult& result);
  bool readResponse(PublishResult& result);
  void applyStatusPolicy(PublishResult& result, uint32_t retryAfterMs);

  EspHttpPublisherConfig config_;
  std::string endpointUrl_;
  std::string host_;
  std::string path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;

  int socketFd_ = -1;
  std::atomic<uint32_t> socketOpenCount_{0};

  TaskHandle_t taskHandle_ = nullptr;
  SemaphoreHandle_t requestReady_ = nullptr;
  std::atomic<bool> requestInFlight_{false};
  std::atomic<bool> requestComplete_{false};
  std::atomic<bool> cancelRequested_{false};
  std::atomic<bool> stopRequested_{false};

  // Written by the runner task before signalling, read by the publisher task
  // after; the semaphore is the ordering barrier.
  std::string requestHeader_;
  std::vector<IngestPayloadParts> payloadParts_;
  std::vector<AudioSegment> audioSegments_;
  bool keepResponseBody_ = false;
  uint32_t requestStartedMs_ = 0;
  PublishResult pendingResult_ = {};

  // Server-requested pacing from a 503 Retry-After, clamped; honoured by
  // sleeping in the publisher task before the next attempt rather than by
  // failing the batch, so the runner's error counters stay meaningful.
  uint32_t retryAfterMs_ = 0;

  std::string responseScratch_;
};

}  // namespace mmpr
