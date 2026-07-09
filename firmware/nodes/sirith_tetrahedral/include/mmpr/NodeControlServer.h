#pragma once

#include <cstddef>
#include <cstdint>

#include "lwip/err.h"
#include "mmpr/BleReportPublisher.h"
#include "mmpr/BleRssiScanner.h"
#include "mmpr/NodeRunner.h"

struct pbuf;
struct tcp_pcb;

namespace mmpr {

class HttpFramePublisher;

class NodeControlServer {
 public:
  NodeControlServer(
      HttpFramePublisher& publisher,
      uint16_t listenPort,
      const char* routePath,
      bool allowRuntimePortChange,
      const RunnerStats* runnerStats = nullptr,
      const char* statsPath = "/api/v1/stats",
      const BleScannerStats* bleScannerStats = nullptr,
      const BleReportPublisherStats* bleReportStats = nullptr);
  ~NodeControlServer();

  NodeControlServer(const NodeControlServer&) = delete;
  NodeControlServer& operator=(const NodeControlServer&) = delete;

  bool begin();
  uint16_t listenPort() const { return listenPort_; }
  const char* routePath() const { return routePath_; }

 private:
  static err_t onAcceptStatic(void* arg, tcp_pcb* newPcb, err_t err);
  static err_t onReceiveStatic(void* arg, tcp_pcb* tpcb, pbuf* packet, err_t err);
  static err_t onPollStatic(void* arg, tcp_pcb* tpcb);
  static void onErrorStatic(void* arg, err_t err);

  err_t onAccept(tcp_pcb* newPcb, err_t err);
  err_t onReceive(tcp_pcb* tpcb, pbuf* packet, err_t err);
  err_t onPoll(tcp_pcb* tpcb);
  void onError(err_t err);

  void resetActiveClient();
  err_t closeActiveClient(bool abortConnection);
  bool requestHeadersComplete() const;
  bool prepareStateBody(char* bodyBuffer, size_t bodyBufferBytes, bool includeChangedField, bool changed) const;
  bool prepareStatsBody(char* bodyBuffer, size_t bodyBufferBytes) const;
  bool tryParseRequest(char* method, size_t methodBytes, char* target, size_t targetBytes) const;
  bool tryParsePortQuery(const char* target, uint16_t* outPort) const;
  err_t sendJsonAndClose(tcp_pcb* tpcb, int statusCode, const char* reason, const char* jsonBody);
  err_t handleControlRequest(tcp_pcb* tpcb);

  HttpFramePublisher& publisher_;
  uint16_t listenPort_ = 0;
  const char* routePath_ = nullptr;
  const char* statsPath_ = nullptr;
  bool allowRuntimePortChange_ = false;
  const RunnerStats* runnerStats_ = nullptr;
  const BleScannerStats* bleScannerStats_ = nullptr;
  const BleReportPublisherStats* bleReportStats_ = nullptr;
  tcp_pcb* listenPcb_ = nullptr;
  tcp_pcb* activeClientPcb_ = nullptr;
  size_t requestBytes_ = 0;
  uint8_t idlePolls_ = 0;
  char requestBuffer_[384] = {};
};

}  // namespace mmpr
