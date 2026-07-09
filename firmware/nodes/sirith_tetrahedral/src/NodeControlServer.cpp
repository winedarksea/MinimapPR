#include "mmpr/NodeControlServer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "lwip/pbuf.h"
#include "lwip/tcp.h"

#include "mmpr/HttpFramePublisher.h"

namespace mmpr {
namespace {

constexpr size_t kMaxMethodBytes = 8;
constexpr size_t kMaxTargetBytes = 192;
constexpr size_t kMaxBodyBytes = 2048;
constexpr size_t kMaxResponseBytes = 2048;
constexpr uint8_t kIdlePollAbortThreshold = 12;

bool appendCopy(char* dest, size_t destBytes, const char* src, size_t srcBytes) {
  if (dest == nullptr || src == nullptr || destBytes == 0 || srcBytes >= destBytes) {
    return false;
  }
  std::memcpy(dest, src, srcBytes);
  dest[srcBytes] = '\0';
  return true;
}

const char* boolJson(bool value) {
  return value ? "true" : "false";
}

const char* gpsFixStatusJson(const GpsRuntimeStats& gps) {
  if (!gps.uartStarted || !gps.nmeaHealthy) {
    return "missing";
  }
  if (!gps.hasFix) {
    return "no_fix";
  }
  if (gps.fixDimension >= 3) {
    return "fix_3d";
  }
  if (gps.fixDimension == 2) {
    return "fix_2d";
  }
  return "fix";
}

const char* gpsPpsStatusJson(const GpsRuntimeStats& gps) {
  if (!gps.ppsConfigured) {
    return "unconfigured";
  }
  if (!gps.ppsObserved) {
    return "missing";
  }
  if (!gps.ppsEpochAligned) {
    return "unanchored";
  }
  return "anchored";
}

const char* timeQualityJson(TimeQuality quality) {
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

}  // namespace

NodeControlServer::NodeControlServer(
    HttpFramePublisher& publisher,
    uint16_t listenPort,
    const char* routePath,
    bool allowRuntimePortChange,
    const RunnerStats* runnerStats,
    const char* statsPath,
    const BleScannerStats* bleScannerStats,
    const BleReportPublisherStats* bleReportStats,
    const GpsRuntimeStats* gpsStats)
    : publisher_(publisher),
      listenPort_(listenPort),
      routePath_(routePath != nullptr ? routePath : "/api/v1/publish-target"),
      statsPath_(statsPath != nullptr ? statsPath : "/api/v1/stats"),
      allowRuntimePortChange_(allowRuntimePortChange),
      runnerStats_(runnerStats),
      bleScannerStats_(bleScannerStats),
      bleReportStats_(bleReportStats),
      gpsStats_(gpsStats) {}

NodeControlServer::~NodeControlServer() {
  closeActiveClient(true);
  if (listenPcb_ != nullptr) {
    tcp_arg(listenPcb_, nullptr);
    tcp_accept(listenPcb_, nullptr);
    const err_t closeErr = tcp_close(listenPcb_);
    if (closeErr != ERR_OK) {
      tcp_abort(listenPcb_);
    }
    listenPcb_ = nullptr;
  }
}

bool NodeControlServer::begin() {
  if (listenPcb_ != nullptr) {
    return true;
  }
  if (listenPort_ == 0 || routePath_ == nullptr || routePath_[0] != '/') {
    return false;
  }

  tcp_pcb* pcb = tcp_new();
  if (pcb == nullptr) {
    return false;
  }
  if (tcp_bind(pcb, IP_ADDR_ANY, listenPort_) != ERR_OK) {
    tcp_abort(pcb);
    return false;
  }

  listenPcb_ = tcp_listen_with_backlog(pcb, 1);
  if (listenPcb_ == nullptr) {
    return false;
  }

  tcp_arg(listenPcb_, this);
  tcp_accept(listenPcb_, &NodeControlServer::onAcceptStatic);
  std::printf(
      "[control] listening port=%u path=%s runtime_port_change=%u\n",
      static_cast<unsigned>(listenPort_),
      routePath_,
      static_cast<unsigned>(allowRuntimePortChange_ ? 1u : 0u));
  return true;
}

err_t NodeControlServer::onAcceptStatic(void* arg, tcp_pcb* newPcb, err_t err) {
  NodeControlServer* server = static_cast<NodeControlServer*>(arg);
  return server != nullptr ? server->onAccept(newPcb, err) : ERR_VAL;
}

err_t NodeControlServer::onReceiveStatic(void* arg, tcp_pcb* tpcb, pbuf* packet, err_t err) {
  NodeControlServer* server = static_cast<NodeControlServer*>(arg);
  return server != nullptr ? server->onReceive(tpcb, packet, err) : ERR_VAL;
}

err_t NodeControlServer::onPollStatic(void* arg, tcp_pcb* tpcb) {
  NodeControlServer* server = static_cast<NodeControlServer*>(arg);
  return server != nullptr ? server->onPoll(tpcb) : ERR_VAL;
}

void NodeControlServer::onErrorStatic(void* arg, err_t err) {
  NodeControlServer* server = static_cast<NodeControlServer*>(arg);
  if (server != nullptr) {
    server->onError(err);
  }
}

err_t NodeControlServer::onAccept(tcp_pcb* newPcb, err_t err) {
  if (err != ERR_OK || newPcb == nullptr) {
    if (newPcb != nullptr) {
      tcp_abort(newPcb);
      return ERR_ABRT;
    }
    return err;
  }

  if (activeClientPcb_ != nullptr) {
    tcp_abort(newPcb);
    return ERR_ABRT;
  }

  activeClientPcb_ = newPcb;
  resetActiveClient();
  tcp_arg(newPcb, this);
  tcp_recv(newPcb, &NodeControlServer::onReceiveStatic);
  tcp_poll(newPcb, &NodeControlServer::onPollStatic, 2);
  tcp_err(newPcb, &NodeControlServer::onErrorStatic);
  return ERR_OK;
}

err_t NodeControlServer::onReceive(tcp_pcb* tpcb, pbuf* packet, err_t err) {
  if (tpcb == nullptr || tpcb != activeClientPcb_) {
    if (packet != nullptr) {
      pbuf_free(packet);
    }
    return ERR_OK;
  }
  if (err != ERR_OK) {
    if (packet != nullptr) {
      pbuf_free(packet);
    }
    closeActiveClient(true);
    return ERR_ABRT;
  }
  if (packet == nullptr) {
    closeActiveClient(false);
    return ERR_OK;
  }

  idlePolls_ = 0;
  const size_t availableBytes = sizeof(requestBuffer_) - 1u - requestBytes_;
  if (packet->tot_len > availableBytes) {
    tcp_recved(tpcb, packet->tot_len);
    pbuf_free(packet);
    return sendJsonAndClose(tpcb, 413, "Payload Too Large", "{\"detail\":\"request exceeded 383 bytes\"}");
  }

  pbuf_copy_partial(packet, requestBuffer_ + requestBytes_, packet->tot_len, 0);
  requestBytes_ += packet->tot_len;
  requestBuffer_[requestBytes_] = '\0';
  tcp_recved(tpcb, packet->tot_len);
  pbuf_free(packet);

  if (!requestHeadersComplete()) {
    return ERR_OK;
  }
  return handleControlRequest(tpcb);
}

err_t NodeControlServer::onPoll(tcp_pcb* tpcb) {
  if (tpcb == nullptr || tpcb != activeClientPcb_) {
    return ERR_OK;
  }
  ++idlePolls_;
  if (idlePolls_ < kIdlePollAbortThreshold) {
    return ERR_OK;
  }
  closeActiveClient(true);
  return ERR_ABRT;
}

void NodeControlServer::onError(err_t err) {
  (void)err;
  activeClientPcb_ = nullptr;
  resetActiveClient();
}

void NodeControlServer::resetActiveClient() {
  requestBytes_ = 0;
  idlePolls_ = 0;
  requestBuffer_[0] = '\0';
}

err_t NodeControlServer::closeActiveClient(bool abortConnection) {
  if (activeClientPcb_ == nullptr) {
    resetActiveClient();
    return ERR_OK;
  }

  tcp_pcb* client = activeClientPcb_;
  activeClientPcb_ = nullptr;
  tcp_arg(client, nullptr);
  tcp_recv(client, nullptr);
  tcp_poll(client, nullptr, 0);
  tcp_err(client, nullptr);

  if (abortConnection) {
    tcp_abort(client);
    resetActiveClient();
    return ERR_ABRT;
  }

  const err_t closeErr = tcp_close(client);
  if (closeErr != ERR_OK) {
    tcp_abort(client);
    resetActiveClient();
    return ERR_ABRT;
  }

  resetActiveClient();
  return ERR_OK;
}

bool NodeControlServer::requestHeadersComplete() const {
  if (requestBytes_ < 4) {
    return false;
  }
  return std::strstr(requestBuffer_, "\r\n\r\n") != nullptr ||
      std::strstr(requestBuffer_, "\n\n") != nullptr;
}

bool NodeControlServer::prepareStateBody(
    char* bodyBuffer,
    size_t bodyBufferBytes,
    bool includeChangedField,
    bool changed) const {
  if (bodyBuffer == nullptr || bodyBufferBytes == 0) {
    return false;
  }

  const int written = std::snprintf(
      bodyBuffer,
      bodyBufferBytes,
      includeChangedField
          ? "{\"control_port\":%u,\"control_path\":\"%s\",\"target_url\":\"%s\",\"target_port\":%u,\"endpoint_valid\":%s,\"publish_in_progress\":%s,\"allow_runtime_port_change\":%s,\"runtime_only\":true,\"changed\":%s}"
          : "{\"control_port\":%u,\"control_path\":\"%s\",\"target_url\":\"%s\",\"target_port\":%u,\"endpoint_valid\":%s,\"publish_in_progress\":%s,\"allow_runtime_port_change\":%s,\"runtime_only\":true}",
      static_cast<unsigned>(listenPort_),
      routePath_,
      publisher_.endpointUrl().c_str(),
      static_cast<unsigned>(publisher_.endpointPort()),
      boolJson(publisher_.endpointValid()),
      boolJson(publisher_.publishInProgress()),
      boolJson(allowRuntimePortChange_),
      boolJson(changed));
  return written > 0 && static_cast<size_t>(written) < bodyBufferBytes;
}

bool NodeControlServer::prepareStatsBody(char* bodyBuffer, size_t bodyBufferBytes) const {
  if (bodyBuffer == nullptr || bodyBufferBytes == 0 || runnerStats_ == nullptr) {
    return false;
  }
  const RunnerStats& stats = *runnerStats_;
  int written = std::snprintf(
      bodyBuffer,
      bodyBufferBytes,
      "{\"frames_captured\":%llu,\"frames_published\":%llu,\"frames_dropped\":%llu,"
      "\"publish_errors\":%llu,\"packet_continuity_violations\":%llu,"
      "\"queue_overflows\":%llu,\"queue_depth\":%u,\"queue_slots_high_water\":%u,"
      "\"queue_slots_capacity\":%u,\"ring_frames_high_water\":%u,\"ring_frames_capacity\":%u,"
      "\"last_publish_status\":%d,\"last_publish_failure_stage\":%u,"
      "\"last_publish_lwip_error\":%ld,\"consecutive_publish_failures\":%u,"
      "\"publish_latency_last_ms\":%u,\"publish_latency_ewma_ms\":%u,"
      "\"publish_latency_max_ms\":%u,\"wifi_rssi_dbm\":%d,\"heap_free_bytes\":%lu,"
      "\"boot_id\":%lu}",
      static_cast<unsigned long long>(stats.framesCaptured),
      static_cast<unsigned long long>(stats.framesPublished),
      static_cast<unsigned long long>(stats.framesDropped),
      static_cast<unsigned long long>(stats.publishErrors),
      static_cast<unsigned long long>(stats.packetContinuityViolations),
      static_cast<unsigned long long>(stats.queueOverflows),
      static_cast<unsigned>(stats.queueDepth),
      static_cast<unsigned>(stats.queueSlotsHighWater),
      static_cast<unsigned>(stats.queueSlotsCapacity),
      static_cast<unsigned>(stats.ringFramesHighWater),
      static_cast<unsigned>(stats.ringFramesCapacity),
      stats.lastPublishStatus,
      static_cast<unsigned>(stats.lastPublishFailureStage),
      static_cast<long>(stats.lastPublishLwipError),
      static_cast<unsigned>(stats.consecutivePublishFailures),
      static_cast<unsigned>(stats.publishLatencyLastMs),
      static_cast<unsigned>(stats.publishLatencyEwmaMs),
      static_cast<unsigned>(stats.publishLatencyMaxMs),
      static_cast<int>(stats.wifiRssiDbm),
      static_cast<unsigned long>(stats.heapFreeBytes),
      static_cast<unsigned long>(stats.bootId));
  if (written <= 0 || static_cast<size_t>(written) >= bodyBufferBytes) {
    return false;
  }
  size_t usedBytes = static_cast<size_t>(written);
  if (bleScannerStats_ != nullptr) {
    const BleScannerStats& ble = *bleScannerStats_;
    written = std::snprintf(
        bodyBuffer + usedBytes - 1u,
        bodyBufferBytes - usedBytes + 1u,
        ",\"ble_scan\":{\"initialized\":%s,\"scanning\":%s,"
        "\"advertisements_observed\":%llu,\"advertisements_dropped\":%llu,"
        "\"table_evictions\":%llu,\"active_advertisers\":%lu}}",
        boolJson(ble.initialized),
        boolJson(ble.scanning),
        static_cast<unsigned long long>(ble.advertisementsObserved),
        static_cast<unsigned long long>(ble.advertisementsDropped),
        static_cast<unsigned long long>(ble.tableEvictions),
        static_cast<unsigned long>(ble.activeAdvertisers));
    if (written <= 0 || static_cast<size_t>(written) >= bodyBufferBytes - usedBytes + 1u) {
      return false;
    }
    usedBytes += static_cast<size_t>(written) - 1u;
  }
  if (bleReportStats_ != nullptr) {
    const BleReportPublisherStats& ble = *bleReportStats_;
    written = std::snprintf(
        bodyBuffer + usedBytes - 1u,
        bodyBufferBytes - usedBytes + 1u,
        ",\"ble_reports\":{\"reports_sent\":%llu,\"reports_dropped\":%llu,"
        "\"report_publish_errors\":%llu,\"last_report_observation_count\":%lu,"
        "\"last_report_status\":%lu}}",
        static_cast<unsigned long long>(ble.reportsSent),
        static_cast<unsigned long long>(ble.reportsDropped),
        static_cast<unsigned long long>(ble.reportPublishErrors),
        static_cast<unsigned long>(ble.lastReportObservationCount),
        static_cast<unsigned long>(ble.lastReportStatus));
    if (written <= 0 || static_cast<size_t>(written) >= bodyBufferBytes - usedBytes + 1u) {
      return false;
    }
    usedBytes += static_cast<size_t>(written) - 1u;
  }
  if (gpsStats_ != nullptr) {
    const GpsRuntimeStats& gps = *gpsStats_;
    written = std::snprintf(
        bodyBuffer + usedBytes - 1u,
        bodyBufferBytes - usedBytes + 1u,
        ",\"gps\":{\"uart_started\":%s,\"nmea_healthy\":%s,\"has_fix\":%s,"
        "\"fix_status\":\"%s\","
        "\"has_datetime\":%s,\"pps_configured\":%s,\"pps_observed\":%s,"
        "\"pps_epoch_aligned\":%s,\"pps_status\":\"%s\",\"clock_quality\":\"%s\","
        "\"fix_dimension\":%u,\"current_baud_rate\":%lu,"
        "\"sentence_age_ms\":%lu,\"fix_age_ms\":%lu,\"pps_age_ms\":%lu,"
        "\"uart_bytes_received\":%llu,\"valid_sentences\":%llu,"
        "\"invalid_checksum_sentences\":%llu,\"unsupported_sentences\":%llu}}",
        boolJson(gps.uartStarted),
        boolJson(gps.nmeaHealthy),
        boolJson(gps.hasFix),
        gpsFixStatusJson(gps),
        boolJson(gps.hasDateTime),
        boolJson(gps.ppsConfigured),
        boolJson(gps.ppsObserved),
        boolJson(gps.ppsEpochAligned),
        gpsPpsStatusJson(gps),
        timeQualityJson(gps.clockQuality),
        static_cast<unsigned>(gps.fixDimension),
        static_cast<unsigned long>(gps.currentBaudRate),
        static_cast<unsigned long>(gps.sentenceAgeMs),
        static_cast<unsigned long>(gps.fixAgeMs),
        static_cast<unsigned long>(gps.ppsAgeMs),
        static_cast<unsigned long long>(gps.uartBytesReceived),
        static_cast<unsigned long long>(gps.validSentences),
        static_cast<unsigned long long>(gps.invalidChecksumSentences),
        static_cast<unsigned long long>(gps.unsupportedSentences));
    if (written <= 0 || static_cast<size_t>(written) >= bodyBufferBytes - usedBytes + 1u) {
      return false;
    }
  }
  return true;
}

bool NodeControlServer::tryParseRequest(char* method, size_t methodBytes, char* target, size_t targetBytes) const {
  if (method == nullptr || target == nullptr || methodBytes == 0 || targetBytes == 0) {
    return false;
  }

  const char* lineEnd = std::strstr(requestBuffer_, "\r\n");
  if (lineEnd == nullptr) {
    lineEnd = std::strchr(requestBuffer_, '\n');
  }
  if (lineEnd == nullptr) {
    return false;
  }

  const char* firstSpace = std::strchr(requestBuffer_, ' ');
  if (firstSpace == nullptr || firstSpace >= lineEnd) {
    return false;
  }
  const char* secondSpace = std::strchr(firstSpace + 1, ' ');
  if (secondSpace == nullptr || secondSpace >= lineEnd) {
    return false;
  }

  if (!appendCopy(method, methodBytes, requestBuffer_, static_cast<size_t>(firstSpace - requestBuffer_))) {
    return false;
  }
  return appendCopy(target, targetBytes, firstSpace + 1, static_cast<size_t>(secondSpace - (firstSpace + 1)));
}

bool NodeControlServer::tryParsePortQuery(const char* target, uint16_t* outPort) const {
  if (target == nullptr || outPort == nullptr) {
    return false;
  }

  const size_t routeLen = std::strlen(routePath_);
  if (std::strncmp(target, routePath_, routeLen) != 0) {
    return false;
  }
  if (target[routeLen] == '\0') {
    return false;
  }
  if (target[routeLen] != '?') {
    return false;
  }

  const char* query = target + routeLen + 1;
  while (*query != '\0') {
    const char* next = std::strchr(query, '&');
    const size_t pairLen = next != nullptr ? static_cast<size_t>(next - query) : std::strlen(query);
    if (pairLen > 5 && std::strncmp(query, "port=", 5) == 0) {
      char portText[8] = {};
      if (!appendCopy(portText, sizeof(portText), query + 5, pairLen - 5)) {
        return false;
      }
      char* endPtr = nullptr;
      const long parsed = std::strtol(portText, &endPtr, 10);
      if (endPtr == nullptr || *endPtr != '\0' || parsed <= 0 || parsed > 65535) {
        return false;
      }
      *outPort = static_cast<uint16_t>(parsed);
      return true;
    }
    if (next == nullptr) {
      break;
    }
    query = next + 1;
  }

  return false;
}

err_t NodeControlServer::sendJsonAndClose(
    tcp_pcb* tpcb,
    int statusCode,
    const char* reason,
    const char* jsonBody) {
  if (tpcb == nullptr || jsonBody == nullptr || reason == nullptr) {
    closeActiveClient(true);
    return ERR_ABRT;
  }

  char response[kMaxResponseBytes] = {};
  const size_t bodyBytes = std::strlen(jsonBody);
  const int responseBytes = std::snprintf(
      response,
      sizeof(response),
      "HTTP/1.1 %d %s\r\n"
      "Connection: close\r\n"
      "Cache-Control: no-store\r\n"
      "Content-Type: application/json\r\n"
      "Content-Length: %lu\r\n\r\n"
      "%s",
      statusCode,
      reason,
      static_cast<unsigned long>(bodyBytes),
      jsonBody);
  if (responseBytes <= 0 || static_cast<size_t>(responseBytes) >= sizeof(response)) {
    closeActiveClient(true);
    return ERR_ABRT;
  }

  const err_t writeErr = tcp_write(
      tpcb,
      response,
      static_cast<uint16_t>(responseBytes),
      TCP_WRITE_FLAG_COPY);
  if (writeErr != ERR_OK) {
    closeActiveClient(true);
    return ERR_ABRT;
  }
  tcp_output(tpcb);
  return closeActiveClient(false);
}

err_t NodeControlServer::handleControlRequest(tcp_pcb* tpcb) {
  char method[kMaxMethodBytes] = {};
  char target[kMaxTargetBytes] = {};
  char body[kMaxBodyBytes] = {};
  if (!tryParseRequest(method, sizeof(method), target, sizeof(target))) {
    return sendJsonAndClose(tpcb, 400, "Bad Request", "{\"detail\":\"unable to parse HTTP request line\"}");
  }

  const size_t statsRouteLen = std::strlen(statsPath_);
  if (std::strcmp(method, "GET") == 0 &&
      std::strncmp(target, statsPath_, statsRouteLen) == 0 &&
      (target[statsRouteLen] == '\0' || target[statsRouteLen] == '?')) {
    if (!prepareStatsBody(body, sizeof(body))) {
      return sendJsonAndClose(tpcb, 500, "Internal Server Error", "{\"detail\":\"stats unavailable\"}");
    }
    return sendJsonAndClose(tpcb, 200, "OK", body);
  }

  const size_t routeLen = std::strlen(routePath_);
  if (std::strncmp(target, routePath_, routeLen) != 0 ||
      (target[routeLen] != '\0' && target[routeLen] != '?')) {
    return sendJsonAndClose(tpcb, 404, "Not Found", "{\"detail\":\"unknown control route\"}");
  }

  if (std::strcmp(method, "GET") == 0) {
    if (!prepareStateBody(body, sizeof(body), false, false)) {
      return sendJsonAndClose(tpcb, 500, "Internal Server Error", "{\"detail\":\"unable to build state response\"}");
    }
    return sendJsonAndClose(tpcb, 200, "OK", body);
  }

  if (std::strcmp(method, "POST") != 0 && std::strcmp(method, "PUT") != 0) {
    return sendJsonAndClose(tpcb, 405, "Method Not Allowed", "{\"detail\":\"use GET or POST\"}");
  }

  if (!allowRuntimePortChange_) {
    return sendJsonAndClose(tpcb, 403, "Forbidden", "{\"detail\":\"runtime port changes are disabled\"}");
  }
  if (publisher_.publishInProgress()) {
    return sendJsonAndClose(tpcb, 409, "Conflict", "{\"detail\":\"publish in progress; retry after current POST finishes\"}");
  }

  uint16_t requestedPort = 0;
  if (!tryParsePortQuery(target, &requestedPort)) {
    return sendJsonAndClose(tpcb, 400, "Bad Request", "{\"detail\":\"expected ?port=<1-65535>\"}");
  }

  const bool changed = publisher_.endpointPort() != requestedPort;
  if (!publisher_.setEndpointPort(requestedPort)) {
    return sendJsonAndClose(tpcb, 500, "Internal Server Error", "{\"detail\":\"unable to retarget publish port\"}");
  }

  std::printf(
      "[control] retargeted publish port=%u url=%s\n",
      static_cast<unsigned>(publisher_.endpointPort()),
      publisher_.endpointUrl().c_str());
  if (!prepareStateBody(body, sizeof(body), true, changed)) {
    return sendJsonAndClose(tpcb, 500, "Internal Server Error", "{\"detail\":\"unable to build state response\"}");
  }
  return sendJsonAndClose(tpcb, 200, "OK", body);
}

}  // namespace mmpr
