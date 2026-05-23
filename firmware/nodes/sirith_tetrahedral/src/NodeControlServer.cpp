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
constexpr size_t kMaxBodyBytes = 320;
constexpr size_t kMaxResponseBytes = 512;
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

}  // namespace

NodeControlServer::NodeControlServer(
    HttpFramePublisher& publisher,
    uint16_t listenPort,
    const char* routePath,
    bool allowRuntimePortChange)
    : publisher_(publisher),
      listenPort_(listenPort),
      routePath_(routePath != nullptr ? routePath : "/api/v1/publish-target"),
      allowRuntimePortChange_(allowRuntimePortChange) {}

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