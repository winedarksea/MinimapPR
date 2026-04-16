#include "mmpr/HttpFramePublisher.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

#include "lwip/dns.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/tcp.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"

#include "mmpr/NodeProtocol.h"

namespace mmpr {
namespace {

bool isWiFiConnected() {
  return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA) == CYW43_LINK_UP;
}

bool equalsIgnoreAsciiCase(char lhs, char rhs) {
  return std::tolower(static_cast<unsigned char>(lhs)) ==
         std::tolower(static_cast<unsigned char>(rhs));
}

bool startsWithIgnoreAsciiCase(const std::string& text, const char* prefix) {
  if (prefix == nullptr) {
    return false;
  }
  size_t index = 0;
  while (prefix[index] != '\0') {
    if (index >= text.size() || !equalsIgnoreAsciiCase(text[index], prefix[index])) {
      return false;
    }
    ++index;
  }
  return true;
}

bool headerHasToken(const std::string& text, const char* token) {
  if (token == nullptr || *token == '\0') {
    return false;
  }

  const size_t tokenLen = std::strlen(token);
  for (size_t index = 0; index + tokenLen <= text.size(); ++index) {
    size_t matched = 0;
    while (matched < tokenLen && equalsIgnoreAsciiCase(text[index + matched], token[matched])) {
      ++matched;
    }
    if (matched == tokenLen) {
      return true;
    }
  }
  return false;
}

void appendResponseChunk(std::string& response, bool keepResponse, size_t responseCap, const char* data, size_t len) {
  if (data == nullptr || len == 0) {
    return;
  }

  if (keepResponse) {
    response.append(data, len);
    return;
  }

  if (response.size() >= responseCap) {
    return;
  }

  size_t appendLen = len;
  const size_t remaining = responseCap - response.size();
  if (appendLen > remaining) {
    appendLen = remaining;
  }
  response.append(data, appendLen);
}

std::string buildRequest(
    const std::string& host,
    uint16_t port,
    const std::string& path,
    const std::string& payload) {
  std::string request;
  request.reserve(payload.size() + 256);

  request += "POST ";
  request += path;
  request += " HTTP/1.1\r\nHost: ";
  request += host;
  if (port != 80) {
    request += ':';
    request += std::to_string(port);
  }
  request += "\r\nConnection: keep-alive\r\nContent-Type: application/json\r\nContent-Length: ";
  request += std::to_string(payload.size());
  request += "\r\n\r\n";
  request += payload;
  return request;
}

}  // namespace

struct HttpFramePublisher::TransportState {
  tcp_pcb* pcb = nullptr;
  ip_addr_t remoteAddr = {};
  bool remoteAddrValid = false;

  bool dnsDone = false;
  bool dnsOk = false;
  bool connectDone = false;
  bool connected = false;
  bool requestDone = false;

  err_t err = ERR_OK;

  std::string request;
  size_t requestOffset = 0;

  std::string response;
  std::string headerBuffer;
  bool keepResponse = false;
  size_t responseCap = 2048;
  int statusCode = -4;
  bool headersParsed = false;
  int64_t contentLength = -1;
  size_t bodyBytesReceived = 0;
  bool responseMustClose = false;
  bool sawResponseClose = false;
};

namespace {

void closeConnection(HttpFramePublisher::TransportState& state) {
  if (state.pcb == nullptr) {
    state.connected = false;
    state.connectDone = true;
    return;
  }

  tcp_arg(state.pcb, nullptr);
  tcp_recv(state.pcb, nullptr);
  tcp_sent(state.pcb, nullptr);
  tcp_poll(state.pcb, nullptr, 0);
  tcp_err(state.pcb, nullptr);

  const err_t closeErr = tcp_close(state.pcb);
  if (closeErr != ERR_OK) {
    tcp_abort(state.pcb);
  }

  state.pcb = nullptr;
  state.connected = false;
  state.connectDone = true;
}

void resetRequestState(HttpFramePublisher::TransportState& state, bool keepResponseBody) {
  state.request.clear();
  state.requestOffset = 0;
  state.response.clear();
  state.headerBuffer.clear();
  state.keepResponse = keepResponseBody;
  state.responseCap = keepResponseBody ? 0 : 2048;
  state.statusCode = -4;
  state.headersParsed = false;
  state.contentLength = -1;
  state.bodyBytesReceived = 0;
  state.responseMustClose = false;
  state.sawResponseClose = false;
  state.requestDone = false;
  state.err = ERR_OK;
}

bool resolveHost(const std::string& host, uint32_t timeoutMs, HttpFramePublisher::TransportState& state) {
  if (state.remoteAddrValid) {
    return true;
  }
  if (ipaddr_aton(host.c_str(), &state.remoteAddr)) {
    state.remoteAddrValid = true;
    return true;
  }

  state.dnsDone = false;
  state.dnsOk = false;
  const err_t dnsErr = dns_gethostbyname(
      host.c_str(),
      &state.remoteAddr,
      [](const char* name, const ip_addr_t* ipaddr, void* arg) {
        (void)name;
        HttpFramePublisher::TransportState* transport =
            static_cast<HttpFramePublisher::TransportState*>(arg);
        if (transport == nullptr) {
          return;
        }
        transport->dnsDone = true;
        if (ipaddr != nullptr) {
          transport->remoteAddr = *ipaddr;
          transport->dnsOk = true;
        }
      },
      &state);
  if (dnsErr == ERR_OK) {
    state.remoteAddrValid = true;
    return true;
  }
  if (dnsErr != ERR_INPROGRESS) {
    return false;
  }

  const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
  while (!state.dnsDone && !time_reached(deadline)) {
    cyw43_arch_poll();
    sleep_ms(1);
  }

  state.remoteAddrValid = state.dnsDone && state.dnsOk;
  return state.remoteAddrValid;
}

void flushTx(HttpFramePublisher::TransportState& state, tcp_pcb* tpcb) {
  if (tpcb == nullptr) {
    return;
  }

  while (state.requestOffset < state.request.size()) {
    const uint16_t sndbuf = tcp_sndbuf(tpcb);
    if (sndbuf == 0) {
      break;
    }

    size_t remaining = state.request.size() - state.requestOffset;
    size_t chunk = remaining;
    if (chunk > sndbuf) {
      chunk = sndbuf;
    }
    if (chunk > 2048) {
      chunk = 2048;
    }

    const err_t writeErr = tcp_write(
        tpcb,
        state.request.data() + state.requestOffset,
        static_cast<u16_t>(chunk),
        TCP_WRITE_FLAG_COPY);
    if (writeErr == ERR_OK) {
      state.requestOffset += chunk;
    } else if (writeErr == ERR_MEM) {
      break;
    } else {
      state.err = writeErr;
      state.requestDone = true;
      closeConnection(state);
      return;
    }
  }

  if (state.requestOffset > 0) {
    (void)tcp_output(tpcb);
  }
}

void parseStatusLine(HttpFramePublisher::TransportState& state) {
  if (state.statusCode > 0) {
    return;
  }

  const size_t lineEnd = state.headerBuffer.find("\r\n");
  if (lineEnd == std::string::npos) {
    return;
  }

  const std::string line = state.headerBuffer.substr(0, lineEnd);
  if (line.rfind("HTTP/", 0) != 0) {
    return;
  }

  const size_t firstSpace = line.find(' ');
  if (firstSpace == std::string::npos) {
    return;
  }

  const size_t secondSpace = line.find(' ', firstSpace + 1);
  const std::string codeText =
      (secondSpace == std::string::npos)
          ? line.substr(firstSpace + 1)
          : line.substr(firstSpace + 1, secondSpace - (firstSpace + 1));

  const long parsed = std::strtol(codeText.c_str(), nullptr, 10);
  if (parsed > 0 && parsed <= 999) {
    state.statusCode = static_cast<int>(parsed);
  }
}

void tryFinalizeResponse(HttpFramePublisher::TransportState& state) {
  if (state.requestDone || !state.headersParsed) {
    return;
  }

  const bool bodyComplete =
      (state.contentLength >= 0) &&
      (state.bodyBytesReceived >= static_cast<size_t>(state.contentLength));
  const bool emptyBodyStatus =
      (state.statusCode >= 100 && state.statusCode < 200) ||
      state.statusCode == 204 ||
      state.statusCode == 304;

  if (bodyComplete || (state.contentLength == 0) || (state.contentLength < 0 && emptyBodyStatus)) {
    state.requestDone = true;
    if (state.responseMustClose) {
      closeConnection(state);
    }
  }
}

void parseHeadersIfReady(HttpFramePublisher::TransportState& state) {
  if (state.headersParsed) {
    return;
  }

  const size_t headerEnd = state.headerBuffer.find("\r\n\r\n");
  if (headerEnd == std::string::npos) {
    return;
  }

  parseStatusLine(state);

  const std::string headers = state.headerBuffer.substr(0, headerEnd + 2);
  size_t lineStart = headers.find("\r\n");
  while (lineStart != std::string::npos && lineStart + 2 < headers.size()) {
    lineStart += 2;
    const size_t lineEnd = headers.find("\r\n", lineStart);
    if (lineEnd == std::string::npos || lineEnd == lineStart) {
      break;
    }

    const std::string line = headers.substr(lineStart, lineEnd - lineStart);
    if (startsWithIgnoreAsciiCase(line, "Content-Length:")) {
      const std::string value = line.substr(std::strlen("Content-Length:"));
      state.contentLength = static_cast<int64_t>(std::strtol(value.c_str(), nullptr, 10));
      if (state.contentLength < 0) {
        state.contentLength = -1;
      }
    } else if (startsWithIgnoreAsciiCase(line, "Connection:")) {
      const std::string value = line.substr(std::strlen("Connection:"));
      if (headerHasToken(value, "close")) {
        state.responseMustClose = true;
      }
    }

    lineStart = lineEnd;
  }

  state.headersParsed = true;
  const size_t bodyOffset = headerEnd + 4;
  state.bodyBytesReceived = state.headerBuffer.size() - bodyOffset;
  state.headerBuffer.clear();
  tryFinalizeResponse(state);
}

err_t onConnected(void* arg, tcp_pcb* tpcb, err_t err) {
  HttpFramePublisher::TransportState* state =
      static_cast<HttpFramePublisher::TransportState*>(arg);
  if (state == nullptr) {
    return ERR_ARG;
  }
  state->connectDone = true;
  if (err != ERR_OK) {
    state->err = err;
    state->requestDone = true;
    closeConnection(*state);
    return err;
  }

  state->connected = true;
  flushTx(*state, tpcb);
  return ERR_OK;
}

err_t onSent(void* arg, tcp_pcb* tpcb, uint16_t len) {
  (void)len;
  HttpFramePublisher::TransportState* state =
      static_cast<HttpFramePublisher::TransportState*>(arg);
  if (state == nullptr) {
    return ERR_ARG;
  }
  flushTx(*state, tpcb);
  return ERR_OK;
}

err_t onPoll(void* arg, tcp_pcb* tpcb) {
  HttpFramePublisher::TransportState* state =
      static_cast<HttpFramePublisher::TransportState*>(arg);
  if (state == nullptr) {
    return ERR_OK;
  }
  if (!state->requestDone) {
    flushTx(*state, tpcb);
  }
  return ERR_OK;
}

err_t onRecv(void* arg, tcp_pcb* tpcb, pbuf* p, err_t err) {
  HttpFramePublisher::TransportState* state =
      static_cast<HttpFramePublisher::TransportState*>(arg);
  if (state == nullptr) {
    if (p != nullptr) {
      pbuf_free(p);
    }
    return ERR_ARG;
  }

  if (err != ERR_OK) {
    state->err = err;
    state->requestDone = true;
    if (p != nullptr) {
      pbuf_free(p);
    }
    closeConnection(*state);
    return err;
  }

  if (p == nullptr) {
    state->sawResponseClose = true;
    parseHeadersIfReady(*state);
    state->requestDone = true;
    closeConnection(*state);
    return ERR_OK;
  }

  for (pbuf* q = p; q != nullptr; q = q->next) {
    const char* chunk = static_cast<const char*>(q->payload);
    appendResponseChunk(state->response, state->keepResponse, state->responseCap, chunk, q->len);
    if (!state->headersParsed) {
      state->headerBuffer.append(chunk, q->len);
      parseHeadersIfReady(*state);
    } else {
      state->bodyBytesReceived += q->len;
      tryFinalizeResponse(*state);
    }
  }

  tcp_recved(tpcb, p->tot_len);
  pbuf_free(p);
  return ERR_OK;
}

void onErr(void* arg, err_t err) {
  HttpFramePublisher::TransportState* state =
      static_cast<HttpFramePublisher::TransportState*>(arg);
  if (state == nullptr) {
    return;
  }
  state->pcb = nullptr;
  state->connected = false;
  state->connectDone = true;
  state->err = err;
  state->requestDone = true;
}

bool ensureConnected(
    const std::string& host,
    uint16_t port,
    uint32_t timeoutMs,
    HttpFramePublisher::TransportState& state) {
  if (state.connected && state.pcb != nullptr) {
    return true;
  }

  closeConnection(state);
  if (!resolveHost(host, timeoutMs, state)) {
    return false;
  }

  state.connectDone = false;
  state.err = ERR_OK;
  state.pcb = tcp_new_ip_type(IP_GET_TYPE(&state.remoteAddr));
  if (state.pcb == nullptr) {
    return false;
  }

  tcp_arg(state.pcb, &state);
  tcp_recv(state.pcb, &onRecv);
  tcp_sent(state.pcb, &onSent);
  tcp_poll(state.pcb, &onPoll, 2);
  tcp_err(state.pcb, &onErr);

  const err_t connectErr = tcp_connect(state.pcb, &state.remoteAddr, port, &onConnected);
  if (connectErr != ERR_OK) {
    closeConnection(state);
    return false;
  }

  const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
  while (!state.connectDone && !time_reached(deadline)) {
    cyw43_arch_poll();
    sleep_ms(1);
  }

  if (!state.connected) {
    state.err = state.err == ERR_OK ? ERR_TIMEOUT : state.err;
    closeConnection(state);
    return false;
  }

  return true;
}

PublishResult post(
    const std::string& host,
    uint16_t port,
    const std::string& path,
    const std::string& payload,
    uint32_t timeoutMs,
    bool keepResponseBody,
    HttpFramePublisher::TransportState& state) {
  PublishResult result = {};
  result.ok = false;
  result.statusCode = -3;

  if (host.empty() || path.empty()) {
    return result;
  }

  if (!ensureConnected(host, port, timeoutMs, state)) {
    result.statusCode = (state.err == ERR_TIMEOUT) ? -4 : -3;
    return result;
  }

  resetRequestState(state, keepResponseBody);
  state.request = buildRequest(host, port, path, payload);
  flushTx(state, state.pcb);

  const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
  while (!state.requestDone && !time_reached(deadline)) {
    cyw43_arch_poll();
    sleep_ms(1);
  }

  if (!state.requestDone) {
    state.err = ERR_TIMEOUT;
    closeConnection(state);
    result.statusCode = -4;
  } else if (state.statusCode > 0) {
    result.statusCode = state.statusCode;
  } else {
    result.statusCode = (state.err == ERR_OK || state.err == ERR_CLSD) ? -4 : -3;
  }

  if (keepResponseBody || result.statusCode < 200 || result.statusCode >= 300) {
    result.responseBody = state.response;
  }
  result.ok = (result.statusCode >= 200 && result.statusCode < 300);
  return result;
}

}  // namespace

HttpFramePublisher::HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs)
    : timeoutMs_(timeoutMs),
      transportState_(new TransportState()) {
  endpointUrl_ = (serverBaseUrl != nullptr ? std::string(serverBaseUrl) : std::string()) +
                 (ingestPath != nullptr ? std::string(ingestPath) : std::string("/api/v1/ingest/frame"));
  endpointValid_ = parseEndpoint();
}

HttpFramePublisher::~HttpFramePublisher() {
  if (transportState_ != nullptr) {
    closeConnection(*transportState_);
    delete transportState_;
    transportState_ = nullptr;
  }
}

PublishResult HttpFramePublisher::publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody) {
  return publish(node, frame, nullptr, keepResponseBody);
}

PublishResult HttpFramePublisher::publish(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    bool keepResponseBody) {
  PublishResult result = {};
  result.ok = false;
  result.statusCode = -1;

  if (!endpointValid_) {
    result.statusCode = -3;
    return result;
  }
  if (!isWiFiConnected()) {
    result.statusCode = -1;
    return result;
  }

  std::string payload;
  if (!buildIngestPayload(node, frame, environment, payload)) {
    result.statusCode = -2;
    return result;
  }

  result = post(host_, port_, path_, payload, timeoutMs_, keepResponseBody, *transportState_);
  if (!result.ok && result.statusCode < 0 && transportState_ != nullptr) {
    closeConnection(*transportState_);
    result = post(host_, port_, path_, payload, timeoutMs_, keepResponseBody, *transportState_);
  }
  return result;
}

void HttpFramePublisher::trimAsciiWhitespace(std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
    ++start;
  }

  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
    --end;
  }

  if (start == 0 && end == s.size()) {
    return;
  }
  s = s.substr(start, end - start);
}

bool HttpFramePublisher::parseEndpoint() {
  host_.clear();
  path_.clear();
  port_ = 80;

  std::string url = endpointUrl_;
  trimAsciiWhitespace(url);

  constexpr const char* kHttpPrefix = "http://";
  if (url.rfind(kHttpPrefix, 0) != 0) {
    return false;
  }
  url.erase(0, std::strlen(kHttpPrefix));

  const size_t pathStart = url.find('/');
  const std::string hostPort = (pathStart == std::string::npos) ? url : url.substr(0, pathStart);
  path_ = (pathStart == std::string::npos) ? "/" : url.substr(pathStart);
  if (hostPort.empty()) {
    return false;
  }

  if (hostPort.front() == '[') {
    const size_t closeBracket = hostPort.find(']');
    if (closeBracket == std::string::npos) {
      return false;
    }
    host_ = hostPort.substr(1, closeBracket - 1);
    if (closeBracket + 1 < hostPort.size()) {
      if (hostPort[closeBracket + 1] != ':') {
        return false;
      }
      const std::string portText = hostPort.substr(closeBracket + 2);
      const long parsedPort = std::strtol(portText.c_str(), nullptr, 10);
      if (parsedPort <= 0 || parsedPort > 65535) {
        return false;
      }
      port_ = static_cast<uint16_t>(parsedPort);
    }
  } else {
    const size_t colonIndex = hostPort.rfind(':');
    if (colonIndex != std::string::npos) {
      host_ = hostPort.substr(0, colonIndex);
      const std::string portText = hostPort.substr(colonIndex + 1);
      const long parsedPort = std::strtol(portText.c_str(), nullptr, 10);
      if (parsedPort <= 0 || parsedPort > 65535) {
        return false;
      }
      port_ = static_cast<uint16_t>(parsedPort);
    } else {
      host_ = hostPort;
    }
  }

  if (host_.empty()) {
    return false;
  }
  if (path_.empty() || path_[0] != '/') {
    path_ = "/" + path_;
  }
  return true;
}

}  // namespace mmpr
