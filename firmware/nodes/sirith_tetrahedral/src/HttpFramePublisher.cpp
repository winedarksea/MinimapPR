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

class RawHttpPostClient {
 public:
  static PublishResult post(
      const std::string& host,
      uint16_t port,
      const std::string& path,
      const std::string& payload,
      uint32_t timeoutMs,
      bool keepResponseBody) {
    PublishResult result = {};
    result.ok = false;
    result.statusCode = -3;

    if (host.empty() || path.empty()) {
      return result;
    }

    State state = {};
    state.keepResponse = keepResponseBody;
    state.responseCap = keepResponseBody ? 0 : 2048;
    state.request = buildRequest(host, port, path, payload);

    if (!resolveHost(host, timeoutMs, state)) {
      result.statusCode = -3;
      return result;
    }

    state.pcb = tcp_new_ip_type(IP_GET_TYPE(&state.remoteAddr));
    if (state.pcb == nullptr) {
      result.statusCode = -3;
      return result;
    }

    tcp_arg(state.pcb, &state);
    tcp_recv(state.pcb, &RawHttpPostClient::onRecv);
    tcp_sent(state.pcb, &RawHttpPostClient::onSent);
    tcp_poll(state.pcb, &RawHttpPostClient::onPoll, 2);
    tcp_err(state.pcb, &RawHttpPostClient::onErr);

    const err_t connectErr = tcp_connect(state.pcb, &state.remoteAddr, port, &RawHttpPostClient::onConnected);
    if (connectErr != ERR_OK) {
      closeConnection(state);
      result.statusCode = -3;
      return result;
    }

    const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
    while (!state.done && !time_reached(deadline)) {
      cyw43_arch_poll();
      sleep_ms(1);
    }

    if (!state.done) {
      state.err = ERR_TIMEOUT;
      closeConnection(state);
      result.statusCode = -4;
    } else {
      parseStatusLine(state);
      if (state.statusCode > 0) {
        result.statusCode = state.statusCode;
      } else {
        result.statusCode = (state.err == ERR_OK) ? -4 : -3;
      }
    }

    if (keepResponseBody || result.statusCode < 200 || result.statusCode >= 300) {
      result.responseBody = state.response;
    }

    result.ok = (result.statusCode >= 200 && result.statusCode < 300);
    return result;
  }

 private:
  struct State {
    tcp_pcb* pcb = nullptr;
    ip_addr_t remoteAddr = {};

    bool dnsDone = false;
    bool dnsOk = false;

    bool done = false;
    err_t err = ERR_OK;

    std::string request;
    size_t requestOffset = 0;

    std::string response;
    bool keepResponse = false;
    size_t responseCap = 2048;
    int statusCode = -4;
  };

  static std::string buildRequest(
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
    request += "\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: ";
    request += std::to_string(payload.size());
    request += "\r\n\r\n";
    request += payload;
    return request;
  }

  static void onDnsFound(const char* name, const ip_addr_t* ipaddr, void* arg) {
    (void)name;
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return;
    }
    state->dnsDone = true;
    if (ipaddr != nullptr) {
      state->remoteAddr = *ipaddr;
      state->dnsOk = true;
    } else {
      state->dnsOk = false;
    }
  }

  static bool resolveHost(const std::string& host, uint32_t timeoutMs, State& state) {
    if (ipaddr_aton(host.c_str(), &state.remoteAddr)) {
      return true;
    }

    const err_t dnsErr = dns_gethostbyname(host.c_str(), &state.remoteAddr, &RawHttpPostClient::onDnsFound, &state);
    if (dnsErr == ERR_OK) {
      state.dnsDone = true;
      state.dnsOk = true;
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

    return state.dnsDone && state.dnsOk;
  }

  static void appendResponseChunk(State& state, const char* data, size_t len) {
    if (data == nullptr || len == 0) {
      return;
    }

    if (state.keepResponse) {
      state.response.append(data, len);
      return;
    }

    if (state.response.size() >= state.responseCap) {
      return;
    }
    size_t appendLen = len;
    const size_t remaining = state.responseCap - state.response.size();
    if (appendLen > remaining) {
      appendLen = remaining;
    }
    state.response.append(data, appendLen);
  }

  static void parseStatusLine(State& state) {
    if (state.statusCode > 0) {
      return;
    }

    const size_t lineEnd = state.response.find("\r\n");
    if (lineEnd == std::string::npos) {
      return;
    }

    const std::string line = state.response.substr(0, lineEnd);
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

  static void closeConnection(State& state) {
    if (state.pcb == nullptr) {
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
  }

  static void flushTx(State& state, tcp_pcb* tpcb) {
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
        state.done = true;
        closeConnection(state);
        return;
      }
    }
    (void)tcp_output(tpcb);
  }

  static err_t onConnected(void* arg, tcp_pcb* tpcb, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_ARG;
    }
    if (err != ERR_OK) {
      state->err = err;
      state->done = true;
      closeConnection(*state);
      return err;
    }
    flushTx(*state, tpcb);
    return ERR_OK;
  }

  static err_t onSent(void* arg, tcp_pcb* tpcb, uint16_t len) {
    (void)len;
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_ARG;
    }
    flushTx(*state, tpcb);
    return ERR_OK;
  }

  static err_t onPoll(void* arg, tcp_pcb* tpcb) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_OK;
    }
    if (!state->done) {
      flushTx(*state, tpcb);
    }
    return ERR_OK;
  }

  static err_t onRecv(void* arg, tcp_pcb* tpcb, pbuf* p, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      if (p != nullptr) {
        pbuf_free(p);
      }
      return ERR_ARG;
    }

    if (err != ERR_OK) {
      state->err = err;
      state->done = true;
      if (p != nullptr) {
        pbuf_free(p);
      }
      closeConnection(*state);
      return err;
    }

    if (p == nullptr) {
      parseStatusLine(*state);
      state->done = true;
      closeConnection(*state);
      return ERR_OK;
    }

    for (pbuf* q = p; q != nullptr; q = q->next) {
      appendResponseChunk(*state, static_cast<const char*>(q->payload), q->len);
    }
    parseStatusLine(*state);
    tcp_recved(tpcb, p->tot_len);
    pbuf_free(p);
    return ERR_OK;
  }

  static void onErr(void* arg, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return;
    }
    state->pcb = nullptr;
    state->err = err;
    state->done = true;
  }
};

}  // namespace

HttpFramePublisher::HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs)
    : timeoutMs_(timeoutMs) {
  endpointUrl_ = (serverBaseUrl != nullptr ? std::string(serverBaseUrl) : std::string()) +
                 (ingestPath != nullptr ? std::string(ingestPath) : std::string("/api/v1/ingest/frame"));
  endpointValid_ = parseEndpoint();
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

  return RawHttpPostClient::post(host_, port_, path_, payload, timeoutMs_, keepResponseBody);
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
