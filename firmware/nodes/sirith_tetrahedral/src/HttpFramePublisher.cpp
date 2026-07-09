#include "mmpr/HttpFramePublisher.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include "lwip/dns.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/tcp.h"
#include "pico/cyw43_arch.h"
#include "pico/time.h"

#include "mmpr/Base64.h"
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

std::string buildRequestHeader(
    const std::string& host,
    uint16_t port,
    const std::string& path,
    size_t payloadSize,
    const char* contentType) {
  std::string header;
  header.reserve(256);

  header += "POST ";
  header += path;
  header += " HTTP/1.1\r\nHost: ";
  header += host;
  if (port != 80) {
    header += ':';
    header += std::to_string(port);
  }
  header += "\r\nConnection: keep-alive\r\nContent-Type: ";
  header += contentType != nullptr ? contentType : "application/octet-stream";
  header += "\r\nContent-Length: ";
  header += std::to_string(payloadSize);
  header += "\r\n\r\n";
  return header;
}

}  // namespace

struct HttpFramePublisher::TransportState {
  enum class SendPhase {
    kHeader,
    kSegmentPrefix,
    kAudio,
    kSegmentSuffix,
    kDone,
  };

  tcp_pcb* pcb = nullptr;
  ip_addr_t remoteAddr = {};
  bool remoteAddrValid = false;

  bool dnsDone = false;
  bool dnsOk = false;
  bool connectDone = false;
  bool connected = false;
  bool requestDone = false;

  err_t err = ERR_OK;

  std::string requestHeader;
  std::vector<IngestPayloadParts> payloadParts;
  std::vector<const uint8_t*> audioSegments;
  std::string base64Chunk;
  bool binaryAudio = false;
  SendPhase sendPhase = SendPhase::kHeader;
  size_t segmentIndex = 0;
  size_t sendOffset = 0;
  size_t audioOffset = 0;

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
  bool asyncPublishActive = false;
  absolute_time_t requestDeadline = {};
  uint32_t requestTimeoutMs = 0;
  uint32_t requestStartedMs = 0;
  HttpFramePublisher::BackgroundPollCallback backgroundPollCallback = nullptr;
  void* backgroundPollContext = nullptr;
  PublishFailureStage failureStage = PublishFailureStage::kNone;
};

namespace {

void runBackgroundPoll(HttpFramePublisher::TransportState& state) {
  if (state.backgroundPollCallback != nullptr) {
    state.backgroundPollCallback(state.backgroundPollContext);
  }
}

void refreshRequestDeadline(HttpFramePublisher::TransportState& state) {
  if (state.requestTimeoutMs > 0) {
    state.requestDeadline = make_timeout_time_ms(state.requestTimeoutMs);
  }
}

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

void abortConnection(HttpFramePublisher::TransportState& state) {
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
  tcp_abort(state.pcb);

  state.pcb = nullptr;
  state.connected = false;
  state.connectDone = true;
}

void resetRequestState(HttpFramePublisher::TransportState& state, bool keepResponseBody) {
  state.requestHeader.clear();
  state.payloadParts.clear();
  state.audioSegments.clear();
  state.base64Chunk.clear();
  state.binaryAudio = false;
  state.sendPhase = HttpFramePublisher::TransportState::SendPhase::kHeader;
  state.segmentIndex = 0;
  state.sendOffset = 0;
  state.audioOffset = 0;
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
  state.asyncPublishActive = false;
  state.requestDeadline = {};
  state.requestTimeoutMs = 0;
  state.requestStartedMs = 0;
  state.requestDone = false;
  state.err = ERR_OK;
  state.failureStage = PublishFailureStage::kNone;
}

bool resolveHostUntil(
    const std::string& host,
    absolute_time_t deadline,
    HttpFramePublisher::TransportState& state) {
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
    state.err = dnsErr;
    state.failureStage = PublishFailureStage::kDns;
    return false;
  }

  while (!state.dnsDone && !time_reached(deadline)) {
    cyw43_arch_poll();
    runBackgroundPoll(state);
    sleep_ms(1);
  }

  state.remoteAddrValid = state.dnsDone && state.dnsOk;
  if (!state.remoteAddrValid) {
    state.err = ERR_TIMEOUT;
    state.failureStage = PublishFailureStage::kDns;
  }
  return state.remoteAddrValid;
}

bool writeStringSegment(
    HttpFramePublisher::TransportState& state,
    tcp_pcb* tpcb,
    const std::string& segment) {
  while (state.sendOffset < segment.size()) {
    const uint16_t sndbuf = tcp_sndbuf(tpcb);
    if (sndbuf == 0) {
      return false;
    }

    size_t chunk = segment.size() - state.sendOffset;
    if (chunk > sndbuf) {
      chunk = sndbuf;
    }
    if (chunk > 8192) {
      chunk = 8192;
    }

    const err_t writeErr = tcp_write(
        tpcb,
        segment.data() + state.sendOffset,
        static_cast<u16_t>(chunk),
        TCP_WRITE_FLAG_COPY);
    if (writeErr == ERR_OK) {
      state.sendOffset += chunk;
      refreshRequestDeadline(state);
    } else if (writeErr == ERR_MEM) {
      return false;
    } else {
      state.err = writeErr;
      state.failureStage = PublishFailureStage::kSend;
      state.requestDone = true;
      closeConnection(state);
      return false;
    }
  }

  state.sendOffset = 0;
  return true;
}

bool refillAudioChunk(HttpFramePublisher::TransportState& state) {
  if (state.segmentIndex >= state.payloadParts.size() ||
      state.segmentIndex >= state.audioSegments.size()) {
    return false;
  }

  const IngestPayloadParts& segment = state.payloadParts[state.segmentIndex];
  const uint8_t* audioData = state.audioSegments[state.segmentIndex];
  if (audioData == nullptr || state.audioOffset >= segment.rawAudioBytes) {
    return false;
  }

  constexpr size_t kRawAudioChunkBytes = 768;
  constexpr size_t kBinaryAudioChunkBytes = 8192;
  if (state.binaryAudio) {
    const size_t remaining = segment.rawAudioBytes - state.audioOffset;
    size_t rawChunkBytes = remaining;
    if (rawChunkBytes > kBinaryAudioChunkBytes) {
      rawChunkBytes = kBinaryAudioChunkBytes;
    }
    state.base64Chunk.assign(
        reinterpret_cast<const char*>(audioData + state.audioOffset),
        rawChunkBytes);
    state.audioOffset += rawChunkBytes;
    state.sendOffset = 0;
    return true;
  }

  const size_t remaining = segment.rawAudioBytes - state.audioOffset;
  size_t rawChunkBytes = remaining;
  if (rawChunkBytes > kRawAudioChunkBytes) {
    rawChunkBytes = kRawAudioChunkBytes;
  }
  if (rawChunkBytes < remaining) {
    rawChunkBytes -= rawChunkBytes % 3;
  }
  if (rawChunkBytes == 0) {
    rawChunkBytes = remaining;
  }

  state.base64Chunk.clear();
  appendBase64(state.base64Chunk, audioData + state.audioOffset, rawChunkBytes);
  state.audioOffset += rawChunkBytes;
  state.sendOffset = 0;
  return true;
}

void advanceSendPhase(HttpFramePublisher::TransportState& state) {
  using SendPhase = HttpFramePublisher::TransportState::SendPhase;
  switch (state.sendPhase) {
    case SendPhase::kHeader:
      state.sendPhase = SendPhase::kSegmentPrefix;
      break;
    case SendPhase::kSegmentPrefix:
      state.sendPhase = SendPhase::kAudio;
      break;
    case SendPhase::kAudio:
      state.sendPhase = SendPhase::kSegmentSuffix;
      break;
    case SendPhase::kSegmentSuffix:
      ++state.segmentIndex;
      state.audioOffset = 0;
      state.base64Chunk.clear();
      state.sendPhase =
          state.segmentIndex < state.payloadParts.size() ? SendPhase::kSegmentPrefix : SendPhase::kDone;
      break;
    case SendPhase::kDone:
      break;
  }
  state.sendOffset = 0;
}

void flushTx(HttpFramePublisher::TransportState& state, tcp_pcb* tpcb) {
  if (tpcb == nullptr) {
    return;
  }

  using SendPhase = HttpFramePublisher::TransportState::SendPhase;
  bool wroteAny = false;
  while (state.sendPhase != SendPhase::kDone) {
    const SendPhase phaseBefore = state.sendPhase;
    bool segmentDone = false;
    bool continueSamePhase = false;
    switch (state.sendPhase) {
      case SendPhase::kHeader:
        segmentDone = writeStringSegment(state, tpcb, state.requestHeader);
        break;
      case SendPhase::kSegmentPrefix:
        if (state.segmentIndex >= state.payloadParts.size()) {
          segmentDone = true;
        } else {
          segmentDone = writeStringSegment(state, tpcb, state.payloadParts[state.segmentIndex].prefix);
        }
        break;
      case SendPhase::kAudio:
        if (state.sendOffset >= state.base64Chunk.size() && !refillAudioChunk(state)) {
          segmentDone = true;
        } else {
          const bool chunkDone = writeStringSegment(state, tpcb, state.base64Chunk);
          if (!chunkDone) {
            segmentDone = false;
          } else if (state.segmentIndex >= state.payloadParts.size() ||
                     state.audioOffset >= state.payloadParts[state.segmentIndex].rawAudioBytes) {
            state.base64Chunk.clear();
            segmentDone = true;
          } else {
            state.base64Chunk.clear();
            continueSamePhase = true;
          }
        }
        break;
      case SendPhase::kSegmentSuffix:
        if (state.segmentIndex >= state.payloadParts.size()) {
          segmentDone = true;
        } else {
          segmentDone = writeStringSegment(state, tpcb, state.payloadParts[state.segmentIndex].suffix);
        }
        break;
      case SendPhase::kDone:
        segmentDone = true;
        break;
    }
    if (state.requestDone) {
      return;
    }
    if (continueSamePhase) {
      wroteAny = true;
      continue;
    }
    if (!segmentDone) {
      break;
    }
    if (phaseBefore == state.sendPhase) {
      advanceSendPhase(state);
      wroteAny = true;
    }
  }

  if (wroteAny || state.sendOffset > 0) {
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
    state->failureStage = PublishFailureStage::kConnect;
    state->requestDone = true;
    closeConnection(*state);
    return err;
  }

  state->connected = true;
  tcp_nagle_disable(tpcb);
  refreshRequestDeadline(*state);
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
  refreshRequestDeadline(*state);
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
    state->failureStage = PublishFailureStage::kRecv;
    state->requestDone = true;
    if (p != nullptr) {
      pbuf_free(p);
    }
    closeConnection(*state);
    return err;
  }

  if (p == nullptr) {
    refreshRequestDeadline(*state);
    state->sawResponseClose = true;
    parseHeadersIfReady(*state);
    state->requestDone = true;
    closeConnection(*state);
    return ERR_OK;
  }

  for (pbuf* q = p; q != nullptr; q = q->next) {
    const char* chunk = static_cast<const char*>(q->payload);
    refreshRequestDeadline(*state);
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
  const bool wasConnected = state->connected;
  state->pcb = nullptr;
  state->connected = false;
  state->connectDone = true;
  state->err = err;
  state->failureStage = wasConnected ? PublishFailureStage::kRecv : PublishFailureStage::kConnect;
  state->requestDone = true;
}

bool ensureConnected(
    const std::string& host,
    uint16_t port,
    absolute_time_t deadline,
    HttpFramePublisher::TransportState& state) {
  if (state.connected && state.pcb != nullptr) {
    return true;
  }
  if (state.pcb != nullptr) {
    closeConnection(state);
  }
  if (!resolveHostUntil(host, deadline, state)) {
    return false;
  }

  state.connectDone = false;
  state.err = ERR_OK;
  state.pcb = tcp_new_ip_type(IP_GET_TYPE(&state.remoteAddr));
  if (state.pcb == nullptr) {
    return false;
  }

  tcp_arg(state.pcb, &state);
  tcp_nagle_disable(state.pcb);
  tcp_recv(state.pcb, &onRecv);
  tcp_sent(state.pcb, &onSent);
  tcp_poll(state.pcb, &onPoll, 2);
  tcp_err(state.pcb, &onErr);

  const err_t connectErr = tcp_connect(state.pcb, &state.remoteAddr, port, &onConnected);
  if (connectErr != ERR_OK) {
    state.err = connectErr;
    state.failureStage = PublishFailureStage::kConnect;
    closeConnection(state);
    return false;
  }

  while (!state.connectDone && !time_reached(deadline)) {
    cyw43_arch_poll();
    runBackgroundPoll(state);
    sleep_ms(1);
  }

  if (!state.connected) {
    state.err = state.err == ERR_OK ? ERR_TIMEOUT : state.err;
    if (state.failureStage == PublishFailureStage::kNone) {
      state.failureStage = PublishFailureStage::kConnect;
    }
    abortConnection(state);
    return false;
  }

  return true;
}

bool beginConnection(
    const std::string& host,
    uint16_t port,
    HttpFramePublisher::TransportState& state) {
  if (state.connected && state.pcb != nullptr) {
    flushTx(state, state.pcb);
    return true;
  }
  if (state.pcb != nullptr) {
    closeConnection(state);
  }
  if (!resolveHostUntil(host, make_timeout_time_ms(0), state)) {
    return false;
  }

  state.connectDone = false;
  state.connected = false;
  state.err = ERR_OK;
  state.pcb = tcp_new_ip_type(IP_GET_TYPE(&state.remoteAddr));
  if (state.pcb == nullptr) {
    return false;
  }

  tcp_arg(state.pcb, &state);
  tcp_nagle_disable(state.pcb);
  tcp_recv(state.pcb, &onRecv);
  tcp_sent(state.pcb, &onSent);
  tcp_poll(state.pcb, &onPoll, 2);
  tcp_err(state.pcb, &onErr);

  const err_t connectErr = tcp_connect(state.pcb, &state.remoteAddr, port, &onConnected);
  if (connectErr != ERR_OK) {
    closeConnection(state);
    state.err = connectErr;
    state.failureStage = PublishFailureStage::kConnect;
    return false;
  }
  return true;
}

PublishResult finishRequestResult(HttpFramePublisher::TransportState& state, bool keepResponseBody) {
  PublishResult result = {};
  if (state.requestStartedMs != 0) {
    result.latencyMs = to_ms_since_boot(get_absolute_time()) - state.requestStartedMs;
  }
  result.failureStage = PublishFailureStage::kNone;
  result.lwipError = static_cast<int32_t>(state.err);
  if (state.statusCode > 0) {
    result.statusCode = state.statusCode;
  } else {
    result.statusCode = (state.err == ERR_OK || state.err == ERR_CLSD) ? -4 : -3;
    if (state.err == ERR_TIMEOUT) {
      result.failureStage = PublishFailureStage::kTimeout;
    } else if (state.failureStage != PublishFailureStage::kNone) {
      result.failureStage = state.failureStage;
    } else {
      result.failureStage = PublishFailureStage::kResponseParse;
    }
  }
  if (keepResponseBody || result.statusCode < 200 || result.statusCode >= 300) {
    result.responseBody = state.response;
  }
  result.ok = (result.statusCode >= 200 && result.statusCode < 300);
  if (result.ok) {
    result.failureStage = PublishFailureStage::kNone;
    result.lwipError = 0;
  }
  if (state.responseMustClose) {
    closeConnection(state);
  }
  state.asyncPublishActive = false;
  return result;
}

PublishResult post(
    const std::string& host,
    uint16_t port,
    const std::string& path,
    std::vector<IngestPayloadParts> payloadParts,
    std::vector<const uint8_t*> audioSegments,
    bool binaryAudio,
    uint32_t timeoutMs,
    bool keepResponseBody,
    HttpFramePublisher::TransportState& state) {
  PublishResult result = {};
  result.ok = false;
  result.statusCode = -3;
  result.failureStage = PublishFailureStage::kNone;
  result.lwipError = 0;

  if (host.empty() || path.empty()) {
    return result;
  }

  resetRequestState(state, keepResponseBody);
  size_t payloadSize = 0;
  for (const IngestPayloadParts& segment : payloadParts) {
    payloadSize += segment.prefix.size() + segment.encodedAudioBytes + segment.suffix.size();
  }
  state.requestHeader = buildRequestHeader(
      host,
      port,
      path,
      payloadSize,
      binaryAudio ? "application/octet-stream" : "application/json");
  state.payloadParts = std::move(payloadParts);
  state.audioSegments = std::move(audioSegments);
  state.binaryAudio = binaryAudio;
  state.requestTimeoutMs = timeoutMs;
  state.requestStartedMs = to_ms_since_boot(get_absolute_time());
  refreshRequestDeadline(state);

  const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
  if (!ensureConnected(host, port, deadline, state)) {
    result.statusCode = (state.err == ERR_TIMEOUT) ? -4 : -3;
    result.failureStage =
        (state.err == ERR_TIMEOUT) ? PublishFailureStage::kTimeout : state.failureStage;
    result.lwipError = static_cast<int32_t>(state.err);
    result.latencyMs = to_ms_since_boot(get_absolute_time()) - state.requestStartedMs;
    return result;
  }

  flushTx(state, state.pcb);

  while (!state.requestDone) {
    cyw43_arch_poll();
    runBackgroundPoll(state);
    if (time_reached(state.requestDeadline)) {
      break;
    }
    sleep_ms(1);
  }

  if (!state.requestDone) {
    state.err = ERR_TIMEOUT;
    state.failureStage = PublishFailureStage::kTimeout;
    abortConnection(state);
    result.statusCode = -4;
    result.failureStage = PublishFailureStage::kTimeout;
    result.lwipError = static_cast<int32_t>(ERR_TIMEOUT);
    result.latencyMs = to_ms_since_boot(get_absolute_time()) - state.requestStartedMs;
    if (keepResponseBody) {
      result.responseBody = state.response;
    }
    closeConnection(state);
    return result;
  }
  return finishRequestResult(state, keepResponseBody);
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

void HttpFramePublisher::setBackgroundPollCallback(BackgroundPollCallback callback, void* context) {
  if (transportState_ == nullptr) {
    return;
  }
  transportState_->backgroundPollCallback = callback;
  transportState_->backgroundPollContext = context;
}

bool HttpFramePublisher::publishInProgress() const {
  return transportState_ != nullptr && transportState_->asyncPublishActive;
}

void HttpFramePublisher::cancelPublish() {
  if (transportState_ == nullptr) {
    return;
  }
  abortConnection(*transportState_);
  transportState_->asyncPublishActive = false;
}

bool HttpFramePublisher::setEndpointPort(uint16_t port) {
  if (transportState_ == nullptr || transportState_->asyncPublishActive) {
    return false;
  }
  if (port == 0 || host_.empty()) {
    return false;
  }

  std::string nextEndpoint = "http://";
  const bool needsIpv6Brackets =
      host_.find(':') != std::string::npos &&
      (host_.empty() || host_.front() != '[' || host_.back() != ']');
  if (needsIpv6Brackets) {
    nextEndpoint.push_back('[');
  }
  nextEndpoint += host_;
  if (needsIpv6Brackets) {
    nextEndpoint.push_back(']');
  }
  nextEndpoint += ':';
  nextEndpoint += std::to_string(static_cast<unsigned>(port));
  nextEndpoint += path_.empty() ? std::string("/") : path_;

  const std::string previousEndpoint = endpointUrl_;
  const std::string previousHost = host_;
  const std::string previousPath = path_;
  const uint16_t previousPort = port_;
  const bool previousValid = endpointValid_;

  endpointUrl_ = nextEndpoint;
  endpointValid_ = parseEndpoint();
  if (endpointValid_) {
    return true;
  }

  endpointUrl_ = previousEndpoint;
  host_ = previousHost;
  path_ = previousPath;
  port_ = previousPort;
  endpointValid_ = previousValid;
  return false;
}

bool HttpFramePublisher::beginPublish(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;
  immediateResult.failureStage = PublishFailureStage::kNone;
  immediateResult.lwipError = 0;

  if (transportState_ == nullptr || transportState_->asyncPublishActive) {
    immediateResult.statusCode = -7;
    return false;
  }
  if (!endpointValid_) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = PublishFailureStage::kResponseParse;
    return false;
  }
  if (!isWiFiConnected()) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kWiFiDisconnected;
    return false;
  }

  IngestPayloadParts payloadParts;
  try {
    if (!buildIngestPayloadParts(node, frame, environment, payloadParts)) {
      immediateResult.statusCode = -2;
      return false;
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  resetRequestState(*transportState_, keepResponseBody);
  const size_t payloadSize =
      payloadParts.prefix.size() + payloadParts.encodedAudioBytes + payloadParts.suffix.size();
  transportState_->requestHeader = buildRequestHeader(host_, port_, path_, payloadSize, "application/json");
  transportState_->payloadParts.clear();
  transportState_->payloadParts.push_back(std::move(payloadParts));
  transportState_->audioSegments.clear();
  transportState_->audioSegments.push_back(reinterpret_cast<const uint8_t*>(frame.interleavedSamples));
  transportState_->binaryAudio = false;
  transportState_->requestTimeoutMs = timeoutMs_;
  transportState_->requestStartedMs = to_ms_since_boot(get_absolute_time());
  refreshRequestDeadline(*transportState_);

  if (!beginConnection(host_, port_, *transportState_)) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = transportState_->failureStage;
    immediateResult.lwipError = static_cast<int32_t>(transportState_->err);
    closeConnection(*transportState_);
    return false;
  }

  transportState_->asyncPublishActive = true;
  return true;
}

bool HttpFramePublisher::beginStoreForwardPublish(
    const NodeDescriptor& node,
    const std::vector<AudioFrame>& frames,
    const std::vector<const EnvironmentalSample*>& environments,
    bool sortByToa,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;
  immediateResult.failureStage = PublishFailureStage::kNone;
  immediateResult.lwipError = 0;

  if (transportState_ == nullptr || transportState_->asyncPublishActive) {
    immediateResult.statusCode = -7;
    return false;
  }
  if (!endpointValid_) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = PublishFailureStage::kResponseParse;
    return false;
  }
  if (!isWiFiConnected()) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kWiFiDisconnected;
    return false;
  }
  if (frames.empty()) {
    immediateResult.statusCode = -2;
    return false;
  }

  std::vector<IngestPayloadParts> payloadParts;
  std::vector<const EnvironmentalSample*> environmentPtrs = environments;
  if (environmentPtrs.size() < frames.size()) {
    environmentPtrs.resize(frames.size(), nullptr);
  }
  try {
    if (!buildStoreForwardPayloadParts(
            node,
            frames.data(),
            environmentPtrs.data(),
            frames.size(),
            sortByToa,
            payloadParts)) {
      immediateResult.statusCode = -2;
      return false;
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  size_t payloadSize = 0;
  std::vector<const uint8_t*> audioSegments;
  try {
    audioSegments.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      payloadSize +=
          payloadParts[i].prefix.size() +
          payloadParts[i].encodedAudioBytes +
          payloadParts[i].suffix.size();
      audioSegments.push_back(reinterpret_cast<const uint8_t*>(frames[i].interleavedSamples));
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  resetRequestState(*transportState_, keepResponseBody);
  transportState_->requestHeader = buildRequestHeader(host_, port_, path_, payloadSize, "application/json");
  transportState_->payloadParts = std::move(payloadParts);
  transportState_->audioSegments = std::move(audioSegments);
  transportState_->binaryAudio = false;
  transportState_->requestTimeoutMs = timeoutMs_;
  transportState_->requestStartedMs = to_ms_since_boot(get_absolute_time());
  refreshRequestDeadline(*transportState_);

  if (!beginConnection(host_, port_, *transportState_)) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = transportState_->failureStage;
    immediateResult.lwipError = static_cast<int32_t>(transportState_->err);
    closeConnection(*transportState_);
    return false;
  }

  transportState_->asyncPublishActive = true;
  return true;
}

bool HttpFramePublisher::beginBinaryStoreForwardPublish(
    const NodeDescriptor& node,
    const std::vector<AudioFrame>& frames,
    const std::vector<const EnvironmentalSample*>& environments,
    bool sortByToa,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;
  immediateResult.failureStage = PublishFailureStage::kNone;
  immediateResult.lwipError = 0;

  if (transportState_ == nullptr || transportState_->asyncPublishActive) {
    immediateResult.statusCode = -7;
    return false;
  }
  if (!endpointValid_) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = PublishFailureStage::kResponseParse;
    return false;
  }
  if (!isWiFiConnected()) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kWiFiDisconnected;
    return false;
  }
  if (frames.empty()) {
    immediateResult.statusCode = -2;
    return false;
  }

  std::vector<IngestPayloadParts> payloadParts;
  std::vector<const EnvironmentalSample*> environmentPtrs = environments;
  if (environmentPtrs.size() < frames.size()) {
    environmentPtrs.resize(frames.size(), nullptr);
  }
  try {
    if (!buildBinaryStoreForwardPayloadParts(
            node,
            frames.data(),
            environmentPtrs.data(),
            frames.size(),
            sortByToa,
            payloadParts)) {
      immediateResult.statusCode = -2;
      return false;
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  size_t payloadSize = 0;
  std::vector<const uint8_t*> audioSegments;
  try {
    audioSegments.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      payloadSize +=
          payloadParts[i].prefix.size() +
          payloadParts[i].encodedAudioBytes +
          payloadParts[i].suffix.size();
      audioSegments.push_back(reinterpret_cast<const uint8_t*>(frames[i].interleavedSamples));
    }
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  resetRequestState(*transportState_, keepResponseBody);
  transportState_->requestHeader = buildRequestHeader(
      host_,
      port_,
      path_,
      payloadSize,
      "application/octet-stream");
  transportState_->payloadParts = std::move(payloadParts);
  transportState_->audioSegments = std::move(audioSegments);
  transportState_->binaryAudio = true;
  transportState_->requestTimeoutMs = timeoutMs_;
  transportState_->requestStartedMs = to_ms_since_boot(get_absolute_time());
  refreshRequestDeadline(*transportState_);

  if (!beginConnection(host_, port_, *transportState_)) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = transportState_->failureStage;
    immediateResult.lwipError = static_cast<int32_t>(transportState_->err);
    closeConnection(*transportState_);
    return false;
  }

  transportState_->asyncPublishActive = true;
  return true;
}

bool HttpFramePublisher::beginJsonPost(
    const std::string& jsonBody,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;
  immediateResult.failureStage = PublishFailureStage::kNone;
  immediateResult.lwipError = 0;

  if (transportState_ == nullptr || transportState_->asyncPublishActive) {
    immediateResult.statusCode = -7;
    return false;
  }
  if (!endpointValid_) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = PublishFailureStage::kResponseParse;
    return false;
  }
  if (!isWiFiConnected()) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kWiFiDisconnected;
    return false;
  }

  IngestPayloadParts payloadPart;
  try {
    payloadPart.prefix = jsonBody;
    payloadPart.rawAudioBytes = 0;
    payloadPart.encodedAudioBytes = 0;
    payloadPart.suffix.clear();
  } catch (const std::bad_alloc&) {
    immediateResult.statusCode = -6;
    return false;
  }

  resetRequestState(*transportState_, keepResponseBody);
  transportState_->requestHeader = buildRequestHeader(
      host_,
      port_,
      path_,
      jsonBody.size(),
      "application/json");
  transportState_->payloadParts.clear();
  transportState_->payloadParts.push_back(std::move(payloadPart));
  transportState_->audioSegments.clear();
  transportState_->audioSegments.push_back(nullptr);
  transportState_->binaryAudio = false;
  transportState_->requestTimeoutMs = timeoutMs_;
  transportState_->requestStartedMs = to_ms_since_boot(get_absolute_time());
  refreshRequestDeadline(*transportState_);

  if (!beginConnection(host_, port_, *transportState_)) {
    immediateResult.statusCode = -3;
    immediateResult.failureStage = transportState_->failureStage;
    immediateResult.lwipError = static_cast<int32_t>(transportState_->err);
    closeConnection(*transportState_);
    return false;
  }

  transportState_->asyncPublishActive = true;
  return true;
}

bool HttpFramePublisher::pollPublish(PublishResult& result) {
  result = {};
  result.ok = false;
  result.statusCode = -1;
  result.failureStage = PublishFailureStage::kNone;
  result.lwipError = 0;

  if (transportState_ == nullptr || !transportState_->asyncPublishActive) {
    return false;
  }

  cyw43_arch_poll();
  runBackgroundPoll(*transportState_);
  if (transportState_->connected && transportState_->pcb != nullptr) {
    flushTx(*transportState_, transportState_->pcb);
  }

  if (!transportState_->requestDone && time_reached(transportState_->requestDeadline)) {
    transportState_->err = ERR_TIMEOUT;
    transportState_->failureStage = PublishFailureStage::kTimeout;
    abortConnection(*transportState_);
    transportState_->asyncPublishActive = false;
    result.statusCode = -4;
    result.failureStage = PublishFailureStage::kTimeout;
    result.lwipError = static_cast<int32_t>(ERR_TIMEOUT);
    result.latencyMs = to_ms_since_boot(get_absolute_time()) - transportState_->requestStartedMs;
    if (transportState_->keepResponse) {
      result.responseBody = transportState_->response;
    }
    return true;
  }

  if (!transportState_->requestDone) {
    return false;
  }

  result = finishRequestResult(*transportState_, transportState_->keepResponse);
  return true;
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
  result.failureStage = PublishFailureStage::kNone;
  result.lwipError = 0;

  if (!endpointValid_) {
    result.statusCode = -3;
    result.failureStage = PublishFailureStage::kResponseParse;
    return result;
  }
  if (!isWiFiConnected()) {
    result.statusCode = -1;
    result.failureStage = PublishFailureStage::kWiFiDisconnected;
    return result;
  }

  IngestPayloadParts payloadParts;
  try {
    if (!buildIngestPayloadParts(node, frame, environment, payloadParts)) {
      result.statusCode = -2;
      return result;
    }

    std::vector<IngestPayloadParts> payloadSegments;
    std::vector<const uint8_t*> audioSegments;
    payloadSegments.push_back(std::move(payloadParts));
    audioSegments.push_back(reinterpret_cast<const uint8_t*>(frame.interleavedSamples));
    result = post(
        host_,
        port_,
        path_,
        std::move(payloadSegments),
        std::move(audioSegments),
        false,
        timeoutMs_,
        keepResponseBody,
        *transportState_);
  } catch (const std::bad_alloc&) {
    result.statusCode = -6;
    return result;
  }
  if (!result.ok && result.statusCode < 0 && transportState_ != nullptr) {
    // Do not retry synchronously here. A timed-out POST already consumed the
    // real-time budget for the audio loop; a second immediate attempt can hold
    // loopOnce() long enough for the DMA ring to overrun or the watchdog path
    // to look like a reboot loop. NodeRunner owns retry cadence via backoff.
    closeConnection(*transportState_);
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
