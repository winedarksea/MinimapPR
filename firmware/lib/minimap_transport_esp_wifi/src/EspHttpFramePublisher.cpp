#include "mmpr/EspHttpFramePublisher.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <errno.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <unistd.h>

#include "esp_log.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"

#include "mmpr/EspWiFiSupport.h"
#include "mmpr/PlatformTime.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-http";

// A 503 may carry Retry-After. Honour it, but never longer than this: the audio
// queue keeps filling while we wait, and past a few hundred ms it is better to
// fail the batch and let NodeRunner's drop policy work.
constexpr uint32_t kMaxRetryAfterMs = 1500;

constexpr size_t kMaxResponseBytes = 2048;
constexpr int kMaxIovPerWrite = 8;

bool caseInsensitiveStartsWith(const char* text, const char* prefix) {
  while (*prefix != '\0') {
    if (std::tolower(static_cast<unsigned char>(*text)) !=
        std::tolower(static_cast<unsigned char>(*prefix))) {
      return false;
    }
    ++text;
    ++prefix;
  }
  return true;
}

}  // namespace

EspHttpFramePublisher::EspHttpFramePublisher(const EspHttpPublisherConfig& config)
    : config_(config) {
  endpointUrl_ = config.serverBaseUrl != nullptr ? config.serverBaseUrl : "";
  path_ = config.ingestPath != nullptr ? config.ingestPath : "/api/v1/ingest/binary";
  endpointValid_ = parseEndpoint();
  if (!endpointUrl_.empty() && !path_.empty()) {
    endpointUrl_ += path_;
  }
}

EspHttpFramePublisher::~EspHttpFramePublisher() {
  stopRequested_.store(true, std::memory_order_release);
  if (requestReady_ != nullptr) {
    xSemaphoreGive(requestReady_);
  }
  // The task exits on the next wake; deleting the semaphore under it would be a
  // use-after-free, so this destructor deliberately leaks it. In practice the
  // publisher lives for the whole process.
  closeSocket();
}

bool EspHttpFramePublisher::parseEndpoint() {
  // "http://host[:port]" -- the only scheme the ingest path uses. TLS would need
  // esp-tls and is out of scope for a LAN uplink measurement.
  const std::string& url = endpointUrl_;
  constexpr const char* kScheme = "http://";
  const size_t schemeLen = std::strlen(kScheme);
  if (url.size() <= schemeLen || url.compare(0, schemeLen, kScheme) != 0) {
    return false;
  }
  std::string authority = url.substr(schemeLen);
  const size_t slash = authority.find('/');
  if (slash != std::string::npos) {
    authority.resize(slash);
  }
  const size_t colon = authority.rfind(':');
  if (colon == std::string::npos) {
    host_ = authority;
    port_ = 80;
  } else {
    host_ = authority.substr(0, colon);
    const long parsed = std::strtol(authority.c_str() + colon + 1, nullptr, 10);
    if (parsed <= 0 || parsed > 65535) {
      return false;
    }
    port_ = static_cast<uint16_t>(parsed);
  }
  return !host_.empty();
}

bool EspHttpFramePublisher::begin() {
  if (!endpointValid_) {
    ESP_LOGE(kTag, "invalid endpoint '%s'", endpointUrl_.c_str());
    return false;
  }
  if (taskHandle_ != nullptr) {
    return true;
  }
  requestReady_ = xSemaphoreCreateBinary();
  if (requestReady_ == nullptr) {
    return false;
  }
  const BaseType_t created = xTaskCreatePinnedToCore(
      &EspHttpFramePublisher::taskEntry,
      "mmpr_pub",
      config_.taskStackBytes,
      this,
      config_.taskPriority,
      &taskHandle_,
      config_.taskCoreId);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "publisher task create failed");
    return false;
  }
  ESP_LOGI(kTag, "uplink %s://%s:%u%s timeout=%u ms",
           "http", host_.c_str(), static_cast<unsigned>(port_), path_.c_str(),
           static_cast<unsigned>(config_.timeoutMs));
  return true;
}

void EspHttpFramePublisher::taskEntry(void* context) {
  static_cast<EspHttpFramePublisher*>(context)->taskLoop();
}

void EspHttpFramePublisher::taskLoop() {
  while (true) {
    xSemaphoreTake(requestReady_, portMAX_DELAY);
    if (stopRequested_.load(std::memory_order_acquire)) {
      break;
    }
    if (!requestInFlight_.load(std::memory_order_acquire)) {
      continue;
    }
    runOneRequest();
    requestComplete_.store(true, std::memory_order_release);
  }
  taskHandle_ = nullptr;
  vTaskDelete(nullptr);
}

void EspHttpFramePublisher::runOneRequest() {
  PublishResult result = {};
  result.ok = false;
  result.statusCode = -1;
  result.failureStage = PublishFailureStage::kNone;

  if (retryAfterMs_ > 0) {
    sleepMs(retryAfterMs_);
    retryAfterMs_ = 0;
  }

  if (!espWiFiConnected()) {
    result.statusCode = -7;
    result.failureStage = PublishFailureStage::kWiFiDisconnected;
  } else if (ensureSocket(result) && sendRequest(result)) {
    readResponse(result);
  }

  const uint32_t nowMs = nowMonotonicMs();
  result.latencyMs = nowMs >= requestStartedMs_ ? (nowMs - requestStartedMs_) : 0;
  pendingResult_ = result;
}

bool EspHttpFramePublisher::ensureSocket(PublishResult& result) {
  if (socketFd_ >= 0) {
    return true;
  }

  char portText[8] = {};
  std::snprintf(portText, sizeof(portText), "%u", static_cast<unsigned>(port_));

  struct addrinfo hints = {};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* resolved = nullptr;
  if (getaddrinfo(host_.c_str(), portText, &hints, &resolved) != 0 || resolved == nullptr) {
    result.statusCode = -2;
    result.failureStage = PublishFailureStage::kDns;
    if (resolved != nullptr) {
      freeaddrinfo(resolved);
    }
    return false;
  }

  const int fd = socket(resolved->ai_family, resolved->ai_socktype, resolved->ai_protocol);
  if (fd < 0) {
    freeaddrinfo(resolved);
    result.statusCode = -3;
    result.failureStage = PublishFailureStage::kConnect;
    return false;
  }

  // Non-blocking connect + select so a black-holed host fails within
  // timeoutMs rather than lwIP's much longer default SYN retry schedule.
  const int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  int connectResult = connect(fd, resolved->ai_addr, resolved->ai_addrlen);
  freeaddrinfo(resolved);
  if (connectResult != 0) {
    if (errno != EINPROGRESS) {
      close(fd);
      result.statusCode = -3;
      result.failureStage = PublishFailureStage::kConnect;
      result.lwipError = errno;
      return false;
    }
    fd_set writable;
    FD_ZERO(&writable);
    FD_SET(fd, &writable);
    struct timeval timeout = {};
    timeout.tv_sec = static_cast<time_t>(config_.timeoutMs / 1000u);
    timeout.tv_usec = static_cast<suseconds_t>((config_.timeoutMs % 1000u) * 1000u);
    const int selected = select(fd + 1, nullptr, &writable, nullptr, &timeout);
    int soError = 0;
    socklen_t soErrorLen = sizeof(soError);
    if (selected <= 0 ||
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &soErrorLen) != 0 ||
        soError != 0) {
      close(fd);
      result.statusCode = selected == 0 ? -4 : -3;
      result.failureStage =
          selected == 0 ? PublishFailureStage::kTimeout : PublishFailureStage::kConnect;
      result.lwipError = soError;
      return false;
    }
  }
  fcntl(fd, F_SETFL, flags);

  struct timeval ioTimeout = {};
  ioTimeout.tv_sec = static_cast<time_t>(config_.timeoutMs / 1000u);
  ioTimeout.tv_usec = static_cast<suseconds_t>((config_.timeoutMs % 1000u) * 1000u);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &ioTimeout, sizeof(ioTimeout));
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &ioTimeout, sizeof(ioTimeout));
  const int nodelay = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

  socketFd_ = fd;
  socketOpenCount_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void EspHttpFramePublisher::closeSocket() {
  if (socketFd_ >= 0) {
    close(socketFd_);
    socketFd_ = -1;
  }
}

bool EspHttpFramePublisher::sendAll(const void* data, size_t bytes, PublishResult& result) {
  const uint8_t* cursor = static_cast<const uint8_t*>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t sent = send(socketFd_, cursor, remaining, 0);
    if (sent > 0) {
      cursor += sent;
      remaining -= static_cast<size_t>(sent);
      continue;
    }
    const bool timedOut = (sent < 0) && (errno == EAGAIN || errno == EWOULDBLOCK);
    result.statusCode = timedOut ? -4 : -3;
    result.failureStage = timedOut ? PublishFailureStage::kTimeout : PublishFailureStage::kSend;
    result.lwipError = errno;
    closeSocket();
    return false;
  }
  return true;
}

bool EspHttpFramePublisher::sendRequest(PublishResult& result) {
  // Scatter-send: header + as many (prefix, audio, suffix) triples as fit one
  // iovec batch. The audio pointers reference NodeRunner's live batch buffers
  // (see the lifetime contract in the header) -- nothing is copied here.
  struct iovec iov[kMaxIovPerWrite];
  int iovCount = 0;

  auto flush = [&]() -> bool {
    int pending = iovCount;
    struct iovec* cursor = iov;
    while (pending > 0) {
      const ssize_t written = lwip_writev(socketFd_, cursor, pending);
      if (written <= 0) {
        const bool timedOut = (written < 0) && (errno == EAGAIN || errno == EWOULDBLOCK);
        result.statusCode = timedOut ? -4 : -3;
        result.failureStage = timedOut ? PublishFailureStage::kTimeout : PublishFailureStage::kSend;
        result.lwipError = errno;
        closeSocket();
        return false;
      }
      size_t consumed = static_cast<size_t>(written);
      while (pending > 0 && consumed >= cursor->iov_len) {
        consumed -= cursor->iov_len;
        ++cursor;
        --pending;
      }
      if (pending > 0 && consumed > 0) {
        cursor->iov_base = static_cast<uint8_t*>(cursor->iov_base) + consumed;
        cursor->iov_len -= consumed;
      }
    }
    iovCount = 0;
    return true;
  };

  auto push = [&](const void* data, size_t bytes) -> bool {
    if (bytes == 0) {
      return true;
    }
    if (iovCount == kMaxIovPerWrite && !flush()) {
      return false;
    }
    iov[iovCount].iov_base = const_cast<void*>(data);
    iov[iovCount].iov_len = bytes;
    ++iovCount;
    return true;
  };

  if (!push(requestHeader_.data(), requestHeader_.size())) {
    return false;
  }
  for (size_t i = 0; i < payloadParts_.size(); ++i) {
    const IngestPayloadParts& parts = payloadParts_[i];
    const AudioSegment& audio = audioSegments_[i];
    if (!push(parts.prefix.data(), parts.prefix.size()) ||
        !push(audio.data, audio.bytes) ||
        !push(parts.suffix.data(), parts.suffix.size())) {
      return false;
    }
  }
  return flush();
}

bool EspHttpFramePublisher::readResponse(PublishResult& result) {
  responseScratch_.clear();
  char buffer[512];
  size_t headerEnd = std::string::npos;

  while (headerEnd == std::string::npos) {
    const ssize_t received = recv(socketFd_, buffer, sizeof(buffer), 0);
    if (received <= 0) {
      const bool timedOut = (received < 0) && (errno == EAGAIN || errno == EWOULDBLOCK);
      result.statusCode = timedOut ? -4 : -3;
      result.failureStage = timedOut ? PublishFailureStage::kTimeout : PublishFailureStage::kRecv;
      result.lwipError = received == 0 ? 0 : errno;
      closeSocket();
      return false;
    }
    responseScratch_.append(buffer, static_cast<size_t>(received));
    headerEnd = responseScratch_.find("\r\n\r\n");
    if (headerEnd == std::string::npos && responseScratch_.size() > kMaxResponseBytes) {
      result.statusCode = -5;
      result.failureStage = PublishFailureStage::kResponseParse;
      closeSocket();
      return false;
    }
  }

  // Status line: "HTTP/1.1 202 Accepted".
  int statusCode = -5;
  if (responseScratch_.size() > 12 && responseScratch_.compare(0, 5, "HTTP/") == 0) {
    statusCode = std::atoi(responseScratch_.c_str() + 9);
  }
  if (statusCode <= 0) {
    result.statusCode = -5;
    result.failureStage = PublishFailureStage::kResponseParse;
    closeSocket();
    return false;
  }

  size_t contentLength = 0;
  bool chunked = false;
  bool serverWillClose = false;
  uint32_t retryAfterMs = 0;
  size_t lineStart = responseScratch_.find("\r\n");
  lineStart = (lineStart == std::string::npos) ? headerEnd : lineStart + 2;
  while (lineStart < headerEnd) {
    size_t lineEnd = responseScratch_.find("\r\n", lineStart);
    if (lineEnd == std::string::npos || lineEnd > headerEnd) {
      lineEnd = headerEnd;
    }
    const char* line = responseScratch_.c_str() + lineStart;
    if (caseInsensitiveStartsWith(line, "content-length:")) {
      contentLength = static_cast<size_t>(std::strtoul(line + 15, nullptr, 10));
    } else if (caseInsensitiveStartsWith(line, "transfer-encoding:") &&
               responseScratch_.find("chunked", lineStart) < lineEnd) {
      chunked = true;
    } else if (caseInsensitiveStartsWith(line, "connection:") &&
               responseScratch_.find("close", lineStart) < lineEnd) {
      serverWillClose = true;
    } else if (caseInsensitiveStartsWith(line, "retry-after:")) {
      const unsigned long seconds = std::strtoul(line + 12, nullptr, 10);
      retryAfterMs = seconds > (kMaxRetryAfterMs / 1000u)
          ? kMaxRetryAfterMs
          : static_cast<uint32_t>(seconds * 1000u);
    }
    lineStart = lineEnd + 2;
  }

  // The body must be consumed in full or the next response on this keep-alive
  // socket parses garbage. A chunked or unbounded body is rare here (the sidecar
  // answers with a short JSON object), so fall back to closing the socket.
  size_t bodyReceived = responseScratch_.size() - (headerEnd + 4);
  if (chunked) {
    serverWillClose = true;
  } else {
    while (bodyReceived < contentLength) {
      const ssize_t received = recv(socketFd_, buffer, sizeof(buffer), 0);
      if (received <= 0) {
        serverWillClose = true;
        break;
      }
      if (keepResponseBody_) {
        responseScratch_.append(buffer, static_cast<size_t>(received));
      }
      bodyReceived += static_cast<size_t>(received);
    }
  }

  result.statusCode = statusCode;
  result.ok = statusCode >= 200 && statusCode < 300;
  if (keepResponseBody_ || !result.ok) {
    result.responseBody = responseScratch_.substr(
        headerEnd + 4, responseScratch_.size() - (headerEnd + 4));
  }
  applyStatusPolicy(result, retryAfterMs);
  if (serverWillClose) {
    closeSocket();
  }
  return result.ok;
}

void EspHttpFramePublisher::applyStatusPolicy(PublishResult& result, uint32_t retryAfterMs) {
  // Maps the ingest sidecar's documented statuses onto the runner's failure
  // vocabulary. NodeRunner has no "shrink the batch" or "never retry" seam --
  // its policy is kPublishBatchMaxRetries attempts then discard -- so 400/413
  // are surfaced as loud, clearly-actionable log lines and left to that path.
  if (result.ok) {
    return;
  }
  switch (result.statusCode) {
    case 503:
      result.failureStage = PublishFailureStage::kRecv;
      retryAfterMs_ = retryAfterMs > kMaxRetryAfterMs ? kMaxRetryAfterMs : retryAfterMs;
      ESP_LOGW(kTag, "server busy (503) retry_after=%u ms",
               static_cast<unsigned>(retryAfterMs_));
      break;
    case 413:
      result.failureStage = PublishFailureStage::kResponseParse;
      ESP_LOGE(kTag,
               "413 payload too large -- lower nodecfg::kTargetPublishBytes / "
               "kPublishBatchFrames; every batch this size will fail");
      break;
    case 499:
      // Sidecar-specific: client disconnected mid-upload. Same shape as a
      // timeout from the node's point of view, and the socket is unusable.
      result.failureStage = PublishFailureStage::kTimeout;
      closeSocket();
      break;
    case 400:
      result.failureStage = PublishFailureStage::kResponseParse;
      ESP_LOGE(kTag, "400 rejected by ingest -- payload is malformed, retrying will not help: %s",
               result.responseBody.c_str());
      break;
    default:
      if (result.statusCode >= 500) {
        result.failureStage = PublishFailureStage::kRecv;
      } else if (result.statusCode >= 400) {
        result.failureStage = PublishFailureStage::kResponseParse;
      }
      break;
  }
}

bool EspHttpFramePublisher::beginBinaryStoreForwardPublish(
    const NodeDescriptor& node,
    const std::vector<AudioFrame>& frames,
    const std::vector<const EnvironmentalSample*>& environments,
    bool sortByToa,
    bool keepResponseBody,
    PublishResult& immediateResult) {
  immediateResult = {};
  immediateResult.ok = false;
  immediateResult.statusCode = -1;

  if (!endpointValid_ || taskHandle_ == nullptr) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kConnect;
    return false;
  }
  if (requestInFlight_.load(std::memory_order_acquire)) {
    immediateResult.statusCode = -6;
    immediateResult.failureStage = PublishFailureStage::kSend;
    return false;
  }
  if (frames.empty()) {
    immediateResult.statusCode = -1;
    immediateResult.failureStage = PublishFailureStage::kSend;
    return false;
  }
  if (!espWiFiConnected()) {
    immediateResult.statusCode = -7;
    immediateResult.failureStage = PublishFailureStage::kWiFiDisconnected;
    return false;
  }

  payloadParts_.clear();
  audioSegments_.clear();
  if (!buildBinaryStoreForwardPayloadParts(
          node,
          frames.data(),
          environments.empty() ? nullptr : environments.data(),
          frames.size(),
          sortByToa,
          payloadParts_) ||
      payloadParts_.size() != frames.size()) {
    immediateResult.statusCode = -6;
    immediateResult.failureStage = PublishFailureStage::kResponseParse;
    return false;
  }

  // Parts are 1:1 with frames in index order -- sortByToa only sets a flag byte
  // in the MMB3 header, it does not permute the frames -- so part i's audio is
  // frames[i]'s buffer.
  size_t contentLength = 0;
  audioSegments_.reserve(payloadParts_.size());
  for (size_t i = 0; i < payloadParts_.size(); ++i) {
    const IngestPayloadParts& parts = payloadParts_[i];
    const AudioFrame& frame = frames[i];
    AudioSegment segment;
    segment.data = reinterpret_cast<const uint8_t*>(frame.interleavedSamples);
    segment.bytes = parts.rawAudioBytes;
    if (segment.data == nullptr || segment.bytes == 0) {
      immediateResult.statusCode = -6;
      immediateResult.failureStage = PublishFailureStage::kResponseParse;
      return false;
    }
    audioSegments_.push_back(segment);
    contentLength += parts.prefix.size() + segment.bytes + parts.suffix.size();
  }

  requestHeader_.clear();
  requestHeader_.reserve(192);
  requestHeader_ += "POST ";
  requestHeader_ += path_;
  requestHeader_ += " HTTP/1.1\r\nHost: ";
  requestHeader_ += host_;
  if (port_ != 80) {
    requestHeader_ += ':';
    requestHeader_ += std::to_string(port_);
  }
  requestHeader_ += "\r\nConnection: keep-alive\r\nContent-Type: application/octet-stream";
  requestHeader_ += "\r\nContent-Length: ";
  requestHeader_ += std::to_string(contentLength);
  requestHeader_ += "\r\n\r\n";

  keepResponseBody_ = keepResponseBody;
  requestStartedMs_ = nowMonotonicMs();
  cancelRequested_.store(false, std::memory_order_relaxed);
  requestComplete_.store(false, std::memory_order_release);
  requestInFlight_.store(true, std::memory_order_release);
  xSemaphoreGive(requestReady_);
  return true;
}

bool EspHttpFramePublisher::pollPublish(PublishResult& result) {
  if (!requestInFlight_.load(std::memory_order_acquire)) {
    return false;
  }
  if (!requestComplete_.load(std::memory_order_acquire)) {
    return false;
  }
  result = pendingResult_;
  if (cancelRequested_.load(std::memory_order_relaxed)) {
    result.ok = false;
    result.statusCode = -6;
    result.failureStage = PublishFailureStage::kSend;
  }
  requestInFlight_.store(false, std::memory_order_release);
  return true;
}

bool EspHttpFramePublisher::publishInProgress() const {
  return requestInFlight_.load(std::memory_order_acquire);
}

void EspHttpFramePublisher::cancelPublish() {
  // The publisher task is inside a blocking socket call; it cannot be
  // interrupted safely, and its request still references NodeRunner's batch
  // buffers. Flag the result as failed and let the request drain -- the socket
  // timeouts bound how long that takes.
  if (requestInFlight_.load(std::memory_order_acquire)) {
    cancelRequested_.store(true, std::memory_order_relaxed);
  }
}

void EspHttpFramePublisher::setBackgroundPollCallback(
    BackgroundPollCallback /*callback*/, void* /*context*/) {
  // Not needed: the transport blocks on its own task, so the runner task is
  // never stalled and has nothing to pump cooperatively.
}

int8_t EspHttpFramePublisher::linkRssiDbm() const {
  return espWiFiRssiDbm();
}

}  // namespace mmpr
