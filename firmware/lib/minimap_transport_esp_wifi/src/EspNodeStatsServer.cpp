#include "mmpr/EspNodeStatsServer.h"

#include <cstdio>
#include <cstring>

#include <errno.h>
#include <sys/time.h>
#include <unistd.h>

#include "esp_log.h"
#include "lwip/sockets.h"

#include "mmpr/EspHttpFramePublisher.h"
#include "mmpr/NodeRunner.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-stats";
constexpr size_t kBodyBytes = 1536;
constexpr size_t kRequestBytes = 512;

const char* boolJson(bool value) {
  return value ? "true" : "false";
}

}  // namespace

EspNodeStatsServer::EspNodeStatsServer(
    const EspNodeStatsServerConfig& config,
    const RunnerStats& runnerStats,
    const EspHttpFramePublisher& publisher)
    : config_(config), runnerStats_(runnerStats), publisher_(publisher) {}

bool EspNodeStatsServer::begin() {
  if (taskHandle_ != nullptr) {
    return true;
  }
  const BaseType_t created = xTaskCreatePinnedToCore(
      &EspNodeStatsServer::taskEntry,
      "mmpr_stats",
      config_.taskStackBytes,
      this,
      config_.taskPriority,
      &taskHandle_,
      config_.taskCoreId);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "stats task create failed");
    return false;
  }
  return true;
}

void EspNodeStatsServer::taskEntry(void* context) {
  static_cast<EspNodeStatsServer*>(context)->taskLoop();
}

void EspNodeStatsServer::taskLoop() {
  const int listenFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listenFd < 0) {
    ESP_LOGE(kTag, "listen socket failed errno=%d", errno);
    vTaskDelete(nullptr);
    return;
  }
  const int reuse = 1;
  setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  struct sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(config_.listenPort);
  if (bind(listenFd, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(listenFd, 2) != 0) {
    ESP_LOGE(kTag, "bind/listen on :%u failed errno=%d",
             static_cast<unsigned>(config_.listenPort), errno);
    close(listenFd);
    vTaskDelete(nullptr);
    return;
  }
  ESP_LOGI(kTag, "stats server on :%u%s",
           static_cast<unsigned>(config_.listenPort), config_.statsPath);

  while (true) {
    const int clientFd = accept(listenFd, nullptr, nullptr);
    if (clientFd < 0) {
      continue;
    }
    // A stuck poller must not wedge the single-threaded accept loop.
    struct timeval timeout = {};
    timeout.tv_sec = 2;
    setsockopt(clientFd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(clientFd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    serveClient(clientFd);
    close(clientFd);
  }
}

void EspNodeStatsServer::serveClient(int clientFd) {
  char request[kRequestBytes] = {};
  const ssize_t received = recv(clientFd, request, sizeof(request) - 1, 0);
  if (received <= 0) {
    return;
  }
  request[received] = '\0';

  char target[128] = {};
  if (std::sscanf(request, "GET %127s", target) != 1) {
    static const char* kMethodNotAllowed =
        "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\n"
        "Content-Length: 26\r\nConnection: close\r\n\r\n{\"detail\":\"use GET only\"}\n";
    send(clientFd, kMethodNotAllowed, std::strlen(kMethodNotAllowed), 0);
    return;
  }
  // Ignore any query string.
  char* query = std::strchr(target, '?');
  if (query != nullptr) {
    *query = '\0';
  }

  char body[kBodyBytes] = {};
  size_t bodyLength = 0;
  int statusCode = 200;
  const char* reason = "OK";
  if (std::strcmp(target, config_.statsPath) == 0) {
    bodyLength = buildStatsBody(body, sizeof(body));
  } else if (std::strcmp(target, config_.routePath) == 0) {
    bodyLength = buildTargetBody(body, sizeof(body));
  } else {
    statusCode = 404;
    reason = "Not Found";
    bodyLength = static_cast<size_t>(
        std::snprintf(body, sizeof(body), "{\"detail\":\"unknown route\"}"));
  }
  if (bodyLength == 0) {
    statusCode = 500;
    reason = "Internal Server Error";
    bodyLength = static_cast<size_t>(
        std::snprintf(body, sizeof(body), "{\"detail\":\"response too large\"}"));
  }

  char header[160] = {};
  const int headerLength = std::snprintf(
      header,
      sizeof(header),
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %u\r\n"
      "Connection: close\r\n\r\n",
      statusCode,
      reason,
      static_cast<unsigned>(bodyLength));
  if (headerLength <= 0) {
    return;
  }
  send(clientFd, header, static_cast<size_t>(headerLength), 0);
  send(clientFd, body, bodyLength, 0);
}

size_t EspNodeStatsServer::buildStatsBody(char* out, size_t outBytes) const {
  // Field-for-field the same object NodeControlServer::prepareStatsBody emits on
  // the Pico node -- keep the two in sync or the A/B tooling silently loses keys.
  const RunnerStats& stats = runnerStats_;
  const int written = std::snprintf(
      out,
      outBytes,
      "{\"frames_captured\":%llu,\"frames_published\":%llu,\"frames_dropped\":%llu,"
      "\"publish_errors\":%llu,\"packet_continuity_violations\":%llu,"
      "\"queue_overflows\":%llu,\"discarded_batches\":%llu,\"queue_depth\":%u,\"queue_slots_high_water\":%u,"
      "\"queue_slots_capacity\":%u,\"queued_packet_depth\":%u,\"queued_packet_capacity\":%u,"
      "\"active_publish_packet_depth\":%u,\"active_publish_packet_capacity\":%u,\"last_packet_age_us\":%llu,"
      "\"last_audio_drain_duration_us\":%u,\"last_publisher_pump_duration_us\":%u,"
      "\"last_holdover_maintenance_duration_us\":%u,"
      "\"ring_frames_high_water\":%u,\"ring_frames_capacity\":%u,"
      "\"last_publish_status\":%d,\"last_publish_failure_stage\":%u,"
      "\"last_publish_lwip_error\":%ld,\"consecutive_publish_failures\":%u,"
      "\"publish_latency_last_ms\":%u,\"publish_latency_ewma_ms\":%u,"
      "\"publish_latency_max_ms\":%u,\"wifi_rssi_dbm\":%d,\"heap_free_bytes\":%lu,"
      "\"boot_id\":%lu,\"socket_open_count\":%lu}",
      static_cast<unsigned long long>(stats.framesCaptured),
      static_cast<unsigned long long>(stats.framesPublished),
      static_cast<unsigned long long>(stats.framesDropped),
      static_cast<unsigned long long>(stats.publishErrors),
      static_cast<unsigned long long>(stats.packetContinuityViolations),
      static_cast<unsigned long long>(stats.queueOverflows),
      static_cast<unsigned long long>(stats.discardedBatches),
      static_cast<unsigned>(stats.queueDepth),
      static_cast<unsigned>(stats.queueSlotsHighWater),
      static_cast<unsigned>(stats.queueSlotsCapacity),
      static_cast<unsigned>(stats.queuedPacketDepth),
      static_cast<unsigned>(stats.queuedPacketCapacity),
      static_cast<unsigned>(stats.activePublishPacketDepth),
      static_cast<unsigned>(stats.activePublishPacketCapacity),
      static_cast<unsigned long long>(stats.lastPacketAgeUs),
      static_cast<unsigned>(stats.lastAudioDrainDurationUs),
      static_cast<unsigned>(stats.lastPublisherPumpDurationUs),
      static_cast<unsigned>(stats.lastHoldoverMaintenanceDurationUs),
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
      static_cast<unsigned long>(stats.bootId),
      static_cast<unsigned long>(publisher_.socketOpenCount()));
  if (written <= 0 || static_cast<size_t>(written) >= outBytes) {
    return 0;
  }
  return static_cast<size_t>(written);
}

size_t EspNodeStatsServer::buildTargetBody(char* out, size_t outBytes) const {
  const int written = std::snprintf(
      out,
      outBytes,
      "{\"control_port\":%u,\"control_path\":\"%s\",\"target_url\":\"%s\",\"target_port\":%u,"
      "\"endpoint_valid\":%s,\"publish_in_progress\":%s,\"allow_runtime_port_change\":false,"
      "\"runtime_only\":true}",
      static_cast<unsigned>(config_.listenPort),
      config_.routePath,
      publisher_.endpointUrl().c_str(),
      static_cast<unsigned>(publisher_.endpointPort()),
      boolJson(publisher_.endpointValid()),
      boolJson(publisher_.publishInProgress()));
  if (written <= 0 || static_cast<size_t>(written) >= outBytes) {
    return 0;
  }
  return static_cast<size_t>(written);
}

}  // namespace mmpr
