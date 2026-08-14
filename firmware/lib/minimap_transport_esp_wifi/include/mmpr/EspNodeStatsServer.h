#pragma once

// Read-only stats/control listener, ABI-compatible with the Pico node's
// NodeControlServer.
//
// This exists so the existing A/B tooling works against the ESP node with no
// changes:
//   scripts/collect_ingest_regression_metrics.py --node-base-url http://<ip>:8082
//   scripts/node_stress_report.py
// Both read GET /api/v1/stats and expect the exact key set that
// NodeControlServer::prepareStatsBody emits, so this server reproduces that JSON
// field for field (minus the ble_scan / ble_reports sub-objects, which have no
// counterpart on this node).
//
// GET /api/v1/publish-target is served too, in the same shape, but read-only:
// the Pico's runtime port-change POST is not reproduced here. Changing the
// target mid-run was a CYW43 bring-up affordance and is not needed for the
// throughput comparison.

#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace mmpr {

struct RunnerStats;
class EspHttpFramePublisher;

struct EspNodeStatsServerConfig {
  uint16_t listenPort = 8082;
  const char* statsPath = "/api/v1/stats";
  const char* routePath = "/api/v1/publish-target";
  int taskCoreId = 0;
  uint32_t taskPriority = 3;
  uint32_t taskStackBytes = 4096;
};

class EspNodeStatsServer {
 public:
  EspNodeStatsServer(
      const EspNodeStatsServerConfig& config,
      const RunnerStats& runnerStats,
      const EspHttpFramePublisher& publisher);

  EspNodeStatsServer(const EspNodeStatsServer&) = delete;
  EspNodeStatsServer& operator=(const EspNodeStatsServer&) = delete;

  bool begin();
  uint16_t listenPort() const { return config_.listenPort; }

 private:
  static void taskEntry(void* context);
  void taskLoop();
  void serveClient(int clientFd);
  size_t buildStatsBody(char* out, size_t outBytes) const;
  size_t buildTargetBody(char* out, size_t outBytes) const;

  EspNodeStatsServerConfig config_;
  const RunnerStats& runnerStats_;
  const EspHttpFramePublisher& publisher_;
  TaskHandle_t taskHandle_ = nullptr;
};

}  // namespace mmpr
