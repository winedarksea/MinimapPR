// sirith_tetra_s3 — 4-mic tetrahedral node on an ESP32-S3.
//
// Same node type, same geometry, same MMB3 wire format and same audio
// parameters (32 kHz / 4 ch / 512 samples per frame) as the Pico 2 W
// sirith_tetrahedral node. The only things that differ are the radio, the
// capture peripheral, and the node id -- which is exactly the point: the live
// Bluefin deployment measures ~45 % audio frame loss on the Pico node with
// firmware reporting PUBLISH_QUEUE_OVERFLOW, and the prime suspect is the
// CYW43439 gSPI uplink against an assumed-but-never-measured 1.5 Mbit/s goodput
// floor. This node swaps the radio for an integrated Wi-Fi MAC and produces a
// number to compare.
//
// The shared runtime is reused verbatim: NodeProtocol (the MMB3 serializer),
// NodeRunner (capture -> packetize -> queue -> batch -> publish, and every
// RunnerStats counter the measurement rests on) and NodeClock all build from
// firmware/lib/minimap_node_runtime with no ESP-specific branches beyond the
// four primitives behind mmpr/PlatformTime.h.
//
// Task model (see nodecfg's "Task placement"):
//   core 1  capture   EspI2sTdmSource's task -- the DMA-IRQ equivalent
//   core 1  runner    this file's runnerTask -> NodeRunner::loopOnce()
//   core 0  publisher EspHttpFramePublisher's task -- blocking sockets
//   core 0  stats/TFT everything else that blocks
//
// The server needs no changes: there is no registration handshake and no
// heartbeat endpoint, the full NodeDescriptor rides in every MMB3 batch, and the
// server upserts it with last_seen_ns from its own receive clock.

#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "esp_log.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "node_config.h"

#include "mmpr/EspHttpFramePublisher.h"
#include "mmpr/EspI2sTdmLoopback.h"
#include "mmpr/EspI2sTdmSource.h"
#include "mmpr/EspNodeStatsServer.h"
#include "mmpr/EspSntpClient.h"
#include "mmpr/EspWiFiSupport.h"
#include "mmpr/NodeClock.h"
#include "mmpr/NodeRunner.h"
#include "mmpr/PlatformTime.h"
#include "mmpr/SyntheticAudioSource.h"
#include "mmpr/TftStatusDisplay.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace {

constexpr const char* kTag = "sirith-s3";

// --- Identity -------------------------------------------------------------

char gNodeId[48] = {};

void initNodeId() {
  uint8_t mac[6] = {};
  if (esp_efuse_mac_get_default(mac) != ESP_OK) {
    std::snprintf(gNodeId, sizeof(gNodeId), "%s0000", nodecfg::kNodeIdPrefix);
    return;
  }
  std::snprintf(gNodeId, sizeof(gNodeId), "%s%02x%02x", nodecfg::kNodeIdPrefix, mac[4], mac[5]);
}

// --- Boot count -----------------------------------------------------------
// NON-NEGOTIABLE, not telemetry garnish: the server derives
// boot_session = f"boot-{boot_count}" and keys replay dedupe on it
// (core/live_ingest_state.py). A boot count stuck at a previous value makes
// every frame after a reboot look like a duplicate and silently vanish.

constexpr const char* kNvsNamespace = "mmpr";
constexpr const char* kNvsBootCountKey = "boot_count";

uint32_t incrementBootCount() {
  nvs_handle_t handle = 0;
  if (nvs_open(kNvsNamespace, NVS_READWRITE, &handle) != ESP_OK) {
    ESP_LOGE(kTag, "NVS open failed; boot_count will not advance and the server "
                   "will treat post-reboot frames as replays");
    return 0;
  }
  uint32_t bootCount = 0;
  const esp_err_t readErr = nvs_get_u32(handle, kNvsBootCountKey, &bootCount);
  if (readErr != ESP_OK && readErr != ESP_ERR_NVS_NOT_FOUND) {
    ESP_LOGW(kTag, "NVS read failed: %s", esp_err_to_name(readErr));
  }
  ++bootCount;
  if (nvs_set_u32(handle, kNvsBootCountKey, bootCount) != ESP_OK ||
      nvs_commit(handle) != ESP_OK) {
    ESP_LOGE(kTag, "NVS boot_count commit failed");
  }
  nvs_close(handle);
  return bootCount;
}

// --- Node descriptor ------------------------------------------------------

mmpr::NodeDescriptor gNodeDescriptor = {
    gNodeId,
    nodecfg::kNodeType,
    nodecfg::kNodeHasFallbackGeoPosition,
    {
        nodecfg::kNodeFallbackLatitudeDeg,
        nodecfg::kNodeFallbackLongitudeDeg,
        nodecfg::kNodeFallbackAltitudeM,
    },
    nodecfg::kSensorOffsetsM,
    nodecfg::kSensorOffsetCount,
    nodecfg::kCapabilities,
    nodecfg::kCapabilityCount,
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
    nodecfg::kGpsSignalStatus,
    nodecfg::kGpsPositionSource,
    0u,
};

// --- Audio ----------------------------------------------------------------

mmpr::EspI2sTdmPins gTdmPins = {
    nodecfg::kTdmBclkPin,
    nodecfg::kTdmWsPin,
    nodecfg::kTdmDataPin,
    nodecfg::kEnableI2sLoopback ? nodecfg::kTdmLoopbackDoutPin : -1,
};

mmpr::EspI2sTdmConfig makeTdmConfig() {
  mmpr::EspI2sTdmConfig config;
  config.sampleRateHz = nodecfg::kActiveAudioSampleRateHz;
  config.frameSamples = nodecfg::kActiveAudioFrameSamples;
  config.tdmSlots = nodecfg::kAudioTdmSlots;
  config.slotBits = nodecfg::kAudioSlotBits;
  config.validBits = nodecfg::kAudioValidBits;
  config.captureBitOffset = nodecfg::kTdmCaptureBitOffset;
  config.wsWidth = nodecfg::kTdmWsWidth;
  config.wsPolarityHigh = nodecfg::kTdmWsPolarityHigh;
  config.bitShift = nodecfg::kTdmBitShift;
  config.leftAlign = nodecfg::kTdmLeftAlign;
  config.dmaFrameNum = nodecfg::kI2sDmaFrameNum;
  config.dmaDescNum = nodecfg::kI2sDmaDescNum;
  config.ringFrames = nodecfg::kAudioRingFrames;
  std::memcpy(config.outputChannelToSlot, nodecfg::kOutputChannelToSlot,
              sizeof(config.outputChannelToSlot));
  config.i2sPort = nodecfg::kI2sPort;
  config.captureTaskCoreId = nodecfg::kCaptureTaskCore;
  config.captureTaskPriority = nodecfg::kCaptureTaskPriority;
  return config;
}

mmpr::EspI2sTdmSource gTdmSource(gTdmPins, makeTdmConfig());
mmpr::EspI2sTdmLoopback gLoopback(gTdmSource);

// Synthetic audio produces the exact same data rate as the TDM path, so the
// uplink measurement is valid on a bare board with nothing attached. This is the
// default build (see MMPR_NODECFG_AUDIO_INPUT_MODE in node_config.h).
mmpr::SyntheticAudioSource gSyntheticAudioSource(
    nodecfg::kActiveAudioSampleRateHz,
    nodecfg::kActiveAudioFrameSamples,
    nodecfg::kActiveAudioChannels);

mmpr::IAudioSource& gAudioSource = nodecfg::kUseTdmAudio
    ? static_cast<mmpr::IAudioSource&>(gTdmSource)
    : static_cast<mmpr::IAudioSource&>(gSyntheticAudioSource);

// --- Uplink, clock, runner ------------------------------------------------

mmpr::EspHttpPublisherConfig makePublisherConfig() {
  mmpr::EspHttpPublisherConfig config;
  config.serverBaseUrl = nodecfg::kServerBaseUrl;
  config.ingestPath = nodecfg::kIngestPath;
  config.timeoutMs = nodecfg::kHttpTimeoutMs;
  config.taskCoreId = nodecfg::kPublisherTaskCore;
  config.taskPriority = nodecfg::kPublisherTaskPriority;
  return config;
}

mmpr::EspHttpFramePublisher gPublisher(makePublisherConfig());
mmpr::NodeClock gClock;
mmpr::EspSntpClient gSntp;

mmpr::NodeRunner gRunner(
    gNodeDescriptor,
    gAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    /*environmentalSource=*/nullptr,
    nodecfg::kMaxPacketSamplesPerChannel,
    nodecfg::kPublishFailureBackoffMs,
    nodecfg::kPublishBatchFrames,
    nodecfg::kPublishBatchByteBudget,
    nodecfg::kUsePublishBatchByteBudget,
    nodecfg::kAudioQueueSlots,
    nodecfg::kPublishBatchMaxRetries,
    nodecfg::kEnableClockHoldoverMaintenance);

mmpr::EspNodeStatsServerConfig makeStatsConfig() {
  mmpr::EspNodeStatsServerConfig config;
  config.listenPort = nodecfg::kStatsServerPort;
  config.statsPath = nodecfg::kStatsPath;
  config.routePath = nodecfg::kPublishTargetPath;
  config.taskCoreId = nodecfg::kAuxTaskCore;
  return config;
}

mmpr::EspNodeStatsServer gStatsServer(makeStatsConfig(), gRunner.stats(), gPublisher);

// --- TFT ------------------------------------------------------------------

mmpr::TftStatusModel gStatusModel;

mmpr::TftStatusPins gTftPins = {
    nodecfg::kTftSckPin,
    nodecfg::kTftMosiPin,
    nodecfg::kTftCsPin,
    nodecfg::kTftDcPin,
    nodecfg::kTftResetPin,
    nodecfg::kTftBacklightPin,
    nodecfg::kTftPowerEnablePin,
};

mmpr::TftStatusConfig makeTftConfig() {
  mmpr::TftStatusConfig config;
  config.refreshIntervalMs = nodecfg::kTftRefreshIntervalMs;
  config.taskCoreId = nodecfg::kAuxTaskCore;
  return config;
}

mmpr::TftStatusDisplay gTftDisplay(gTftPins, makeTftConfig(), gRunner.stats(), gStatusModel);

const char* timeQualityName(mmpr::TimeQuality quality) {
  switch (quality) {
    case mmpr::TimeQuality::kGpsLocked: return "gps_locked";
    case mmpr::TimeQuality::kGpsHoldover: return "gps_hold";
    case mmpr::TimeQuality::kNtpDisciplined: return "ntp";
    case mmpr::TimeQuality::kFreeRunning: return "free_run";
  }
  return "unknown";
}

void refreshStatusModel() {
  gStatusModel.clockQuality = timeQualityName(gClock.timeQuality());
  gStatusModel.estimatedPpm = gClock.estimatedPpm();
  gStatusModel.ringFramesAvailable = gAudioSource.availableFrames();
  gStatusModel.ringFramesCapacity = gAudioSource.ringFramesCapacity();
  gStatusModel.ipAddress = mmpr::espWiFiIpAddress();
  gStatusModel.ntpSyncCount = gSntp.syncCount();
  gStatusModel.driverOverflowCount =
      nodecfg::kUseTdmAudio ? gTdmSource.driverOverflowCount() : 0u;
}

// Runner task (B). NodeRunner stays single-threaded here, preserving every
// invariant it relies on; capture and publish are the only things that moved off
// this thread relative to the Pico node.
void runnerTask(void* /*context*/) {
  uint32_t lastAuxRefreshMs = 0;
  while (true) {
    gRunner.loopOnce();

    // Slow-cadence work the runner does not own: drain any NTP observation the
    // SNTP callback latched, refresh the display model, and keep
    // RunnerStats::heapFreeBytes populated (a field the Pico node leaves at
    // zero, and one the server's transport-health section already carries).
    const uint32_t nowMs = mmpr::nowMonotonicMs();
    if ((nowMs - lastAuxRefreshMs) >= 1000u) {
      lastAuxRefreshMs = nowMs;
      uint64_t utcNs = 0;
      uint64_t monotonicUs = 0;
      uint64_t roundTripUs = 0;
      if (gSntp.takePendingObservation(utcNs, monotonicUs, roundTripUs)) {
        gClock.applyNtpObservation(utcNs, monotonicUs, roundTripUs);
      }
      refreshStatusModel();
    }
  }
}

void logDerivedConfig() {
  ESP_LOGI(kTag,
           "audio %u Hz x%u ch frame=%u ring=%u queue=%u batch=%u timeout=%u ms",
           static_cast<unsigned>(nodecfg::kActiveAudioSampleRateHz),
           static_cast<unsigned>(nodecfg::kActiveAudioChannels),
           static_cast<unsigned>(nodecfg::kActiveAudioFrameSamples),
           static_cast<unsigned>(nodecfg::kAudioRingFrames),
           static_cast<unsigned>(nodecfg::kAudioQueueSlots),
           static_cast<unsigned>(nodecfg::kPublishBatchFrames),
           static_cast<unsigned>(nodecfg::kHttpTimeoutMs));
  const uint32_t bitsPerSecond = nodecfg::kActiveAudioSampleRateHz *
                                 nodecfg::kActiveAudioChannels * 16u;
  ESP_LOGI(kTag, "audio payload %u kbit/s (assumed uplink floor %u kbit/s)",
           static_cast<unsigned>(bitsPerSecond / 1000u),
           static_cast<unsigned>(nodecfg::kAssumedFloorKbitPerSec));
}

}  // namespace

extern "C" void app_main() {
  initNodeId();
  gStatusModel.nodeId = gNodeId;
  gStatusModel.ssid = nodecfg::kWifiSsid;
  gStatusModel.sampleRateHz = nodecfg::kActiveAudioSampleRateHz;
  gStatusModel.channels = nodecfg::kActiveAudioChannels;
  gStatusModel.audioMode = nodecfg::kUseTdmAudio ? "tdm" : "synthetic";

  ESP_LOGI(kTag, "booting id=%s hardware=%s firmware=%s",
           gNodeId, nodecfg::kHardwareName, MMPR_FW_VERSION);
  logDerivedConfig();

  {
    mmpr::NodeClockHoldoverConfig holdoverConfig;
    holdoverConfig.enableTempComp = nodecfg::kEnableHoldoverTempComp;
    holdoverConfig.driftUncertaintyPpm = nodecfg::kHoldoverDriftUncertaintyPpm;
    holdoverConfig.errorBudgetNs = nodecfg::kHoldoverErrorBudgetNs;
    holdoverConfig.maxAgeUs = nodecfg::kHoldoverMaxAgeUs;
    gClock.setHoldoverConfig(holdoverConfig);
  }

  // Wi-Fi first: it also initialises NVS, which the boot count needs.
  mmpr::EspWiFiConfig wifiConfig;
  wifiConfig.ssid = nodecfg::kWifiSsid;
  wifiConfig.password = nodecfg::kWifiPassword;
  wifiConfig.connectTimeoutMs = nodecfg::kWiFiConnectTimeoutMs;
  wifiConfig.hostname = gNodeId;
  if (!mmpr::espWiFiBegin(wifiConfig)) {
    // Keep going. The publisher reports kWiFiDisconnected until the auto-reconnect
    // handler succeeds, and capture/telemetry stay useful in the meantime.
    ESP_LOGE(kTag, "Wi-Fi did not come up; continuing and retrying in background");
  }

  gNodeDescriptor.bootCount = incrementBootCount();
  ESP_LOGI(kTag, "boot_count=%u ip=%s",
           static_cast<unsigned>(gNodeDescriptor.bootCount), mmpr::espWiFiIpAddress());

  if (nodecfg::kEnableNtpSync) {
    mmpr::EspSntpConfig sntpConfig;
    sntpConfig.server = nodecfg::kNtpServer;
    sntpConfig.syncIntervalMs = nodecfg::kNtpSyncIntervalMs;
    sntpConfig.initialSyncTimeoutMs = nodecfg::kNtpInitialSyncTimeoutMs;
    gSntp.begin(sntpConfig);
  }

  if (nodecfg::kEnableTftStatus && !gTftDisplay.begin()) {
    ESP_LOGW(kTag, "TFT status page unavailable; continuing headless");
  }

  if (!gPublisher.begin()) {
    ESP_LOGE(kTag, "fatal: uplink transport failed to start");
    while (true) {
      mmpr::sleepMs(1000);
    }
  }

  // Boot-time TDM self-test. Runs before NodeRunner starts so a framing or
  // channel-order fault is a loud line on the console rather than plausible
  // wrong audio in the pipeline. Requires the DOUT -> DIN jumper.
  if (nodecfg::kUseTdmAudio && nodecfg::kEnableI2sLoopback) {
    if (!gTdmSource.begin()) {
      ESP_LOGE(kTag, "fatal: TDM capture failed to start");
      while (true) {
        mmpr::sleepMs(1000);
      }
    }
    if (gLoopback.begin()) {
      mmpr::EspI2sTdmLoopbackResult result = {};
      if (!gLoopback.runSelfTest(result)) {
        ESP_LOGE(kTag, "loopback self-test could not run");
      } else if (!result.passed) {
        ESP_LOGE(kTag, "loopback self-test FAILED -- TDM framing or channel map is wrong");
      } else {
        ESP_LOGI(kTag, "loopback self-test passed over %u frames",
                 static_cast<unsigned>(result.framesChecked));
      }
    }
    // The self-test leaves the source running; NodeRunner::begin() below will
    // call begin() again, which is a no-op once initialised.
  }

  if (!gRunner.begin(/*syncNtp=*/false)) {
    ESP_LOGE(kTag, "fatal: runner failed to start");
    while (true) {
      mmpr::sleepMs(1000);
    }
  }

  if (nodecfg::kEnableStatsServer && !gStatsServer.begin()) {
    ESP_LOGW(kTag, "stats server unavailable on :%u",
             static_cast<unsigned>(nodecfg::kStatsServerPort));
  }

  refreshStatusModel();

  TaskHandle_t runnerHandle = nullptr;
  const BaseType_t created = xTaskCreatePinnedToCore(
      &runnerTask,
      "mmpr_runner",
      nodecfg::kRunnerTaskStackBytes,
      nullptr,
      nodecfg::kRunnerTaskPriority,
      &runnerHandle,
      nodecfg::kRunnerTaskCore);
  if (created != pdPASS) {
    ESP_LOGE(kTag, "fatal: runner task create failed");
    while (true) {
      mmpr::sleepMs(1000);
    }
  }

  // app_main returns; the runner, capture, publisher, stats and TFT tasks carry
  // the node from here.
  ESP_LOGI(kTag, "running: id=%s -> %s", gNodeId, gPublisher.endpointUrl().c_str());
}
