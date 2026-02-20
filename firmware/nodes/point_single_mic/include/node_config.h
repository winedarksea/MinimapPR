#pragma once

#include "mmpr/Esp32I2SMonoSource.h"
#include "mmpr/Types.h"

namespace nodecfg {

static constexpr const char* kWifiSsid = "REPLACE_WIFI_SSID";
static constexpr const char* kWifiPassword = "REPLACE_WIFI_PASSWORD";
static constexpr const char* kServerBaseUrl = "http://192.168.1.50:8080";
static constexpr const char* kIngestPath = "/api/v1/ingest/frame";

static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
static constexpr uint32_t kHttpTimeoutMs = 2000;

static constexpr const char* kNodeId = "point-node-01";
static constexpr mmpr::NodeType kNodeType = mmpr::NodeType::kPoint;
static constexpr mmpr::Vec3 kNodePositionM = {0.0f, 0.0f, 2.0f};

static constexpr mmpr::Vec3 kSensorOffsetsM[] = {
    {0.0f, 0.0f, 0.0f},
};

static constexpr const char* kCapabilities[] = {
    "audio",
    "gps_pps",
};

static constexpr const char* kHardwareName = "esp32_point_i2s_mic";

static constexpr mmpr::Esp32I2SPins kI2sPins = {
    26,
    25,
    33,
};

static constexpr mmpr::Esp32I2SMonoConfig kAudioConfig = {
    I2S_NUM_0,
    16000,
    1024,
    16,
    8,
    256,
    false,
    0,
};

static constexpr bool kEnableNtpSync = true;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;
static constexpr uint32_t kLogEveryFrames = 100;

}  // namespace nodecfg
