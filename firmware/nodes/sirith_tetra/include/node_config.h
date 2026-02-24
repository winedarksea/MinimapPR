#pragma once

#include "mmpr/SirithDualI2SSource.h"
#include "mmpr/Types.h"

namespace nodecfg {

// --- Network and backend ---
static constexpr const char* kWifiSsid = "REPLACE_WIFI_SSID";
static constexpr const char* kWifiPassword = "REPLACE_WIFI_PASSWORD";
static constexpr const char* kServerBaseUrl = "http://192.168.1.50:8080";
static constexpr const char* kIngestPath = "/api/v1/ingest/frame";

static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
static constexpr uint32_t kHttpTimeoutMs = 2000;

// --- Node identity ---
static constexpr const char* kNodeId = "sirith-tetra-01";
static constexpr mmpr::NodeType kNodeType = mmpr::NodeType::kSirithTetra;
static constexpr mmpr::Vec3 kNodePositionM = {6.0f, 0.0f, 2.0f};

// Regular tetrahedron, 50 mm edge, centered at origin.
static constexpr mmpr::Vec3 kSensorOffsetsM[] = {
    {-0.025000f, -0.014434f, -0.010206f},
    {0.025000f, -0.014434f, -0.010206f},
    {0.000000f, 0.028868f, -0.010206f},
    {0.000000f, 0.000000f, 0.030619f},
};

static constexpr const char* kCapabilities[] = {
    "audio",
    "array_localization",
    "temperature",
};

static constexpr const char* kHardwareName = "sirith_tetra_dual_i2s";

// --- Audio capture ---
// Sirith in dual-I2S mode: two stereo streams -> four channels.
// Update these pin mappings to your board wiring.
static constexpr mmpr::SirithDualI2SPins kI2sPins = {
    5,
    6,
    7,
    8,
    9,
    10,
};

static constexpr mmpr::SirithDualI2SConfig kAudioConfig = {
    16000,
    1024,
    16,
    8,
    256,
    false,
};

// --- Optional IMU temperature telemetry (LSM6 family) ---
// Update I2C pin mapping for your ESP32-S3 board wiring.
static constexpr bool kEnableImuTemperature = true;
static constexpr int kI2cSdaPin = 3;
static constexpr int kI2cSclPin = 4;
static constexpr uint8_t kImuI2cAddressPrimary7Bit = 0x6A;
static constexpr uint8_t kImuI2cAddressSecondary7Bit = 0x6B;
static constexpr uint32_t kImuTemperatureSampleIntervalMs = 2000;

// --- Time sync ---
static constexpr bool kEnableNtpSync = true;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

}  // namespace nodecfg
