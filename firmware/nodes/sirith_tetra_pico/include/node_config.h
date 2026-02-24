#pragma once

#include "mmpr/SirithPicoTdmSource.h"
#include "mmpr/Types.h"

namespace nodecfg {

// --- Network and backend ---
static constexpr const char* kWifiSsid = "REPLACE_WIFI_SSID";
static constexpr const char* kWifiPassword = "REPLACE_WIFI_PASSWORD";
static constexpr const char* kServerBaseUrl = "http://192.168.1.50:8080";
static constexpr const char* kIngestPath = "/api/v1/ingest/frame";

static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
static constexpr uint32_t kHttpTimeoutMs = 2500;

// --- Node identity ---
static constexpr const char* kNodeId = "sirith-tetra-pico-01";
static constexpr mmpr::NodeType kNodeType = mmpr::NodeType::kSirithTetra;
static constexpr mmpr::Vec3 kNodePositionM = {6.0f, 0.0f, 2.0f};

// Physical microphone geometry (regular tetrahedron, 50 mm edge length).
// Mic indices:
//   0,1,2 = base triangle
//   3 = top vertex (MK4)
static constexpr mmpr::Vec3 kPhysicalSensorOffsetsM[] = {
    {-0.025000f, -0.014434f, -0.010206f},
    {0.025000f, -0.014434f, -0.010206f},
    {0.000000f, 0.028868f, -0.010206f},
    {0.000000f, 0.000000f, 0.030619f},
};

// For calibration: rotate only the base plane indexing while keeping MK4 fixed.
// 0 = no rotation, 1 = +120 degrees, 2 = +240 degrees.
static constexpr uint8_t kBasePlaneRotationSteps = 0;

// TDM slot mapping from hardware strap configuration.
// Default assumption from board wiring/jumper plan:
//   slot 1 -> base mic 0
//   slot 2 -> base mic 1
//   slot 3 -> top mic (MK4)
//   slot 4 -> base mic 2
static constexpr uint8_t kSlotToPhysicalMic[4] = {0, 1, 3, 2};

static constexpr const char* kCapabilities[] = {
    "audio",
    "array_localization",
    "gps_optional",
};

static constexpr const char* kHardwareName = "sirith_tetra_pico_tdm";

// --- Audio capture (TDM master) ---
static constexpr mmpr::SirithPicoTdmPins kTdmPins = {
    7,   // SDATA input
    8,   // BCLK output
    9,   // FSYNC/WS output (must be BCLK + 1 for current PIO program)
};

static constexpr mmpr::SirithPicoTdmConfig kAudioConfig = {
    16000,              // sampleRateHz
    1024,               // frameSamples
    16,                 // sampleShiftBits (32-bit slot -> 16-bit PCM)
    4,                  // tdmSlots (TDM-4)
    32,                 // slotBits
    {0, 1, 3, 2},       // outputChannelToSlot: keeps output channel 3 as the top mic
    true,               // useSafeDriveStrength
};

// --- Optional GPS/PPS (M10Q style) ---
static constexpr bool kEnableGpsUart = false;
static constexpr uint32_t kGpsUartBaud = 9600;
static constexpr int kGpsTxPin = 12;
static constexpr int kGpsRxPin = 13;
static constexpr int kGpsPpsPin = 10;

// --- Optional compass bus wiring (LIS2MDLTR) ---
static constexpr bool kEnableCompassAutoOrientation = false;
static constexpr int kI2cSclPin = 19;
static constexpr int kI2cSdaPin = 18;
static constexpr uint8_t kCompassI2cAddress7Bit = 0x1E;
static constexpr uint8_t kCompassOutputDataRateBits = 0;  // 0:10Hz, 1:20Hz, 2:50Hz, 3:100Hz
static constexpr uint32_t kCompassSampleIntervalMs = 500;
static constexpr float kCompassSmoothingAlpha = 0.03f;
static constexpr float kCompassHeadingOffsetDeg = 0.0f;
static constexpr float kCompassMinHorizontalFieldLsb = 50.0f;
static constexpr uint16_t kCompassStableSamplesRequired = 18;

// --- Time sync ---
// For standalone arrays, strict absolute UTC is optional; GPS/NTP can be enabled later.
static constexpr bool kEnableNtpSync = false;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

}  // namespace nodecfg
