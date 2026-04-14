#pragma once

#include <cstdint>

namespace nodecfg {

// --- Network and backend ---
static constexpr const char* kWifiSsid = "REPLACE_WIFI_SSID";
static constexpr const char* kWifiPassword = "REPLACE_WIFI_PASSWORD";
static constexpr const char* kServerBaseUrl = "http://192.168.1.50:8080";
static constexpr const char* kIngestPath = "/api/v1/ingest/frame";

static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
static constexpr uint32_t kHttpTimeoutMs = 2500;

// --- Node identity ---
static constexpr const char* kNodeId = "sirith-tetra-pico2w-01";
static constexpr const char* kNodeTypeWire = "sirith_tetra";
static constexpr float kNodePositionM[3] = {6.0f, 0.0f, 2.0f};

// Physical microphone geometry — centroid-centred, metres.
// Positions derived from the LIS2MDLTR / LSM6DSV16X sensor frame:
//   MK1 = (0, 50, 0) mm,  MK2 = (43.3, 25, 0) mm,
//   MK3 = (0, 0, 0) mm,   MK4 = (21.65, 25, 40.82) mm.
// Centroid = (16.2375, 25.0, 10.205) mm.
//
// Mic indices: 0 = MK1, 1 = MK2, 2 = MK3 (base), 3 = MK4 (top).
// Base-mic bearings from centroid in sensor frame:
//   MK2 ≈ 0°,  MK1 ≈ 120°,  MK3 ≈ 240°.
static constexpr float kPhysicalSensorOffsetsM[4][3] = {
    {-0.016238f,  0.025000f, -0.010205f},   // MK1
    { 0.027063f,  0.000000f, -0.010205f},   // MK2
    {-0.016238f, -0.025000f, -0.010205f},   // MK3
    { 0.005413f,  0.000000f,  0.030615f},   // MK4 (top)
};

// For calibration: rotate only the base plane indexing while keeping MK4 fixed.
// 0 = no rotation, 1 = +120 degrees, 2 = +240 degrees.
static constexpr uint8_t kBasePlaneRotationSteps = 0;

// TDM slot mapping from ADAU7112 strap configuration.
// Per schematic I2S / TDM channel assignments:
//   TDM1 (slot 0) = MK2 = mic 1   (DATA1-Left,  i2s1)
//   TDM2 (slot 1) = MK1 = mic 0   (DATA2-Right, i2s1)
//   TDM3 (slot 2) = MK4 = mic 3   (DATA1-Left,  i2s2)
//   TDM4 (slot 3) = MK3 = mic 2   (DATA2-Right, i2s2)
static constexpr uint8_t kSlotToPhysicalMic[4] = {1, 0, 3, 2};

static constexpr const char* kCapabilities[] = {
    "audio",
    "array_localization",
    "gps_optional",
    "temperature",
};

static constexpr const char* kHardwareName = "sirith_tetra_pico2w_tdm";

// --- Bring-up mode ---
// Bare-board validation keeps the switched 3V3 rail off and avoids driving
// any external buses so firmware can be verified without risking the mic array.
static constexpr bool kBareBoardValidationMode = true;
static constexpr bool kEnableExternal3v3Rail = !kBareBoardValidationMode;
static constexpr bool kEnableExternalAudioHardware = !kBareBoardValidationMode;
static constexpr bool kEnableExternalPeripheralBuses = !kBareBoardValidationMode;

// --- LED / status indicator ---
// P-channel FET on GP26 controls LED + switchable 3V3 power header.
// LOW = FET on (LED lit), HIGH = FET off (LED dark).
static constexpr uint8_t kLedPin = 26;
static constexpr uint32_t kLedBlinkFrames = 8;  // Toggle every N frames (~0.5s at 16kHz/1024)

// --- Audio capture (TDM master) ---
static constexpr uint8_t kTdmDataPin = 7;  // SDATA input
static constexpr uint8_t kTdmBclkPin = 8;  // BCLK output
static constexpr uint8_t kTdmWsPin = 9;    // FSYNC/WS output (must be BCLK + 1)

static constexpr uint32_t kAudioSampleRateHz = 16000;
static constexpr uint32_t kAudioFrameSamples = 1024;
static constexpr int32_t kAudioSampleShiftBits = 16;  // 32-bit slot -> 16-bit PCM
static constexpr uint8_t kAudioTdmSlots = 4;
static constexpr uint8_t kAudioSlotBits = 32;
// Output channel order: MK1, MK2, MK3, MK4.
static constexpr uint8_t kOutputChannelToSlot[4] = {1, 0, 3, 2};
static constexpr bool kUseSafeDriveStrength = true;

// --- Optional GPS/PPS (M10Q style) ---
static constexpr bool kEnableGpsUart = false;
static constexpr uint32_t kGpsUartBaud = 9600;
static constexpr int kGpsTxPin = 12;
static constexpr int kGpsRxPin = 13;
static constexpr int kGpsPpsPin = 10;

// --- Optional compass and IMU bus wiring ---
// GP18/GP19 map to I2C1 on RP2040/RP2350 pin mux.
static constexpr bool kEnableCompassAutoOrientation = false;
static constexpr bool kEnableImuTemperature = false;
static constexpr int kI2cSdaPin = 18;
static constexpr int kI2cSclPin = 19;
static constexpr uint32_t kI2cBaudHz = 400000;

// LIS2MDLTR magnetometer driver config.
static constexpr uint8_t kCompassI2cAddress7Bit = 0x1E;
static constexpr uint8_t kCompassOdrBits = 2;          // 0:10Hz, 1:20Hz, 2:50Hz, 3:100Hz(LP)
static constexpr bool kCompassEnableTempComp = true;
static constexpr bool kCompassEnableLpf = true;         // HW LPF: bandwidth = ODR/4
static constexpr bool kCompassEnableOffsetCancel = true; // SET/RESET offset cancellation
static constexpr float kCompassHardIronX = 0.0f;  // calibration offsets (LSB)
static constexpr float kCompassHardIronY = 0.0f;
static constexpr float kCompassHardIronZ = 0.0f;

// Auto-orientation estimator config (Kalman-filtered heading).
static constexpr uint32_t kCompassSampleIntervalMs = 200;
static constexpr float kCompassHeadingOffsetDeg = 0.0f;
static constexpr float kCompassMinFieldMagnitude = 50.0f;  // LSB, reject below this
static constexpr uint16_t kCompassStableSamplesRequired = 18;
static constexpr float kCompassKalmanQ = 0.001f;   // process noise  (deg²/step)
static constexpr float kCompassKalmanR = 4.0f;     // measurement noise (deg²)
static constexpr float kCompassKalmanInitP = 400.0f; // initial covariance (deg²)

// LSM6 family temperature
static constexpr uint8_t kImuI2cAddressPrimary7Bit = 0x6A;
static constexpr uint8_t kImuI2cAddressSecondary7Bit = 0x6B;
static constexpr uint32_t kImuTemperatureSampleIntervalMs = 2000;

// --- Time sync ---
// For standalone arrays, strict absolute UTC is optional; GPS/NTP can be enabled later.
static constexpr bool kEnableNtpSync = false;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

}  // namespace nodecfg
