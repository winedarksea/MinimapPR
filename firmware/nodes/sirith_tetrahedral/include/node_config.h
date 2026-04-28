#pragma once

#include <cstdint>

#include "mmpr/Types.h"

namespace nodecfg {

enum class AudioInputMode : uint8_t {
  kTdm4Mic = 0,
  kI2sMono = 1,
};

enum class I2sMonoChannelSide : uint8_t {
  kLeft = 0,
  kRight = 1,
};

enum class AudioSerialSampleEdge : uint8_t {
  kRising = 0,
  kFalling = 1,
};

enum class AudioDataPinBias : uint8_t {
  kDisabled = 0,
  kPullDown = 1,
};

#ifndef MMPR_NODECFG_AUDIO_INPUT_MODE
#define MMPR_NODECFG_AUDIO_INPUT_MODE 0
#endif

#ifndef MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE
#define MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE 0
#endif

#ifndef MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE
#define MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE 0
#endif

#ifndef MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET
#define MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET 0
#endif

#ifndef MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS
#define MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS 0
#endif

#ifndef MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS
#define MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS 0
#endif

#ifndef MMPR_NODECFG_TDM_SAMPLE_EDGE
#define MMPR_NODECFG_TDM_SAMPLE_EDGE 0
#endif

#ifndef MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET
#define MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET 1
#endif

#ifndef MMPR_NODECFG_TDM_DATA_PIN_BIAS
#define MMPR_NODECFG_TDM_DATA_PIN_BIAS 1
#endif

#ifndef MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS
#define MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS 0
#endif

#ifndef MMPR_NODECFG_GPS_PPS_PIN
#define MMPR_NODECFG_GPS_PPS_PIN 10
#endif

static_assert(
    MMPR_NODECFG_AUDIO_INPUT_MODE == 0 || MMPR_NODECFG_AUDIO_INPUT_MODE == 1,
    "MMPR_NODECFG_AUDIO_INPUT_MODE must be 0 (TDM) or 1 (I2S mono)");
static_assert(
    MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE == 0 || MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE == 1,
    "MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE must be 0 (left) or 1 (right)");
static_assert(
    MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE == 0 || MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE == 1,
    "MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE must be 0 (rising) or 1 (falling)");
static_assert(
    MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET >= -8 && MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET <= 8,
    "MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET must be in [-8, 8]");
static_assert(
    MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS == 0 || MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS == 1,
    "MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS must be 0 (disabled) or 1 (pull-down)");
static_assert(
    MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS == 0 || MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS == 1,
    "MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS must be 0 or 1");
static_assert(
    MMPR_NODECFG_TDM_SAMPLE_EDGE == 0 || MMPR_NODECFG_TDM_SAMPLE_EDGE == 1,
    "MMPR_NODECFG_TDM_SAMPLE_EDGE must be 0 (rising) or 1 (falling)");
static_assert(
    MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET >= -8 && MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET <= 8,
    "MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET must be in [-8, 8]");
static_assert(
    MMPR_NODECFG_TDM_DATA_PIN_BIAS == 0 || MMPR_NODECFG_TDM_DATA_PIN_BIAS == 1,
    "MMPR_NODECFG_TDM_DATA_PIN_BIAS must be 0 (disabled) or 1 (pull-down)");
static_assert(
    MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS == 0 || MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS == 1,
    "MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS must be 0 or 1");

static constexpr AudioInputMode kAudioInputMode = (MMPR_NODECFG_AUDIO_INPUT_MODE == 1)
    ? AudioInputMode::kI2sMono
    : AudioInputMode::kTdm4Mic;
static constexpr I2sMonoChannelSide kI2sMonoChannelSide = (MMPR_NODECFG_I2S_MONO_CHANNEL_SIDE == 1)
    ? I2sMonoChannelSide::kRight
    : I2sMonoChannelSide::kLeft;
static constexpr AudioSerialSampleEdge kI2sMonoSampleEdge = (MMPR_NODECFG_I2S_MONO_SAMPLE_EDGE == 1)
    ? AudioSerialSampleEdge::kFalling
    : AudioSerialSampleEdge::kRising;
static constexpr int8_t kI2sMonoCaptureBitOffset = MMPR_NODECFG_I2S_MONO_CAPTURE_BIT_OFFSET;
static constexpr AudioDataPinBias kI2sMonoDataPinBias = (MMPR_NODECFG_I2S_MONO_DATA_PIN_BIAS == 1)
    ? AudioDataPinBias::kPullDown
    : AudioDataPinBias::kDisabled;
static constexpr bool kI2sMonoEnableWordDiagnostics = MMPR_NODECFG_I2S_MONO_ENABLE_WORD_DIAGNOSTICS == 1;
static constexpr AudioSerialSampleEdge kTdmSampleEdge = (MMPR_NODECFG_TDM_SAMPLE_EDGE == 1)
    ? AudioSerialSampleEdge::kFalling
    : AudioSerialSampleEdge::kRising;
static constexpr int8_t kTdmCaptureBitOffset = MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET;
static constexpr AudioDataPinBias kTdmDataPinBias = (MMPR_NODECFG_TDM_DATA_PIN_BIAS == 1)
    ? AudioDataPinBias::kPullDown
    : AudioDataPinBias::kDisabled;
static constexpr bool kTdmEnableWordDiagnostics = MMPR_NODECFG_TDM_ENABLE_WORD_DIAGNOSTICS == 1;
static constexpr bool kUseTdmAudio = kAudioInputMode == AudioInputMode::kTdm4Mic;

// --- Network and backend ---
static constexpr const char* kWifiSsid = "REPLACE_WIFI_SSID";
static constexpr const char* kWifiPassword = "REPLACE_WIFI_PASSWORD";
static constexpr const char* kServerBaseUrl = "http://192.168.1.50:8080";
static constexpr const char* kIngestPath = "/api/v1/ingest/frame";

static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
// Keep this below the DMA ring slack. At 16 kHz / 1024 samples / 16 buffered
// blocks, the capture ring holds about 1.0 s; longer synchronous publishes
// overrun capture and make the heartbeat look randomly slow.
static constexpr uint32_t kHttpTimeoutMs = 450;
// After a failed publish, skip network attempts briefly so capture and Wi-Fi
// polling recover. Keep this short: a multi-second backoff discards dozens of
// audio packets and makes the debug stream sparse even after transient stalls.
static constexpr uint32_t kPublishFailureBackoffMs = 0;
// ingest control
static constexpr const char* kIngestPath = "/api/v1/ingest/binary";
static constexpr size_t kStoreForwardBatchFrames = 4;

// Max samples per channel in a single published packet. Keep diagnostic HTTP
// POSTs near one TCP send window so a timeout does not leave body bytes that
// uvicorn parses as a malformed follow-up request.
static constexpr size_t kMaxPacketSamplesPerChannel = 512;

// --- Node identity ---
// Prefix only — the full node ID is built at runtime by appending the chip's
// unique board ID (last 4 hex digits) so each physical device is distinct.
static constexpr const char* kNodeIdPrefix = "sirith-tetra-";
// Audio mode is expected to be chosen per board/configuration, not switched on
// one deployed node identity at runtime.
static constexpr mmpr::NodeType kNodeType = kUseTdmAudio ? mmpr::NodeType::kSirithTetra : mmpr::NodeType::kPoint;
static constexpr mmpr::Vec3 kNodePositionM = {6.0f, 0.0f, 2.0f};
static constexpr bool kNodeHasFallbackGeoPosition = true;
static constexpr float kNodeFallbackLatitudeDeg = 44.98698840878797f;
static constexpr float kNodeFallbackLongitudeDeg = -93.2579197515542f;
static constexpr float kNodeFallbackAltitudeM = 0.0f;

// Physical microphone geometry — centroid-centred, metres.
// Positions derived from the LIS2MDLTR / LSM6DSV16X sensor frame:
//   MK1 = (0, 50, 0) mm,  MK2 = (43.3, 25, 0) mm,
//   MK3 = (0, 0, 0) mm,   MK4 = (21.65, 25, 40.82) mm.
// Centroid = (16.2375, 25.0, 10.205) mm.
//
// Mic indices: 0 = MK1, 1 = MK2, 2 = MK3 (base), 3 = MK4 (top).
// Base-mic bearings from centroid in sensor frame:
//   MK2 ≈ 0°,  MK1 ≈ 120°,  MK3 ≈ 240°.
static constexpr mmpr::Vec3 kPhysicalSensorOffsetsM[4] = {
    {-0.016238f,  0.025000f, -0.010205f},  // MK1
    { 0.027063f,  0.000000f, -0.010205f},  // MK2
    {-0.016238f, -0.025000f, -0.010205f},  // MK3
    { 0.005413f,  0.000000f,  0.030615f},  // MK4 (top)
};

static constexpr mmpr::Vec3 kPointSensorOffsetsM[1] = {
    {0.0f, 0.0f, 0.0f},
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

static constexpr const char* kTdmCapabilities[] = {
    "audio",
    "array_localization",
    "gps_optional",
    "temperature",
    "humidity",
};

static constexpr const char* kI2sMonoCapabilities[] = {
    "audio",
    "gps_optional",
    "temperature",
    "humidity",
};

static constexpr const char* const* kCapabilities = kUseTdmAudio ? kTdmCapabilities : kI2sMonoCapabilities;
static constexpr size_t kCapabilityCount = kUseTdmAudio
    ? (sizeof(kTdmCapabilities) / sizeof(kTdmCapabilities[0]))
    : (sizeof(kI2sMonoCapabilities) / sizeof(kI2sMonoCapabilities[0]));

static constexpr const char* kHardwareName = kUseTdmAudio
    ? "sirith_tetra_pico2w_tdm"
    : "sirith_tetra_pico2w_i2s_mono";
static constexpr const mmpr::Vec3* kSensorOffsetsM = kUseTdmAudio ? kPhysicalSensorOffsetsM : kPointSensorOffsetsM;
static constexpr size_t kSensorOffsetCount = kUseTdmAudio
    ? (sizeof(kPhysicalSensorOffsetsM) / sizeof(kPhysicalSensorOffsetsM[0]))
    : (sizeof(kPointSensorOffsetsM) / sizeof(kPointSensorOffsetsM[0]));

// --- Bring-up mode ---
// Bare-board validation avoids driving external buses so firmware can be
// verified without risking the mic array.
static constexpr bool kBareBoardValidationMode = false;
// GP26 controls a switched Vin rail through a FET and also drives an board LED.
// This rail does not power anything right now, and is optional
static constexpr bool kEnableExternalVFetRail = false;
static constexpr bool kEnableExternalAudioHardware = !kBareBoardValidationMode;
static constexpr bool kEnableExternalPeripheralBuses = !kBareBoardValidationMode;

// --- LED / status indicator ---
// P-channel FET on GP26 controls LED + switchable 3V3 power header.
// LOW = FET on (LED lit), HIGH = FET off (LED dark).
static constexpr uint8_t kLedPin = 26;
static constexpr uint32_t kLedBlinkFrames = 8;  // Toggle every N frames (~0.5s at 16kHz/1024)

// GP27 I2C activity LED.
// Hardware PWM dims average current (kActivityLedDimPercent% duty = ~2.6 mA at 33 mA full).
// Each I2C bus read triggers a brief full-brightness pulse; PWM resets to dim after
// kActivityLedPulseMs.  The PWM peripheral runs autonomously — no CPU cost at steady state.
static constexpr uint8_t kActivityLedPin = 27;
static constexpr uint8_t kActivityLedDimPercent = 2;  // 2/100 duty => <1.0 mA
static constexpr uint32_t kActivityLedPulseMs = 80;   // visible flash duration

// --- Audio capture (TDM master) ---
static constexpr uint8_t kTdmDataPin = 7;  // SDATA input
static constexpr uint8_t kTdmBclkPin = 8;  // BCLK output
static constexpr uint8_t kTdmWsPin = 9;    // FSYNC/WS output (must be BCLK + 1)

static constexpr uint32_t kAudioSampleRateHz = 16000;
static constexpr uint32_t kAudioFrameSamples = 512;
static constexpr uint8_t kAudioValidBits = 24;  // ADAU7112 emits 24-bit PCM in 32-bit slots.
static constexpr uint8_t kAudioTdmSlots = 4;
static constexpr uint8_t kAudioSlotBits = 32;
// ADAU7112 TDM output typically has a 1-bit delay (I2S-style TDM). The first BCLK is High-Z,
// followed by 24 data bits (MSB first). To align the 24-bit payload's MSB to the sign bit (bit 31)
// of the 32-bit PIO word, a 1-bit left shift is required rather than a right shift.
static constexpr AudioSerialSampleEdge kAudioTdmSampleEdge = kTdmSampleEdge;
static constexpr int8_t kAudioTdmCaptureBitOffset = kTdmCaptureBitOffset;
static constexpr AudioDataPinBias kAudioTdmDataPinBias = kTdmDataPinBias;
static constexpr bool kAudioTdmEnableWordDiagnostics = kTdmEnableWordDiagnostics;
// Output channel order: MK1, MK2, MK3, MK4.
static constexpr uint8_t kOutputChannelToSlot[4] = {1, 0, 3, 2};
static constexpr bool kUseSafeDriveStrength = true;

// --- Audio capture (mono I2S master) ---
// Mono-I2S pin routing is board-specific. These defaults intentionally mirror
// the current audio bring-up wiring so the firmware compiles, but they do not
// guarantee a true separate I2S lane on this PCB. Update them per board when
// enabling kI2sMono so the selected lane matches the intended microphone path.
static constexpr uint8_t kI2sMonoDataPin = kTdmDataPin;
static constexpr uint8_t kI2sMonoBclkPin = kTdmBclkPin;
static constexpr uint8_t kI2sMonoWsPin = kTdmWsPin;
static constexpr uint32_t kI2sMonoSampleRateHz = 16000;
static constexpr uint32_t kI2sMonoFrameSamples = 1024;
static constexpr uint8_t kI2sMonoSlotBits = 32;
static constexpr uint8_t kI2sMonoValidBits = 24;
// ICS-43434 uses standard I2S timing with the MSB delayed by one BCLK after
// LRCLK changes. It tri-states SD outside the active slot, so keep a weak
// pulldown available unless the board already provides one.
static constexpr AudioSerialSampleEdge kAudioI2sMonoSampleEdge = kI2sMonoSampleEdge;
static constexpr int8_t kAudioI2sMonoCaptureBitOffset = kI2sMonoCaptureBitOffset;
static constexpr AudioDataPinBias kAudioI2sMonoDataPinBias = kI2sMonoDataPinBias;
static constexpr bool kAudioI2sMonoEnableWordDiagnostics = kI2sMonoEnableWordDiagnostics;
static constexpr bool kI2sMonoPinsAliasTdmPins =
    (kI2sMonoDataPin == kTdmDataPin) &&
    (kI2sMonoBclkPin == kTdmBclkPin) &&
    (kI2sMonoWsPin == kTdmWsPin);

static constexpr uint32_t kActiveAudioSampleRateHz = kUseTdmAudio ? kAudioSampleRateHz : kI2sMonoSampleRateHz;
static constexpr uint32_t kActiveAudioFrameSamples = kUseTdmAudio ? kAudioFrameSamples : kI2sMonoFrameSamples;
static constexpr uint8_t kActiveAudioChannels = kUseTdmAudio ? 4u : 1u;

// --- Optional GPS/PPS (M10Q style) ---
static constexpr bool kEnableGpsUart = true;
static constexpr uint32_t kGpsUartBaud = 9600;
static constexpr int kGpsTxPin = 12;
static constexpr int kGpsRxPin = 13;
// Configures the PPS-capable GPIO. Runtime clock discipline still tolerates no
// PPS at boot, stale PPS during a run, and later PPS recovery.
static constexpr int kGpsPpsPin = MMPR_NODECFG_GPS_PPS_PIN;
static constexpr uint32_t kGpsMissingSentenceTimeoutMs = 5000;
static constexpr uint32_t kGpsStaleFixTimeoutMs = 5000;
static constexpr const char* kGpsSignalStatus = "missing";
static constexpr const char* kGpsPositionSource = "fallback_static";

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

// Sensirion SHT4x family environmental sensor (SHT45 on SparkFun SAM-M10Q carrier)
static constexpr bool kEnableSht45Environment = true;
static constexpr uint8_t kSht4xI2cAddress7Bit = 0x44;
static constexpr uint32_t kSht4xSampleIntervalMs = 2000;

// --- Time sync ---
// For standalone arrays, strict absolute UTC is optional; GPS/NTP can be enabled later.
static constexpr bool kEnableNtpSync = true;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

}  // namespace nodecfg
