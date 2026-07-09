#pragma once

#include <cstdint>

#include "mmpr/Types.h"

namespace nodecfg {

// ============================================================================
// Frequently edited settings
// ============================================================================

// --- Audio input mode ---
// Selects which capture path the board uses; see AudioInputMode values below.
enum class AudioInputMode : uint8_t {
  kTdm4Mic = 0,     // 4-mic TDM array (ADAU7112)
  kI2sMono = 1,     // single-channel I2S mic
  kPdmDirect = 2,   // reserved for future PDM hardware, not implemented
  kSynthetic = 3,   // no hardware, synthetic/simulated audio
};

#ifndef MMPR_NODECFG_AUDIO_INPUT_MODE
#define MMPR_NODECFG_AUDIO_INPUT_MODE 0  // 0=TDM 4-mic, 1=I2S mono, 2=reserved, 3=synthetic
#endif

// --- Ingest/backend server ---
// Full base URL including scheme, host, and port. This must point at the
// ingest API, which defaults to running on port 8081
#ifndef MMPR_NODECFG_SERVER_BASE_URL
#define MMPR_NODECFG_SERVER_BASE_URL "http://192.168.8.165:8081"
#endif

static constexpr const char* kWifiSsid = "catlin";
static constexpr const char* kWifiPassword = "TEST";

// --- GPS UART baud rate ---
// 9600 for the SparkFun GPS carrier board, 38400 for a bare ublox NEO-M10Q.
// A mismatch here is a common cause of GPS_MISSING status.
static constexpr uint32_t kGpsUartBaud = 9600;

// ============================================================================
// Audio input — remaining tunables
// ============================================================================

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

// frequency is more radio limited now than memory.
// if losing packets, 32 khz, or perhaps 16 khz, would be a useful test for that condition
#ifndef MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ
#define MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ 44100
#endif

#ifndef MMPR_NODECFG_AUDIO_CHANNELS
#define MMPR_NODECFG_AUDIO_CHANNELS 4
#endif

#ifndef MMPR_NODECFG_AUDIO_FRAME_SAMPLES
#define MMPR_NODECFG_AUDIO_FRAME_SAMPLES 512
#endif

// kAudioRingFrames, kAudioQueueSlots, and kPublishBatchFrames are derived at
// compile time from the audio data rate (see "Dynamic buffer sizing" below).
// Define MMPR_NODECFG_AUDIO_RING_FRAMES / _AUDIO_QUEUE_SLOTS /
// _PUBLISH_BATCH_FRAMES (e.g. via -D) only to override the derived defaults.

#ifndef MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET 20480
#endif

#ifndef MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET 0
#endif

#ifndef MMPR_ENABLE_BLE_SCAN
#define MMPR_ENABLE_BLE_SCAN 0
#endif

// BLE scan timing. Units are 0.625 ms each. The scanner is PASSIVE
// it only *listens*, and BLE therefore costs shared-radio RECEIVE time, never a
// TX collision on the audio uplink. Duty = window/interval is the airtime the
// CYW43439 steals from Wi-Fi; keep it low so 44.1 kHz audio is not starved.
// Window must exceed BLE's 0-10 ms advertising jitter to catch reliably, hence >= ~20 ms.
// intermittently -- fine for approximate, slow-updating RSSI triangulation).
// Raise the window (e.g. 0x0030) for snappier coverage at higher airtime cost,
// or lengthen the interval (e.g. 0x0800 = 1280 ms) for an even lighter footprint.
#ifndef MMPR_NODECFG_BLE_SCAN_INTERVAL_UNITS
#define MMPR_NODECFG_BLE_SCAN_INTERVAL_UNITS 0x0400
#endif

#ifndef MMPR_NODECFG_BLE_SCAN_WINDOW_UNITS
#define MMPR_NODECFG_BLE_SCAN_WINDOW_UNITS 0x0020
#endif

#ifndef MMPR_NODECFG_BLE_REPORT_INTERVAL_MS
#define MMPR_NODECFG_BLE_REPORT_INTERVAL_MS 1000
#endif

#ifndef MMPR_NODECFG_BLE_REPORT_MAX_OBSERVATIONS
#define MMPR_NODECFG_BLE_REPORT_MAX_OBSERVATIONS 48
#endif

#ifndef MMPR_NODECFG_BLE_INGEST_PATH
#define MMPR_NODECFG_BLE_INGEST_PATH "/api/v1/ingest/ble"
#endif

static constexpr AudioInputMode kAudioInputMode = (MMPR_NODECFG_AUDIO_INPUT_MODE == 3)
    ? AudioInputMode::kSynthetic
    : ((MMPR_NODECFG_AUDIO_INPUT_MODE == 1) ? AudioInputMode::kI2sMono : AudioInputMode::kTdm4Mic);
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
static constexpr bool kUseSyntheticAudio = kAudioInputMode == AudioInputMode::kSynthetic;

// ============================================================================
// Dynamic buffer sizing
// ============================================================================
// The DMA ring, publish queue, publish batch, and HTTP timeout are derived from
// the audio data rate (channels x sample rate x bytes) so a higher rate or more
// mics automatically provisions more buffering -- while always fitting under a
// fixed RAM ceiling on the Pico 2W (RP2350, 520 KiB SRAM; audio budget capped
// well below that to leave room for the CYW43/lwIP heap and stacks).

namespace detail {
constexpr uint32_t ceilDivU32(uint64_t num, uint64_t den) {
  return den == 0 ? 0u : static_cast<uint32_t>((num + den - 1ull) / den);
}
constexpr uint32_t clampU32(uint32_t value, uint32_t lo, uint32_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}
constexpr size_t clampSize(size_t value, size_t lo, size_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}
}

// --- Intent knobs (primary tunables) ---
static constexpr uint32_t kTargetRingMs = 200;             // capture-side DMA absorb window
static constexpr uint32_t kTargetQueueMs = 300;            // publish-side jitter buffer
static constexpr size_t kTargetPublishBytes = 28u * 1024u; // amortize HTTP overhead per round-trip
// Hard ceiling for ring + queue + active batch on the RP2350's 520 KiB SRAM.
// Reduced when BLE is enabled: the CYW43439 is a shared Wi-Fi/BLE part and
// enabling BLE pulls btstack + HCI buffers into RP2350 SRAM, so audio must leave
// heapFreeBytes telemetry is not currently populated, so an over-commit surfaces
// only as a bad_alloc -> watchdog reboot, not a graceful signal.
static constexpr size_t kAudioBufferRamCeiling =
    (MMPR_ENABLE_BLE_SCAN == 1 ? 240u : 300u) * 1024u;
static constexpr uint32_t kAssumedFloorKbitPerSec = 1500;  // conservative CYW43 goodput, for timeout sizing

// --- Raw data-rate inputs (both TDM and I2S-mono capture at this rate/frame) ---
static constexpr uint32_t kDerivSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kDerivFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
static constexpr uint32_t kDerivChannels =
    kUseTdmAudio ? 4u
                 : (kUseSyntheticAudio ? static_cast<uint32_t>(MMPR_NODECFG_AUDIO_CHANNELS) : 1u);

static constexpr size_t kDerivPacketSlotBytes =
    static_cast<size_t>(kDerivFrameSamples) * kDerivChannels * sizeof(int16_t);
static constexpr size_t kDerivRingFrameBytes =
    static_cast<size_t>(kDerivFrameSamples) * kDerivChannels * sizeof(uint32_t);

// Ring frames: capture-side absorb window, clamped to the PIO ring limit [8, 32].
static constexpr uint32_t kAudioRingFramesAuto = detail::clampU32(
    detail::ceilDivU32(static_cast<uint64_t>(kDerivSampleRateHz) * kTargetRingMs,
                       1000ull * kDerivFrameSamples),
    8u, 32u);

// Publish batch frames: enough packets to reach the target payload, clamp [1, 16].
static constexpr size_t kPublishBatchFramesAuto = detail::clampSize(
    kDerivPacketSlotBytes > 0 ? (kTargetPublishBytes / kDerivPacketSlotBytes) : 1u,
    1u, 16u);

// Queue slots: publish-side jitter buffer sized by time, then shrunk to fit the
// RAM ceiling after the ring (capture safety) and active batch are reserved.
static constexpr uint32_t kAudioQueueSlotsWant = detail::clampU32(
    detail::ceilDivU32(static_cast<uint64_t>(kDerivSampleRateHz) * kTargetQueueMs,
                       1000ull * kDerivFrameSamples),
    8u, 96u);
static constexpr size_t kDerivRingBytes =
    static_cast<size_t>(kAudioRingFramesAuto) * kDerivRingFrameBytes;
static constexpr size_t kDerivActiveBatchBytes = kPublishBatchFramesAuto * kDerivPacketSlotBytes;
static constexpr size_t kDerivQueueBudgetBytes =
    kAudioBufferRamCeiling > (kDerivRingBytes + kDerivActiveBatchBytes)
        ? (kAudioBufferRamCeiling - kDerivRingBytes - kDerivActiveBatchBytes)
        : kDerivPacketSlotBytes;
static constexpr uint32_t kAudioQueueSlotsFit = kDerivPacketSlotBytes > 0
    ? static_cast<uint32_t>(kDerivQueueBudgetBytes / kDerivPacketSlotBytes)
    : kAudioQueueSlotsWant;
static constexpr uint32_t kAudioQueueSlotsAuto = detail::clampU32(
    kAudioQueueSlotsWant < kAudioQueueSlotsFit ? kAudioQueueSlotsWant : kAudioQueueSlotsFit,
    4u, 96u);

// HTTP timeout: time to push one batch at the assumed floor goodput, doubled for
// margin plus a fixed setup allowance, clamped [150, 850] ms. Kept short so a
// stalled publish is abandoned quickly and the next fresh batch goes out.
static constexpr uint32_t kHttpTimeoutSendMs = static_cast<uint32_t>(
    (static_cast<uint64_t>(kPublishBatchFramesAuto) * kDerivPacketSlotBytes * 8ull) /
    (kAssumedFloorKbitPerSec == 0 ? 1u : kAssumedFloorKbitPerSec));
static constexpr uint32_t kHttpTimeoutMsAuto =
    detail::clampU32(kHttpTimeoutSendMs * 2u + 120u, 150u, 850u);

// Final values: derived by default, overridable via -D for bring-up/bench work.
#ifdef MMPR_NODECFG_AUDIO_RING_FRAMES
static constexpr uint32_t kAudioRingFrames = MMPR_NODECFG_AUDIO_RING_FRAMES;
#else
static constexpr uint32_t kAudioRingFrames = kAudioRingFramesAuto;
#endif
#ifdef MMPR_NODECFG_AUDIO_QUEUE_SLOTS
static constexpr size_t kAudioQueueSlots = MMPR_NODECFG_AUDIO_QUEUE_SLOTS;
#else
static constexpr size_t kAudioQueueSlots = kAudioQueueSlotsAuto;
#endif
#ifdef MMPR_NODECFG_PUBLISH_BATCH_FRAMES
static constexpr size_t kPublishBatchFrames = MMPR_NODECFG_PUBLISH_BATCH_FRAMES;
#else
static constexpr size_t kPublishBatchFrames = kPublishBatchFramesAuto;
#endif
#ifdef MMPR_NODECFG_HTTP_TIMEOUT_MS
static constexpr uint32_t kHttpTimeoutMs = MMPR_NODECFG_HTTP_TIMEOUT_MS;
#else
static constexpr uint32_t kHttpTimeoutMs = kHttpTimeoutMsAuto;
#endif

// Publish retry policy: after this many retries of the SAME batch fail, discard
// it so fresh audio flows (losing a stuck batch beats falling permanently behind).
static constexpr uint32_t kPublishBatchMaxRetries = 1;

// --- Network and backend ---
static constexpr const char* kServerBaseUrl = MMPR_NODECFG_SERVER_BASE_URL;
static constexpr bool kEnableBleScan = MMPR_ENABLE_BLE_SCAN == 1;
static constexpr uint16_t kBleScanIntervalUnits = MMPR_NODECFG_BLE_SCAN_INTERVAL_UNITS;
static constexpr uint16_t kBleScanWindowUnits = MMPR_NODECFG_BLE_SCAN_WINDOW_UNITS;
static constexpr uint32_t kBleReportIntervalMs = MMPR_NODECFG_BLE_REPORT_INTERVAL_MS;
static constexpr size_t kBleReportMaxObservations = MMPR_NODECFG_BLE_REPORT_MAX_OBSERVATIONS;
static constexpr const char* kBleIngestPath = MMPR_NODECFG_BLE_INGEST_PATH;


static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
// kHttpTimeoutMs is derived above (see "Dynamic buffer sizing"). It bounds the
// async store-forward request; HttpFramePublisher::pollPublish is non-blocking
// and polled from the main loop, so a slow publish does not steal time from
// drainAvailableAudioFrames()/capture. Kept short so a stalled publish is
// abandoned quickly rather than backing up the capture queue.
// After a failed publish, skip network attempts briefly so capture and Wi-Fi polling recover.
static constexpr uint32_t kPublishFailureBackoffMs = 0;
// ingest control
static constexpr const char* kIngestPath = "/api/v1/ingest/binary";
// kAudioQueueSlots and kPublishBatchFrames are derived above.
static constexpr size_t kPublishBatchByteBudget = MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET;
static constexpr bool kUsePublishBatchByteBudget = MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET == 1;

// Tiny debug/control listener for reading and chaning the current publish target
// Changes are RAM-only and reset on reboot.
static constexpr bool kEnablePublishTargetControlServer = true;
static constexpr bool kAllowRuntimePublishPortChange = true;
static constexpr uint16_t kPublishTargetControlPort = 8082;
static constexpr const char* kPublishTargetControlPath = "/api/v1/publish-target";

// Max samples per channel in a single published packet. Keep diagnostic HTTP POSTs near one TCP send window
static constexpr size_t kMaxPacketSamplesPerChannel = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;

// --- Node identity ---
// Prefix only — the full node ID is built at runtime by appending the chip's
// unique board ID (last 4 hex digits) so each physical device is distinct.
static constexpr const char* kNodeIdPrefix = "sirith-tetra-";
// Audio mode is expected to be chosen per board/configuration
static constexpr mmpr::NodeType kNodeType = kUseTdmAudio ? mmpr::NodeType::kSirithTetra : mmpr::NodeType::kPoint;
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

// Coarse installation correction retained for existing 120-degree mounts.
// 0 = no correction, 1 = +120 degrees, 2 = +240 degrees.
static constexpr uint8_t kBasePlaneRotationSteps = 0;
// Clockwise installation heading from local-world +X, provisioned per node.
// This is the authoritative orientation when automatic compass orientation is not available
static constexpr float kProvisionedWorldHeadingDeg = 0.0f;

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
// Bare-board validation avoids driving external buses
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
static constexpr uint32_t kLedBlinkFrames = 32;  // Toggle every N frames (~0.5s at 16kHz/1024)

// GP27 I2C activity LED.
// Hardware PWM dims average current (kActivityLedDimPercent% duty = ~2.6 mA at 33 mA full).
// Each I2C bus read triggers a brief full-brightness pulse; PWM resets to dim after
static constexpr uint8_t kActivityLedPin = 27;
static constexpr uint8_t kActivityLedDimPercent = 2;  // 2/100 duty => <1.0 mA
static constexpr uint32_t kActivityLedPulseMs = 40;   // visible flash duration

// --- Audio capture (TDM master) ---
static constexpr uint8_t kTdmDataPin = 7;  // SDATA input
static constexpr uint8_t kTdmBclkPin = 8;  // BCLK output
static constexpr uint8_t kTdmWsPin = 9;    // FSYNC/WS output (must be BCLK + 1)

static constexpr uint32_t kAudioSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kAudioFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
// kAudioRingFrames is derived from the data rate (see "Dynamic buffer sizing").
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
static constexpr uint32_t kI2sMonoSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kI2sMonoFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
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
static constexpr uint8_t kActiveAudioChannels = kUseTdmAudio
    ? 4u
    : (kUseSyntheticAudio ? static_cast<uint8_t>(MMPR_NODECFG_AUDIO_CHANNELS) : 1u);

// --- Optional GPS/PPS (M10Q style) ---
static constexpr bool kEnableGpsUart = true;
// we might want a "PSS active but NMEA failing" GPS status to help with this
static constexpr uint32_t kGpsBaudScanIntervalMs = 3000;
// Pin names are MCU-relative: module TX must connect to kGpsRxPin (GP13) for
// incoming NMEA; kGpsTxPin (GP12) is only MCU TX to the module RX.
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

// ============================================================================
// Validation (static_asserts)
// ============================================================================

static_assert(
    MMPR_NODECFG_AUDIO_INPUT_MODE == 0 || MMPR_NODECFG_AUDIO_INPUT_MODE == 1 ||
        MMPR_NODECFG_AUDIO_INPUT_MODE == 2 || MMPR_NODECFG_AUDIO_INPUT_MODE == 3,
    "MMPR_NODECFG_AUDIO_INPUT_MODE must be 0 (TDM), 1 (I2S mono), 2 (reserved PDM), or 3 (synthetic)");
static_assert(
    MMPR_NODECFG_AUDIO_INPUT_MODE != 2,
    "MMPR_NODECFG_AUDIO_INPUT_MODE=2 reserves future PDM_DIRECT hardware and is not implemented yet");
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
static_assert(kAudioSampleRateHz >= 12000 && kAudioSampleRateHz <= 96000, "audio sample rate must be 12-96 kHz");
static_assert(kAudioFrameSamples > 0 && kAudioFrameSamples <= 4096, "audio frame samples out of supported range");
static_assert(kAudioRingFrames > 0 && kAudioRingFrames <= 32, "audio ring frames must be 1-32");
static_assert(kAudioQueueSlots > 0 && kAudioQueueSlots <= 96, "audio queue slots must be 1-96");
static_assert(kPublishBatchFrames > 0 && kPublishBatchFrames <= 16, "publish batch frames must be 1-16");
static_assert(kPublishBatchByteBudget >= 4096 && kPublishBatchByteBudget <= 32768, "publish byte budget must be 4-32 KiB");
// Ring (32-bit DMA words) + publish queue (int16 packets) + active publish batch
// must fit under the RAM ceiling. Derivation guarantees this; the assert also
// guards any manual -D override that would blow the budget.
static_assert(
    (static_cast<size_t>(kAudioFrameSamples) * static_cast<size_t>(kActiveAudioChannels) * sizeof(uint32_t) *
         static_cast<size_t>(kAudioRingFrames)) +
        (static_cast<size_t>(kMaxPacketSamplesPerChannel) * static_cast<size_t>(kActiveAudioChannels) *
         sizeof(int16_t) *
         (static_cast<size_t>(kAudioQueueSlots) +
          (kPublishBatchFrames < static_cast<size_t>(kAudioQueueSlots)
               ? kPublishBatchFrames
               : static_cast<size_t>(kAudioQueueSlots)))) <= kAudioBufferRamCeiling,
    "audio ring + queue + active batch budget exceeds kAudioBufferRamCeiling");

}  // namespace nodecfg
