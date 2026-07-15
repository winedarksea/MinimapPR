#pragma once

#include <cstdint>

#include "mmpr/Types.h"

// ============================================================================
// sirith_planar node configuration — 5-mic PDM planar array on RP2354A
// ============================================================================
// Center-square geometry: 4 corner mics at a 25 mm radius + 1 center mic,
// coplanar (all z = 0). Captured over PDM (3 data lines, 2+2+1 sharing) and
// decimated on core 1 (see PdmCicDecimator, Phase 2). No CYW43/lwIP/btstack in
// this build — Wi-Fi is offloaded to an ESP32-C5 (Phase 3). This header is only
// included by the node main + CMakeLists; the shared runtime never sees it.

namespace nodecfg {

// --- Audio input mode ---
enum class AudioInputMode : uint8_t {
  kTdm4Mic = 0,     // 4-mic TDM array (ADAU7112) — tetra
  kI2sMono = 1,     // single-channel I2S mic
  kPdmDirect = 2,   // 5-mic PDM planar array (this node)
  kSynthetic = 3,   // no hardware, synthetic/simulated audio
};

// Planar defaults to PDM direct. The skeleton (Phase 1) still boots on synthetic
// audio until PdmPlanarSource lands (Phase 2); flip to 3 for bench bring-up
// without PDM hardware.
#ifndef MMPR_NODECFG_AUDIO_INPUT_MODE
#define MMPR_NODECFG_AUDIO_INPUT_MODE 2  // 2=PDM 5-mic planar, 3=synthetic
#endif

// --- Ingest/backend server ---
#ifndef MMPR_NODECFG_SERVER_BASE_URL
#define MMPR_NODECFG_SERVER_BASE_URL "http://192.168.8.165:8081"
#endif

// --- GPS UART baud rate ---
// Bare u-blox M10Q default is 38400; 9600 for the SparkFun carrier board.
static constexpr uint32_t kGpsUartBaud = 38400;

// ============================================================================
// Audio input — tunables
// ============================================================================

enum class AudioSerialSampleEdge : uint8_t { kRising = 0, kFalling = 1 };
enum class AudioDataPinBias : uint8_t { kDisabled = 0, kPullDown = 1 };

#ifndef MMPR_NODECFG_GPS_PPS_PIN
#define MMPR_NODECFG_GPS_PPS_PIN 10
#endif

// Planar runs full-band: 48 kHz output from the CIC/half-band chain.
#ifndef MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ
#define MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ 48000
#endif

#ifndef MMPR_NODECFG_AUDIO_CHANNELS
#define MMPR_NODECFG_AUDIO_CHANNELS 5
#endif

#ifndef MMPR_NODECFG_AUDIO_FRAME_SAMPLES
#define MMPR_NODECFG_AUDIO_FRAME_SAMPLES 512
#endif

#ifndef MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET 20480
#endif

#ifndef MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET 0
#endif

// No BLE in the planar build (radio is off-board on the ESP32-C5). The nodecfg
// CMake helper greps this value, so it must be present.
#ifndef MMPR_ENABLE_BLE_SCAN
#define MMPR_ENABLE_BLE_SCAN 0
#endif

static constexpr AudioInputMode kAudioInputMode = (MMPR_NODECFG_AUDIO_INPUT_MODE == 3)
    ? AudioInputMode::kSynthetic
    : AudioInputMode::kPdmDirect;
static constexpr bool kUsePdmAudio = kAudioInputMode == AudioInputMode::kPdmDirect;
static constexpr bool kUseSyntheticAudio = kAudioInputMode == AudioInputMode::kSynthetic;

// ============================================================================
// Physical microphone geometry — center-square planar array, metres.
// ============================================================================
// Firmware-defined and configurable (backend consumes sensor_offsets_m; no new
// backend geometry constants). Corners sit on a circle of radius r at 45° steps
// so the square's side is r*sqrt(2); the center mic is at the origin. All z = 0
// (coplanar) — the backend applies the up/down half-space constraint (D7).
static constexpr float kPlanarArrayRadiusM = 0.025f;  // 25 mm corner radius

// r / sqrt(2)
static constexpr float kPlanarCornerXY = kPlanarArrayRadiusM * 0.70710678118654752440f;

// Deinterleave → output channel map (documented for PdmPlanarSource, Phase 2):
//   PDM DATA0 (GP1): rising→ch0, falling→ch1   (two corner mics)
//   PDM DATA1 (GP2): rising→ch2, falling→ch3   (two corner mics)
//   PDM DATA2 (GP3): rising→ch4, falling→discard (center mic)
// Output channel order: ch0..ch3 = corners (NE, NW, SW, SE), ch4 = center.
static constexpr uint8_t kPlanarChannelCount = 5;
static constexpr mmpr::Vec3 kPlanarSensorOffsetsM[kPlanarChannelCount] = {
    { kPlanarCornerXY,  kPlanarCornerXY, 0.0f},  // ch0 corner NE
    {-kPlanarCornerXY,  kPlanarCornerXY, 0.0f},  // ch1 corner NW
    {-kPlanarCornerXY, -kPlanarCornerXY, 0.0f},  // ch2 corner SW
    { kPlanarCornerXY, -kPlanarCornerXY, 0.0f},  // ch3 corner SE
    { 0.0f,            0.0f,            0.0f},    // ch4 center
};

// ============================================================================
// Dynamic buffer sizing (reused from the tetra derivation machinery)
// ============================================================================
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

static constexpr uint32_t kTargetRingMs = 200;
static constexpr uint32_t kTargetQueueMs = 300;
static constexpr size_t kTargetPublishBytes = 28u * 1024u;
// No CYW43/lwIP/btstack heap in this build, so the audio budget can be much
// larger than the tetra ceiling. RP2350 has 520 KiB SRAM; leave headroom for
// stacks, the core-1 DSP scratch/LUTs, and the ESP32-C5 SPI link buffers.
static constexpr size_t kAudioBufferRamCeiling = 352u * 1024u;
static constexpr uint32_t kAssumedFloorKbitPerSec = 15000;  // 5 GHz link via C5

static constexpr uint32_t kDerivSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kDerivFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
static constexpr uint32_t kDerivChannels = kPlanarChannelCount;

static constexpr size_t kDerivPacketSlotBytes =
    static_cast<size_t>(kDerivFrameSamples) * kDerivChannels * sizeof(int16_t);
static constexpr size_t kDerivRingFrameBytes =
    static_cast<size_t>(kDerivFrameSamples) * kDerivChannels * sizeof(uint32_t);

static constexpr uint32_t kAudioRingFramesAuto = detail::clampU32(
    detail::ceilDivU32(static_cast<uint64_t>(kDerivSampleRateHz) * kTargetRingMs,
                       1000ull * kDerivFrameSamples),
    8u, 32u);

static constexpr size_t kPublishBatchFramesAuto = detail::clampSize(
    kDerivPacketSlotBytes > 0 ? (kTargetPublishBytes / kDerivPacketSlotBytes) : 1u,
    1u, 16u);

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

static constexpr uint32_t kHttpTimeoutSendMs = static_cast<uint32_t>(
    (static_cast<uint64_t>(kPublishBatchFramesAuto) * kDerivPacketSlotBytes * 8ull) /
    (kAssumedFloorKbitPerSec == 0 ? 1u : kAssumedFloorKbitPerSec));
static constexpr uint32_t kHttpTimeoutMsAuto =
    detail::clampU32(kHttpTimeoutSendMs * 2u + 120u, 150u, 850u);

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

static constexpr uint32_t kPublishBatchMaxRetries = 1;

// --- Network and backend ---
static constexpr const char* kServerBaseUrl = MMPR_NODECFG_SERVER_BASE_URL;
static constexpr uint32_t kPublishFailureBackoffMs = 0;
static constexpr const char* kIngestPath = "/api/v1/ingest/binary";
static constexpr size_t kPublishBatchByteBudget = MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET;
static constexpr bool kUsePublishBatchByteBudget = MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET == 1;
static constexpr size_t kMaxPacketSamplesPerChannel = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;

// --- Node identity ---
static constexpr const char* kNodeIdPrefix = "sirith-planar-";
static constexpr mmpr::NodeType kNodeType = mmpr::NodeType::kSirithPlanar;
static constexpr bool kNodeHasFallbackGeoPosition = true;
static constexpr float kNodeFallbackLatitudeDeg = 44.98698840878797f;
static constexpr float kNodeFallbackLongitudeDeg = -93.2579197515542f;
static constexpr float kNodeFallbackAltitudeM = 0.0f;

static constexpr const char* kCapabilities[] = {
    "audio",
    "array_localization",
    "planar_half_space",
    "gps_optional",
    "temperature",
    "humidity",
};
static constexpr size_t kCapabilityCount = sizeof(kCapabilities) / sizeof(kCapabilities[0]);

static constexpr const char* kHardwareName = "sirith-planar-5pdm-rp2354a";
static constexpr const mmpr::Vec3* kSensorOffsetsM = kPlanarSensorOffsetsM;
static constexpr size_t kSensorOffsetCount = kPlanarChannelCount;

static constexpr uint32_t kActiveAudioSampleRateHz = kDerivSampleRateHz;
static constexpr uint32_t kActiveAudioFrameSamples = kDerivFrameSamples;
static constexpr uint8_t kActiveAudioChannels = kPlanarChannelCount;

// Installation orientation (world heading, clockwise from +X). Authoritative
// until compass auto-orientation is enabled.
static constexpr float kProvisionedWorldHeadingDeg = 0.0f;

// --- Bring-up mode ---
static constexpr bool kBareBoardValidationMode = false;
static constexpr bool kEnableExternalAudioHardware = !kBareBoardValidationMode;
static constexpr bool kEnableExternalPeripheralBuses = !kBareBoardValidationMode;

// ============================================================================
// Pin map (RP2354A QFN-60) — see HARDWARE_REVIEW.md for the authoritative table.
// ============================================================================
// PSRAM CS reserved (QMI CS1n): GP0 primary / GP8 fallback (unconnected).
static constexpr int kPsramCsPinPrimary = 0;
static constexpr int kPsramCsPinFallback = 8;

// PDM capture: 3 data lines (contiguous PIO `in pins` base) + 1 side-set clock.
static constexpr uint8_t kPdmDataPinBase = 1;   // GP1..GP3 (count 3)
static constexpr uint8_t kPdmDataPinCount = 3;
static constexpr uint8_t kPdmClockPin = 4;      // GP4
// E9: never use the internal pull-down on PDM inputs — external ~10k pull-downs
// on GP1/GP2/GP3 define the float behaviour of the shared lines.
static constexpr AudioDataPinBias kPdmDataPinBias = AudioDataPinBias::kDisabled;

// Buzzer (piezo driver, PWM) — reserved; stub firmware (Phase 6).
static constexpr uint8_t kBuzzerPwmPin = 6;

// LEDs (existing patterns).
static constexpr uint8_t kLedPin = 26;
static constexpr uint32_t kLedBlinkFrames = 48;
static constexpr uint8_t kActivityLedPin = 27;
static constexpr uint8_t kActivityLedDimPercent = 2;
static constexpr uint32_t kActivityLedPulseMs = 40;

// GPS (M10Q) — same pins as tetra.
static constexpr bool kEnableGpsUart = true;
static constexpr uint32_t kGpsBaudScanIntervalMs = 3000;
static constexpr int kGpsTxPin = 12;
static constexpr int kGpsRxPin = 13;
static constexpr int kGpsPpsPin = MMPR_NODECFG_GPS_PPS_PIN;
static constexpr uint32_t kGpsMissingSentenceTimeoutMs = 5000;
static constexpr uint32_t kGpsStaleFixTimeoutMs = 5000;
static constexpr const char* kGpsSignalStatus = "missing";
static constexpr const char* kGpsPositionSource = "fallback_static";

// I2C1 (SHT45 + IMU/mag). Phase 6: these drivers already exist (shared with
// tetra) and are enabled here; heading is measured/logged but not yet used to
// rotate the planar array geometry (kProvisionedWorldHeadingDeg stays
// authoritative — see comment above).
static constexpr bool kEnableCompassAutoOrientation = true;
static constexpr bool kEnableImuTemperature = true;
static constexpr int kI2cSdaPin = 18;
static constexpr int kI2cSclPin = 19;
static constexpr uint32_t kI2cBaudHz = 400000;

static constexpr uint8_t kCompassI2cAddress7Bit = 0x1E;
static constexpr uint8_t kCompassOdrBits = 2;
static constexpr bool kCompassEnableTempComp = true;
static constexpr bool kCompassEnableLpf = true;
static constexpr bool kCompassEnableOffsetCancel = true;
static constexpr float kCompassHardIronX = 0.0f;
static constexpr float kCompassHardIronY = 0.0f;
static constexpr float kCompassHardIronZ = 0.0f;
static constexpr uint32_t kCompassSampleIntervalMs = 200;
static constexpr float kCompassHeadingOffsetDeg = 0.0f;
static constexpr float kCompassMinFieldMagnitude = 50.0f;
static constexpr uint16_t kCompassStableSamplesRequired = 18;
static constexpr float kCompassKalmanQ = 0.001f;
static constexpr float kCompassKalmanR = 4.0f;
static constexpr float kCompassKalmanInitP = 400.0f;

static constexpr uint8_t kImuI2cAddressPrimary7Bit = 0x6A;
static constexpr uint8_t kImuI2cAddressSecondary7Bit = 0x6B;
static constexpr uint32_t kImuTemperatureSampleIntervalMs = 2000;

static constexpr bool kEnableSht45Environment = true;
static constexpr uint8_t kSht4xI2cAddress7Bit = 0x44;
static constexpr uint32_t kSht4xSampleIntervalMs = 2000;

// ESP32-C5 link pins (transport groundwork lands in Phase 3).
static constexpr uint8_t kEspUartTxPin = 28;   // UART1 TX -> C5 U0RXD
static constexpr uint8_t kEspUartRxPin = 29;   // UART1 RX <- C5 U0TXD
static constexpr uint8_t kEspEnablePin = 14;   // EN / CHIP_PU
static constexpr uint8_t kEspBootPin = 15;     // BOOT strap
static constexpr uint8_t kEspHostWakePin = 11; // host-wake IRQ
static constexpr uint8_t kEspSpiRxPin = 20;    // SPI0 RX (MISO)
static constexpr uint8_t kEspSpiCsPin = 21;    // SPI0 CSn
static constexpr uint8_t kEspSpiSckPin = 22;   // SPI0 SCK
static constexpr uint8_t kEspSpiTxPin = 23;    // SPI0 TX (MOSI)

// PSRAM readiness: reserved-but-unset. Setting this to a valid CS pin (see
// kPsramCsPin*) and expanding the ring ceiling is the one-constant path to a
// larger capture buffer once PSRAM is populated.
#ifndef MMPR_NODECFG_PSRAM_CS_PIN
#define MMPR_NODECFG_PSRAM_CS_PIN (-1)
#endif
static constexpr int kPsramCsPin = MMPR_NODECFG_PSRAM_CS_PIN;

// --- Time sync ---
// NTP-over-C5 is a Phase 3 stub; disabled for the skeleton (GPS/PPS disciplines
// the clock).
static constexpr bool kEnableNtpSync = false;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr long kGmtOffsetSeconds = 0;
static constexpr int kDaylightOffsetSeconds = 0;

// --- GPS holdover / crystal frequency model ---
// Final board uses a 12.288 MHz 2.5 ppm TCXO -> tighter drift than the tetra
// crystal. Keep the legacy budget until bench-characterised.
static constexpr bool kEnableHoldoverTempComp = false;
static constexpr double kHoldoverDriftUncertaintyPpm = 0.1;   // TCXO
static constexpr uint64_t kHoldoverErrorBudgetNs = 60000ULL;
static constexpr uint64_t kHoldoverMaxAgeUs = 900000000ULL;

#ifndef MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE
#define MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE 1
#endif
static constexpr bool kEnableClockHoldoverMaintenance =
    MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE == 1;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

// ============================================================================
// Validation (static_asserts)
// ============================================================================
static_assert(
    MMPR_NODECFG_AUDIO_INPUT_MODE == 2 || MMPR_NODECFG_AUDIO_INPUT_MODE == 3,
    "sirith_planar supports MMPR_NODECFG_AUDIO_INPUT_MODE 2 (PDM 5-mic) or 3 (synthetic)");
static_assert(kPlanarChannelCount == 5, "planar array is 5 channels (4 corners + center)");
static_assert(kActiveAudioChannels == 5, "planar active channels must be 5");
static_assert(kSensorOffsetCount == 5, "planar sensor offsets must be 5");
static_assert(MMPR_NODECFG_AUDIO_CHANNELS == 5, "planar audio channels must be 5");
static_assert(kActiveAudioSampleRateHz >= 12000 && kActiveAudioSampleRateHz <= 96000,
    "audio sample rate must be 12-96 kHz");
static_assert(kActiveAudioFrameSamples > 0 && kActiveAudioFrameSamples <= 4096,
    "audio frame samples out of supported range");
static_assert(kAudioRingFrames > 0 && kAudioRingFrames <= 32, "audio ring frames must be 1-32");
static_assert(kAudioQueueSlots > 0 && kAudioQueueSlots <= 96, "audio queue slots must be 1-96");
static_assert(kPublishBatchFrames > 0 && kPublishBatchFrames <= 16, "publish batch frames must be 1-16");
static_assert(kPdmDataPinCount == 3, "planar PDM uses 3 data lines (2+2+1 sharing)");
static_assert(
    (static_cast<size_t>(kActiveAudioFrameSamples) * static_cast<size_t>(kActiveAudioChannels) * sizeof(uint32_t) *
         static_cast<size_t>(kAudioRingFrames)) +
        (static_cast<size_t>(kMaxPacketSamplesPerChannel) * static_cast<size_t>(kActiveAudioChannels) *
         sizeof(int16_t) *
         (static_cast<size_t>(kAudioQueueSlots) +
          (kPublishBatchFrames < static_cast<size_t>(kAudioQueueSlots)
               ? kPublishBatchFrames
               : static_cast<size_t>(kAudioQueueSlots)))) <= kAudioBufferRamCeiling,
    "audio ring + queue + active batch budget exceeds kAudioBufferRamCeiling");

}  // namespace nodecfg
