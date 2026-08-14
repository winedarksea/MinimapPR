#pragma once

// sirith_tetra_s3 — node configuration.
//
// Ported from nodes/sirith_tetrahedral/include/node_config.h and re-tuned for
// the ESP32-S3. Audio parameters (32 kHz / 4 ch / 512 samples per frame) are
// held IDENTICAL to the Pico node on purpose: this node exists to produce a
// frame-loss number comparable with that one, and any difference in the audio
// data rate would make the comparison meaningless.
//
// What changed relative to the Pico config, and why:
//   * kAssumedFloorKbitPerSec 1500 -> 6000. That constant is the never-measured
//     CYW43 gSPI goodput assumption at the heart of the whole investigation; the
//     S3's integrated MAC has no 2 Mbit/s SPI shim in the path. It is only used
//     to size the HTTP timeout.
//   * The RAM ceiling is the S3's internal SRAM budget rather than the RP2350's,
//     and there is no BLE stack to leave room for.
//   * The ring stores decoded int16 (the capture task decodes), not raw 32-bit
//     DMA words, so a ring frame is half the size it is on the Pico.
//   * TDM capture bit offset is 0, not 1: the S3's I2S peripheral absorbs the
//     ADAU7112's 1-BCLK delay in hardware via bit_shift.
//   * No GPS/PPS, no I2C sensors, no BLE, no piezo -- all deferred.

#include <cstdint>

#include "mmpr/Types.h"

namespace nodecfg {

// ============================================================================
// Frequently edited settings
// ============================================================================

// --- Audio input mode ---
enum class AudioInputMode : uint8_t {
  kTdm4Mic = 0,     // 4-mic TDM array (ADAU7112) via the S3 I2S peripheral
  kSynthetic = 3,   // no hardware; deterministic synthetic audio
};

// DEFAULTS TO SYNTHETIC. A bare Feather with nothing attached streams valid
// MMB3 at the exact same data rate as the Pico node, which is all the uplink
// measurement needs. Build with -DMMPR_NODECFG_AUDIO_INPUT_MODE=0 once the
// ADAU7112 is wired to capture real audio.
#ifndef MMPR_NODECFG_AUDIO_INPUT_MODE
#define MMPR_NODECFG_AUDIO_INPUT_MODE 3
#endif

// --- Ingest/backend server ---
// Same target as the Pico node, so both can run side by side against one server.
#ifndef MMPR_NODECFG_SERVER_BASE_URL
#define MMPR_NODECFG_SERVER_BASE_URL "http://192.168.8.103:8081"
#endif

static constexpr const char* kWifiSsid = "catlin";
static constexpr const char* kWifiPassword = "DarthWiFi";

// --- Optional subsystems ---
#ifndef MMPR_NODECFG_ENABLE_TFT_STATUS
#define MMPR_NODECFG_ENABLE_TFT_STATUS 1  // on for the dev board; off in production
#endif

// Boot-time TDM self-test against an internal loopback. Requires a jumper from
// kTdmLoopbackDoutPin to kTdmDataPin. See EspI2sTdmLoopback.h.
#ifndef MMPR_NODECFG_ENABLE_I2S_LOOPBACK
#define MMPR_NODECFG_ENABLE_I2S_LOOPBACK 0
#endif

#ifndef MMPR_NODECFG_ENABLE_STATS_SERVER
#define MMPR_NODECFG_ENABLE_STATS_SERVER 1
#endif

// ============================================================================
// Audio input — tunables
// ============================================================================

#ifndef MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ
#define MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ 32000
#endif

#ifndef MMPR_NODECFG_AUDIO_CHANNELS
#define MMPR_NODECFG_AUDIO_CHANNELS 4
#endif

#ifndef MMPR_NODECFG_AUDIO_FRAME_SAMPLES
#define MMPR_NODECFG_AUDIO_FRAME_SAMPLES 512
#endif

// TDM framing knobs. Defaults describe the ADAU7112 waveform the Pico's PIO
// produces (one 1-BCLK positive FSYNC pulse per frame) == I2S_TDM_PCM_SHORT.
// Exposed because framing is the single most likely thing to be wrong on first
// contact with real hardware.
#ifndef MMPR_NODECFG_TDM_WS_WIDTH
#define MMPR_NODECFG_TDM_WS_WIDTH 1
#endif
#ifndef MMPR_NODECFG_TDM_WS_POLARITY_HIGH
#define MMPR_NODECFG_TDM_WS_POLARITY_HIGH 1
#endif
#ifndef MMPR_NODECFG_TDM_BIT_SHIFT
#define MMPR_NODECFG_TDM_BIT_SHIFT 1
#endif
#ifndef MMPR_NODECFG_TDM_LEFT_ALIGN
#define MMPR_NODECFG_TDM_LEFT_ALIGN 0
#endif
// 0, not the Pico's 1: bit_shift above already accounts for the 1-BCLK delay.
#ifndef MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET
#define MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET 0
#endif

#ifndef MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET 20480
#endif

#ifndef MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET
#define MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET 0
#endif

static constexpr AudioInputMode kAudioInputMode = (MMPR_NODECFG_AUDIO_INPUT_MODE == 3)
    ? AudioInputMode::kSynthetic
    : AudioInputMode::kTdm4Mic;
static constexpr bool kUseTdmAudio = kAudioInputMode == AudioInputMode::kTdm4Mic;
static constexpr bool kUseSyntheticAudio = kAudioInputMode == AudioInputMode::kSynthetic;

static constexpr uint32_t kTdmWsWidth = MMPR_NODECFG_TDM_WS_WIDTH;
static constexpr bool kTdmWsPolarityHigh = MMPR_NODECFG_TDM_WS_POLARITY_HIGH == 1;
static constexpr bool kTdmBitShift = MMPR_NODECFG_TDM_BIT_SHIFT == 1;
static constexpr bool kTdmLeftAlign = MMPR_NODECFG_TDM_LEFT_ALIGN == 1;
static constexpr int8_t kTdmCaptureBitOffset = MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET;

// ============================================================================
// Dynamic buffer sizing
// ============================================================================
// Same derivation scheme as the Pico node: the DMA ring, publish queue, publish
// batch and HTTP timeout all fall out of the audio data rate, so changing the
// rate or channel count re-provisions everything -- while staying under a fixed
// RAM ceiling.

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
static constexpr uint32_t kTargetRingMs = 200;             // capture-side absorb window
static constexpr uint32_t kTargetQueueMs = 300;            // publish-side jitter buffer
static constexpr size_t kTargetPublishBytes = 28u * 1024u; // amortize HTTP overhead per round-trip

// Hard ceiling for ring + queue + active batch. The ESP32-S3 has 512 KiB of
// internal SRAM; Wi-Fi/lwIP buffers, the four task stacks and the TFT row buffer
// all come out of the same pool, and PSRAM is deliberately NOT used for audio
// (it is QSPI on this board -- putting the hot path behind it would trade away
// the headroom this node exists to measure).
static constexpr size_t kAudioBufferRamCeiling = 200u * 1024u;

// Used only to size the HTTP timeout. On the Pico this is 1500 kbit/s -- the
// unmeasured CYW43 gSPI assumption that motivated this whole node. The S3's
// integrated MAC does tens of Mbit/s; 6000 is still conservative.
static constexpr uint32_t kAssumedFloorKbitPerSec = 6000;

// --- Raw data-rate inputs ---
static constexpr uint32_t kDerivSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kDerivFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
static constexpr uint32_t kDerivChannels = MMPR_NODECFG_AUDIO_CHANNELS;

static constexpr size_t kDerivPacketSlotBytes =
    static_cast<size_t>(kDerivFrameSamples) * kDerivChannels * sizeof(int16_t);
// Unlike the Pico (which rings raw 32-bit DMA words and decodes on the consumer
// side), the S3 capture task decodes in place, so a ring frame is int16-sized.
static constexpr size_t kDerivRingFrameBytes = kDerivPacketSlotBytes;

// Ring frames: capture-side absorb window, clamped [8, 32].
static constexpr uint32_t kAudioRingFramesAuto = detail::clampU32(
    detail::ceilDivU32(static_cast<uint64_t>(kDerivSampleRateHz) * kTargetRingMs,
                       1000ull * kDerivFrameSamples),
    8u, 32u);

// Publish batch frames: enough packets to reach the target payload, clamp [1, 16].
static constexpr size_t kPublishBatchFramesAuto = detail::clampSize(
    kDerivPacketSlotBytes > 0 ? (kTargetPublishBytes / kDerivPacketSlotBytes) : 1u,
    1u, 16u);

// Queue slots: sized by time, then shrunk to fit the RAM ceiling after the ring
// and the active batch are reserved.
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

// HTTP timeout: time to push one batch at the assumed floor goodput, doubled,
// plus a fixed connect/response allowance. The floor is wider than the Pico's
// [150, 850] because the publish blocks on its own task here: a generous timeout
// costs no capture time, while an over-tight one throws away batches that would
// have landed.
static constexpr uint32_t kHttpTimeoutSendMs = static_cast<uint32_t>(
    (static_cast<uint64_t>(kPublishBatchFramesAuto) * kDerivPacketSlotBytes * 8ull) /
    (kAssumedFloorKbitPerSec == 0 ? 1u : kAssumedFloorKbitPerSec));
static constexpr uint32_t kHttpTimeoutMsAuto =
    detail::clampU32(kHttpTimeoutSendMs * 2u + 250u, 300u, 1500u);

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

// After this many retries of the SAME batch fail, discard it so fresh audio
// flows. Also the node's only expression of "do not retry" for a 400/413.
static constexpr uint32_t kPublishBatchMaxRetries = 1;

// --- Network and backend ---
static constexpr const char* kServerBaseUrl = MMPR_NODECFG_SERVER_BASE_URL;
static constexpr uint32_t kWiFiConnectTimeoutMs = 15000;
static constexpr const char* kIngestPath = "/api/v1/ingest/binary";
static constexpr size_t kPublishBatchByteBudget = MMPR_NODECFG_PUBLISH_BATCH_BYTE_BUDGET;
static constexpr bool kUsePublishBatchByteBudget = MMPR_NODECFG_USE_PUBLISH_BATCH_BYTE_BUDGET == 1;
static constexpr uint32_t kPublishFailureBackoffMs = 0;

// Read-only stats listener, same port/paths as the Pico node's control server so
// scripts/collect_ingest_regression_metrics.py works unchanged.
static constexpr bool kEnableStatsServer = MMPR_NODECFG_ENABLE_STATS_SERVER == 1;
static constexpr uint16_t kStatsServerPort = 8082;
static constexpr const char* kStatsPath = "/api/v1/stats";
static constexpr const char* kPublishTargetPath = "/api/v1/publish-target";

static constexpr size_t kMaxPacketSamplesPerChannel = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;

// --- Node identity ---
// Distinct from the Pico node's "sirith-tetra-" so both can run side by side
// against one server without colliding. Completed at runtime with the last two
// bytes of the eFuse MAC.
static constexpr const char* kNodeIdPrefix = "sirith-tetra-s3-";
static constexpr mmpr::NodeType kNodeType = mmpr::NodeType::kSirithTetra;
static constexpr bool kNodeHasFallbackGeoPosition = true;
static constexpr float kNodeFallbackLatitudeDeg = 44.98698840878797f;
static constexpr float kNodeFallbackLongitudeDeg = -93.2579197515542f;
static constexpr float kNodeFallbackAltitudeM = 0.0f;

// Physical microphone geometry — identical to the Pico tetra node, since the
// array is the same. Centroid-centred, metres. Mic indices: 0 = MK1, 1 = MK2,
// 2 = MK3 (base), 3 = MK4 (top).
static constexpr mmpr::Vec3 kPhysicalSensorOffsetsM[4] = {
    {-0.016238f,  0.025000f, -0.010205f},  // MK1
    { 0.027063f,  0.000000f, -0.010205f},  // MK2
    {-0.016238f, -0.025000f, -0.010205f},  // MK3
    { 0.005413f,  0.000000f,  0.030615f},  // MK4 (top)
};

// TDM slot mapping from the ADAU7112 strap configuration:
//   TDM1 (slot 0) = MK2 = mic 1, TDM2 (slot 1) = MK1 = mic 0,
//   TDM3 (slot 2) = MK4 = mic 3, TDM4 (slot 3) = MK3 = mic 2.
// Output channel i -> slot kOutputChannelToSlot[i], so channel i still
// corresponds to kPhysicalSensorOffsetsM[i].
static constexpr uint8_t kOutputChannelToSlot[4] = {1, 0, 3, 2};

static constexpr const char* kCapabilitiesList[] = {
    "audio",
    "array_localization",
};
static constexpr const char* const* kCapabilities = kCapabilitiesList;
static constexpr size_t kCapabilityCount =
    sizeof(kCapabilitiesList) / sizeof(kCapabilitiesList[0]);

static constexpr const char* kHardwareName = kUseTdmAudio
    ? "sirith_tetra_esp32s3_tdm"
    : "sirith_tetra_esp32s3_synth";
static constexpr const mmpr::Vec3* kSensorOffsetsM = kPhysicalSensorOffsetsM;
static constexpr size_t kSensorOffsetCount =
    sizeof(kPhysicalSensorOffsetsM) / sizeof(kPhysicalSensorOffsetsM[0]);

// No GPS on this node (deferred). These strings ride in every MMB3 batch.
static constexpr const char* kGpsSignalStatus = "missing";
static constexpr const char* kGpsPositionSource = "fallback_static";

// ============================================================================
// Pin map — bare Adafruit ESP32-S3 TFT Feather (provisional)
// ============================================================================
// Occupied by the board: 7/39/40/45/21/35/36 (TFT), 41/42 (I2C/STEMMA),
// 33/34 (NeoPixel). Free after the assignments below: 1, 2, 8, 11-18. The S3
// routes I2S through the GPIO matrix, so any of these can be reassigned when
// the tetrahedral board mounts -- there is no BCLK+1 adjacency constraint like
// the Pico's PIO had.
static constexpr int kTdmBclkPin = 5;          // D5, output (S3 is TDM master)
static constexpr int kTdmWsPin = 6;            // D6, FSYNC/WS output
static constexpr int kTdmDataPin = 9;          // D9, SDATA input
static constexpr int kTdmLoopbackDoutPin = 10; // D10, self-test only; jumper to D9

static constexpr int kI2sPort = 0;
// One descriptor is capped at 4092 B; a frame is 4 slots x 4 B = 16 B, so 128
// frames = 2048 B per descriptor and 16 descriptors = 64 ms of hardware
// buffering ahead of the capture task.
static constexpr uint32_t kI2sDmaFrameNum = 128;
static constexpr uint32_t kI2sDmaDescNum = 16;

// --- TFT (Adafruit ESP32-S3 TFT Feather) ---
static constexpr bool kEnableTftStatus = MMPR_NODECFG_ENABLE_TFT_STATUS == 1;
static constexpr int kTftSckPin = 36;
static constexpr int kTftMosiPin = 35;
static constexpr int kTftCsPin = 7;
static constexpr int kTftDcPin = 39;
static constexpr int kTftResetPin = 40;
static constexpr int kTftBacklightPin = 45;   // TFT_BACKLITE
static constexpr int kTftPowerEnablePin = 21; // TFT_I2C_POWER; panel is dead without it
static constexpr uint32_t kTftRefreshIntervalMs = 500;

// --- I2S loopback self-test ---
static constexpr bool kEnableI2sLoopback = MMPR_NODECFG_ENABLE_I2S_LOOPBACK == 1;

// ============================================================================
// Task placement
// ============================================================================
// Core 1: capture (A) + runner (B) -- the audio path, kept together so a frame
// never crosses a core boundary between capture and packetization.
// Core 0: publisher (C) + stats server + TFT, i.e. everything that blocks on the
// network or a slow peripheral. Wi-Fi's own tasks also live on core 0 by
// default, which is fine: they are exactly what the publisher waits on.
static constexpr int kCaptureTaskCore = 1;
static constexpr uint32_t kCaptureTaskPriority = 20;
static constexpr int kRunnerTaskCore = 1;
static constexpr uint32_t kRunnerTaskPriority = 10;
static constexpr uint32_t kRunnerTaskStackBytes = 8192;
static constexpr int kPublisherTaskCore = 0;
static constexpr uint32_t kPublisherTaskPriority = 5;
static constexpr int kAuxTaskCore = 0;

// ============================================================================
// Time sync
// ============================================================================
static constexpr bool kEnableNtpSync = true;
static constexpr const char* kNtpServer = "pool.ntp.org";
static constexpr uint32_t kNtpSyncIntervalMs = 600000;
static constexpr uint32_t kNtpInitialSyncTimeoutMs = 8000;

// --- Holdover / crystal frequency model ---
// Telemetry-only without GPS/PPS, but wired so the model behaves identically to
// the Pico node's.
static constexpr bool kEnableHoldoverTempComp = false;
static constexpr double kHoldoverDriftUncertaintyPpm = 1.0;
static constexpr uint64_t kHoldoverErrorBudgetNs = 60000ULL;
static constexpr uint64_t kHoldoverMaxAgeUs = 900000000ULL;
#ifndef MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE
#define MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE 1
#endif
static constexpr bool kEnableClockHoldoverMaintenance =
    MMPR_NODECFG_ENABLE_CLOCK_HOLDOVER_MAINTENANCE == 1;

// --- Runtime logging ---
static constexpr uint32_t kLogEveryFrames = 100;

// --- Active audio parameters ---
static constexpr uint32_t kActiveAudioSampleRateHz = MMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ;
static constexpr uint32_t kActiveAudioFrameSamples = MMPR_NODECFG_AUDIO_FRAME_SAMPLES;
static constexpr uint8_t kActiveAudioChannels = MMPR_NODECFG_AUDIO_CHANNELS;
static constexpr uint8_t kAudioValidBits = 24;  // ADAU7112 emits 24-bit PCM in 32-bit slots
static constexpr uint8_t kAudioSlotBits = 32;
static constexpr uint8_t kAudioTdmSlots = 4;

// ============================================================================
// Validation (static_asserts)
// ============================================================================

static_assert(
    MMPR_NODECFG_AUDIO_INPUT_MODE == 0 || MMPR_NODECFG_AUDIO_INPUT_MODE == 3,
    "MMPR_NODECFG_AUDIO_INPUT_MODE must be 0 (TDM) or 3 (synthetic) on this node");
static_assert(
    MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET >= -8 && MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET <= 8,
    "MMPR_NODECFG_TDM_CAPTURE_BIT_OFFSET must be in [-8, 8]");
static_assert(kActiveAudioSampleRateHz >= 12000 && kActiveAudioSampleRateHz <= 96000,
              "audio sample rate must be 12-96 kHz");
static_assert(kActiveAudioChannels == 4, "the tetrahedral array is 4-channel");
static_assert(kActiveAudioFrameSamples > 0 && kActiveAudioFrameSamples <= 4096,
              "audio frame samples out of supported range");
static_assert(kAudioRingFrames > 0 && kAudioRingFrames <= 32, "audio ring frames must be 1-32");
static_assert(kAudioQueueSlots > 0 && kAudioQueueSlots <= 96, "audio queue slots must be 1-96");
static_assert(kPublishBatchFrames > 0 && kPublishBatchFrames <= 16,
              "publish batch frames must be 1-16");
static_assert(kPublishBatchByteBudget >= 4096 && kPublishBatchByteBudget <= 32768,
              "publish byte budget must be 4-32 KiB");
// One DMA descriptor must stay under the S3 GDMA's 4092-byte cap.
static_assert(
    kI2sDmaFrameNum * kAudioTdmSlots * (kAudioSlotBits / 8u) <= 4092u,
    "kI2sDmaFrameNum yields an oversized DMA descriptor");
// The capture task assembles whole audio frames out of whole DMA descriptors.
static_assert(
    (kActiveAudioFrameSamples % kI2sDmaFrameNum) == 0,
    "audio frame samples must be a multiple of kI2sDmaFrameNum");
// Ring + publish queue + active batch must fit under the RAM ceiling. The
// derivation guarantees this; the assert also guards a manual -D override.
static_assert(
    (static_cast<size_t>(kActiveAudioFrameSamples) * kActiveAudioChannels * sizeof(int16_t) *
         static_cast<size_t>(kAudioRingFrames)) +
        (static_cast<size_t>(kMaxPacketSamplesPerChannel) * kActiveAudioChannels * sizeof(int16_t) *
         (static_cast<size_t>(kAudioQueueSlots) +
          (kPublishBatchFrames < static_cast<size_t>(kAudioQueueSlots)
               ? kPublishBatchFrames
               : static_cast<size_t>(kAudioQueueSlots)))) <= kAudioBufferRamCeiling,
    "audio ring + queue + active batch budget exceeds kAudioBufferRamCeiling");

}  // namespace nodecfg
