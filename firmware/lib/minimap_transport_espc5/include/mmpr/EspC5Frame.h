#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mmpr {

// ESP32-C5 SPI bridge framing (design decision D6 in the sirith_planar plan):
// the C5 co-processor receives framed requests (endpoint path + opaque MMB3
// body) over SPI, performs the HTTP POST itself, and returns a status frame.
// This lets NodeRunner / NodeProtocol / the backend stay completely unaware
// that the uplink is SPI-proxied rather than a direct CYW43/lwIP socket --
// EspC5Publisher (see EspC5Publisher.h) maps this 1:1 onto the existing
// beginBinaryStoreForwardPublish/pollPublish contract.
//
// This header (and its .cpp) is intentionally pico-sdk-free so the framing
// logic -- the part with actual bugs to find -- can be unit tested on the
// host. The pico-sdk-dependent SPI HAL lives in SpiHostLink.h/.cpp.
//
// Wire format, all multi-byte integers little-endian:
//
//   [0]      sync    u8   always kEspC5FrameSyncByte (0xA5)
//   [1]      type    u8   EspC5FrameType
//   [2..3]   seq     u16  sender-assigned sequence number (echoed in replies)
//   [4..7]   len     u32  payload length in bytes
//   [8..8+len)       payload, type-specific body
//   [..+4)   crc32   u32  CRC-32/IEEE over type+seq+len+payload
//
// The CRC intentionally does NOT cover the leading sync byte: the sync byte's
// only job is letting a reader resynchronize after noise/dropped bytes on the
// wire, so it must be checkable (and skippable) before any CRC is known.
//
// A full encoded frame is therefore:
//   1 (sync) + kEspC5FrameHeaderSize (type+seq+len) + len + kEspC5FrameCrcSize
enum class EspC5FrameType : uint8_t {
  kDataPost = 0x01,      // RP2350 -> C5: endpoint path + opaque MMB3 body
  kPostStatus = 0x02,    // C5 -> RP2350: HTTP result for a preceding DATA_POST
  kWifiConfig = 0x03,    // RP2350 -> C5: SSID/PSK provisioning
  kLinkStatus = 0x04,    // C5 -> RP2350: link up/down + RSSI
  // Reserved frame types (D6): wire shape fixed and tested below, but no
  // producer/consumer wired up on either side yet.
  kTimeQuery = 0x05,     // RESERVED (future): NTP-over-C5 -- not implemented
  kBtScanReport = 0x06,  // RESERVED (future seam): BLE scan over C5 -- not implemented
};

inline constexpr uint8_t kEspC5FrameSyncByte = 0xA5;
// type(1) + seq(2) + len(4)
inline constexpr size_t kEspC5FrameHeaderSize = 1 + 2 + 4;
inline constexpr size_t kEspC5FrameCrcSize = 4;
// Sanity ceiling so a corrupt/garbage length field can't make decode try to
// wait for an unbounded number of bytes. Well above any realistic MMB3 batch
// (kPublishBatchByteBudget in node_config.h tops out in the tens of KB).
inline constexpr uint32_t kEspC5FrameMaxPayloadBytes = 256u * 1024u;

struct EspC5Frame {
  EspC5FrameType type = EspC5FrameType::kDataPost;
  uint16_t seq = 0;
  std::vector<uint8_t> payload;
};

// CRC-32/IEEE (the zlib/PNG/802.3 polynomial 0xEDB88320), reflected in/out,
// final XOR 0xFFFFFFFF. `seed` lets callers chain partial computations; pass
// the previous call's return value to continue a CRC across buffers.
uint32_t crc32Ieee(const uint8_t* data, size_t len, uint32_t seed = 0xFFFFFFFFu);

// Encodes `frame` to its complete wire representation (sync..crc32
// inclusive), appending to `outBytes` (does not clear it first).
void encodeEspC5Frame(const EspC5Frame& frame, std::vector<uint8_t>& outBytes);

enum class EspC5DecodeStatus : uint8_t {
  kOk = 0,
  // Fewer bytes are available than the frame needs -- either the header
  // itself isn't fully in yet, or the header parsed fine but its declared len
  // means the payload+crc aren't fully in yet. Both cases mean the same thing
  // to a caller polling a live SPI link: accumulate more bytes and retry. A
  // caller reading a bounded/closed buffer (e.g. "the C5 link dropped mid-
  // frame") treats this the same status as "frame truncated" -- there is no
  // separate wire signal that distinguishes "more is coming" from "no more is
  // coming"; that's a property of the transport, not the framing.
  kIncomplete,
  kBadSync,      // bytes[0] isn't kEspC5FrameSyncByte
  kBadLength,    // len field exceeds kEspC5FrameMaxPayloadBytes
  kCrcMismatch,  // frame complete but CRC-32 does not match
};

// Attempts to decode exactly one frame starting at bytes[0]. On kOk, outFrame
// is populated and consumedBytes is set to the number of bytes the frame
// occupied on the wire (so the caller can advance a ring/stream cursor). On
// kBadSync, consumedBytes is set to 1 so callers can skip a single byte of
// noise and retry resync; on every other non-kOk status consumedBytes is 0.
EspC5DecodeStatus decodeEspC5Frame(
    const uint8_t* bytes,
    size_t len,
    EspC5Frame& outFrame,
    size_t& consumedBytes);

// --- DATA_POST helpers ------------------------------------------------------
// DATA_POST payload layout:
//   [u32 postDeadlineMs][u16 pathLen][path bytes][body bytes (rest)].
// `postDeadlineMs` is the bridge's whole budget for this POST (connect, send,
// TCP loss recovery, response). The host owns the timeout chain so the
// ordering "each timeout outlasts the recovery of the layer below it" is
// enforced in one place (sirith_planar node_config.h) instead of relying on a
// matching constant in the C5 firmware: the bridge must give up and answer
// kEspC5BridgeStatusDeadline by then, and the host waits strictly longer. A
// host timeout that fired while the C5 was still working is the planar
// equivalent of the tetra abort/reconnect spiral (2026-11-23 bench).
// `body` is the already-encoded, opaque MMB3 binary blob built by
// NodeProtocol::buildBinaryStoreForwardPayloadParts -- this module never
// interprets it, only carries it.
void encodeDataPostFrame(
    uint16_t seq,
    uint32_t postDeadlineMs,
    const std::string& endpointPath,
    const std::string& body,
    std::vector<uint8_t>& outBytes);

// Splits a decoded DATA_POST frame's payload back into deadline/path/body.
// Returns false if the payload is too short for the fixed prefix or the
// declared pathLen overruns the payload.
bool decodeDataPostPayload(
    const std::vector<uint8_t>& payload,
    uint32_t& outPostDeadlineMs,
    std::string& outPath,
    std::string& outBody);

// --- POST_STATUS helpers -----------------------------------------------------
// POST_STATUS payload layout: [u8 ok][i32 statusCode] (5 bytes, fixed).
// statusCode is the HTTP status when the POST reached the server, otherwise
// one of the bridge codes below (with ok = 0).
//
// kEspC5BridgeStatusBusy is backpressure, not failure: the C5 had no buffer
// for this DATA_POST and discarded it without attempting the POST. The host
// resends the same frame rather than dropping audio. The bridge must never
// discard a DATA_POST silently -- an unacknowledged drop is indistinguishable
// from a slow server and costs the host its whole timeout.
inline constexpr int32_t kEspC5BridgeStatusBusy = -100;
// The POST did not complete within the frame's postDeadlineMs.
inline constexpr int32_t kEspC5BridgeStatusDeadline = -101;
// No Wi-Fi association / IP when the DATA_POST arrived.
inline constexpr int32_t kEspC5BridgeStatusNoLink = -102;
// TCP connect or send failed before a response (reset, refused, DNS).
inline constexpr int32_t kEspC5BridgeStatusTransportError = -103;

// Minimum margin the host's per-attempt POST_STATUS wait keeps over the
// DATA_POST's postDeadlineMs (+ SPI transfer), so the bridge's deadline reply
// always arrives before the host gives up.
inline constexpr uint32_t kEspC5MinHostTimeoutMarginMs = 100;
void encodePostStatusFrame(uint16_t seq, bool ok, int32_t statusCode, std::vector<uint8_t>& outBytes);
bool decodePostStatusPayload(const std::vector<uint8_t>& payload, bool& outOk, int32_t& outStatusCode);

// --- LINK_STATUS / WIFI_CONFIG wire shapes ----------------------------------
// Stubs: encode/decode exist so the layout is fixed and testable now, even
// though no handler acts on these frames yet (LINK_STATUS is unsolicited
// C5->host telemetry; WIFI_CONFIG provisioning is a future host->C5 op).
//
// LINK_STATUS carries the counters that made the Pico CYW43 loss diagnosable
// on the 2026-11-23 bench (retransmits, buffer high-water) -- without them a
// stall on the far side of the SPI link looks identical to a slow server.
struct LinkStatusPayload {
  bool linkUp = false;
  int8_t rssiDbm = 0;
  uint8_t wifiChannel = 0;              // 0 = unknown / not associated
  bool band5GHz = false;
  uint32_t tcpRetransmitsTotal = 0;     // monotonic since bridge boot
  uint32_t bridgeBufferHighWaterBytes = 0;
  uint32_t postsRefusedBusyTotal = 0;   // DATA_POSTs answered kEspC5BridgeStatusBusy
};
// Payload layout (16 bytes, fixed):
//   [u8 linkUp][i8 rssiDbm][u8 wifiChannel][u8 band5GHz]
//   [u32 tcpRetransmitsTotal][u32 bridgeBufferHighWaterBytes][u32 postsRefusedBusyTotal]
inline constexpr size_t kEspC5LinkStatusPayloadBytes = 16;
void encodeLinkStatusFrame(uint16_t seq, const LinkStatusPayload& status, std::vector<uint8_t>& outBytes);
bool decodeLinkStatusPayload(const std::vector<uint8_t>& payload, LinkStatusPayload& outStatus);

struct WifiConfigPayload {
  std::string ssid;
  std::string psk;
};
// Payload layout: [u8 ssidLen][ssid bytes][u8 pskLen][psk bytes].
void encodeWifiConfigFrame(uint16_t seq, const WifiConfigPayload& config, std::vector<uint8_t>& outBytes);
bool decodeWifiConfigPayload(const std::vector<uint8_t>& payload, WifiConfigPayload& outConfig);

}  // namespace mmpr
