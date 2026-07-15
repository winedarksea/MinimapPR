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
// DATA_POST payload layout: [u16 pathLen][path bytes][body bytes (rest)].
// `body` is the already-encoded, opaque MMB3 binary blob built by
// NodeProtocol::buildBinaryStoreForwardPayloadParts -- this module never
// interprets it, only carries it.
void encodeDataPostFrame(
    uint16_t seq,
    const std::string& endpointPath,
    const std::string& body,
    std::vector<uint8_t>& outBytes);

// Splits a decoded DATA_POST frame's payload back into path/body. Returns
// false if the payload is too short to contain a valid pathLen prefix (or the
// declared pathLen overruns the payload).
bool decodeDataPostPayload(
    const std::vector<uint8_t>& payload,
    std::string& outPath,
    std::string& outBody);

// --- POST_STATUS helpers -----------------------------------------------------
// POST_STATUS payload layout: [u8 ok][i32 statusCode] (5 bytes, fixed).
void encodePostStatusFrame(uint16_t seq, bool ok, int32_t statusCode, std::vector<uint8_t>& outBytes);
bool decodePostStatusPayload(const std::vector<uint8_t>& payload, bool& outOk, int32_t& outStatusCode);

// --- LINK_STATUS / WIFI_CONFIG wire shapes ----------------------------------
// Stubs: encode/decode exist so the layout is fixed and testable now, even
// though no handler acts on these frames yet (LINK_STATUS is unsolicited
// C5->host telemetry; WIFI_CONFIG provisioning is a future host->C5 op).
struct LinkStatusPayload {
  bool linkUp = false;
  int8_t rssiDbm = 0;
};
// Payload layout: [u8 linkUp][i8 rssiDbm] (2 bytes, fixed).
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
