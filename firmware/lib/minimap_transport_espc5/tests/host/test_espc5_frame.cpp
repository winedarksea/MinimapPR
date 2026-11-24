#include "mmpr/EspC5Frame.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool value, const char* label) {
  if (!value) {
    ++failures;
    std::printf("FAIL %s\n", label);
  }
}

}  // namespace

int main() {
  using mmpr::EspC5DecodeStatus;
  using mmpr::EspC5Frame;
  using mmpr::EspC5FrameType;

  // --- CRC-32/IEEE known-answer test (the standard "123456789" check value) --
  {
    const std::string kCheckInput = "123456789";
    const uint32_t crc = mmpr::crc32Ieee(reinterpret_cast<const uint8_t*>(kCheckInput.data()), kCheckInput.size()) ^
                          0xFFFFFFFFu;
    check(crc == 0xCBF43926u, "crc32Ieee matches the standard CRC-32/IEEE check value");
  }

  // --- DATA_POST encode round-trip -------------------------------------------
  {
    std::vector<uint8_t> wire;
    const std::string path = "/api/v1/ingest/binary";
    const std::string body = std::string("MMB3") + std::string(37, '\x07');  // stand-in opaque MMB3 bytes
    mmpr::encodeDataPostFrame(/*seq=*/42, /*postDeadlineMs=*/3000, path, body, wire);

    EspC5Frame decoded;
    size_t consumed = 0;
    const EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kOk, "DATA_POST frame decodes as kOk");
    check(consumed == wire.size(), "DATA_POST decode consumes the whole encoded buffer");
    check(decoded.type == EspC5FrameType::kDataPost, "DATA_POST decoded type is kDataPost");
    check(decoded.seq == 42, "DATA_POST decoded seq round-trips");

    std::string decodedPath, decodedBody;
    uint32_t decodedDeadlineMs = 0;
    check(mmpr::decodeDataPostPayload(decoded.payload, decodedDeadlineMs, decodedPath, decodedBody),
          "DATA_POST payload splits");
    check(decodedDeadlineMs == 3000, "DATA_POST postDeadlineMs round-trips");
    check(decodedPath == path, "DATA_POST path round-trips");
    check(decodedBody == body, "DATA_POST opaque MMB3 body round-trips byte-for-byte");

    // Truncations inside the fixed deadline+pathLen prefix, and a pathLen that
    // overruns the payload, must be rejected rather than read out of bounds.
    std::vector<uint8_t> shortPayload(decoded.payload.begin(), decoded.payload.begin() + 5);
    check(!mmpr::decodeDataPostPayload(shortPayload, decodedDeadlineMs, decodedPath, decodedBody),
          "DATA_POST payload shorter than the fixed prefix is rejected");
    std::vector<uint8_t> overrun(decoded.payload.begin(), decoded.payload.begin() + 6 + path.size() - 1);
    check(!mmpr::decodeDataPostPayload(overrun, decodedDeadlineMs, decodedPath, decodedBody),
          "DATA_POST pathLen overrunning the payload is rejected");
  }

  // --- Corrupt-CRC rejection ---------------------------------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodeDataPostFrame(1, 3000, "/x", "body", wire);
    // Flip a bit deep in the payload; CRC must catch it.
    wire[wire.size() - mmpr::kEspC5FrameCrcSize - 1] ^= 0xFF;

    EspC5Frame decoded;
    size_t consumed = 0;
    const EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kCrcMismatch, "corrupted payload byte is rejected as kCrcMismatch");
    check(consumed == 0, "kCrcMismatch does not report bytes consumed");
  }
  {
    std::vector<uint8_t> wire;
    mmpr::encodeDataPostFrame(1, 3000, "/x", "body", wire);
    // Flip a bit in the CRC field itself.
    wire.back() ^= 0x01;

    EspC5Frame decoded;
    size_t consumed = 0;
    const EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kCrcMismatch, "corrupted CRC field itself is rejected as kCrcMismatch");
  }

  // --- Bad sync byte ------------------------------------------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodeDataPostFrame(1, 3000, "/x", "body", wire);
    wire[0] = 0x00;

    EspC5Frame decoded;
    size_t consumed = 0;
    const EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kBadSync, "wrong sync byte is rejected as kBadSync");
    check(consumed == 1, "kBadSync reports exactly one consumable noise byte for resync");
  }

  // --- Truncated / incomplete frame handling -------------------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodeDataPostFrame(7, 3000, "/api/v1/ingest/binary", "some-mmb3-bytes", wire);

    // Only the sync byte -- header not even fully in yet.
    EspC5Frame decoded;
    size_t consumed = 0;
    EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), 1, decoded, consumed);
    check(status == EspC5DecodeStatus::kIncomplete, "sync-byte-only buffer is kIncomplete");

    // Header fully in, but payload+crc chopped off partway through.
    const size_t shortLen = wire.size() - mmpr::kEspC5FrameCrcSize - 3;
    status = mmpr::decodeEspC5Frame(wire.data(), shortLen, decoded, consumed);
    check(status == EspC5DecodeStatus::kIncomplete, "payload/crc-truncated buffer is kIncomplete");
    check(consumed == 0, "kIncomplete does not report bytes consumed");

    // Exactly one byte short of the full frame.
    status = mmpr::decodeEspC5Frame(wire.data(), wire.size() - 1, decoded, consumed);
    check(status == EspC5DecodeStatus::kIncomplete, "one-byte-short buffer is kIncomplete");

    // Full frame present: must now succeed (sanity check the above weren't
    // just permanently broken).
    status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kOk, "full buffer decodes fine after the truncated attempts");
  }

  // --- Oversized/garbage length field -------------------------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodeDataPostFrame(1, 3000, "/x", "body", wire);
    // Stomp the len field (bytes [4..8): sync(1)+type(1)+seq(2)=offset 4,
    // little-endian u32) with something absurd.
    wire[4] = 0xFF;
    wire[5] = 0xFF;
    wire[6] = 0xFF;
    wire[7] = 0xFF;

    EspC5Frame decoded;
    size_t consumed = 0;
    const EspC5DecodeStatus status = mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    check(status == EspC5DecodeStatus::kBadLength, "absurd length field is rejected as kBadLength");
  }

  // --- POST_STATUS round-trip (success and failure) -------------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodePostStatusFrame(/*seq=*/9, /*ok=*/true, /*statusCode=*/200, wire);

    EspC5Frame decoded;
    size_t consumed = 0;
    check(mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed) == EspC5DecodeStatus::kOk,
          "POST_STATUS(ok) decodes as kOk");
    check(decoded.type == EspC5FrameType::kPostStatus, "POST_STATUS decoded type is kPostStatus");

    bool ok = false;
    int32_t statusCode = 0;
    check(mmpr::decodePostStatusPayload(decoded.payload, ok, statusCode), "POST_STATUS payload parses");
    check(ok, "POST_STATUS(ok) round-trips ok=true");
    check(statusCode == 200, "POST_STATUS(ok) round-trips statusCode=200");
  }
  {
    std::vector<uint8_t> wire;
    mmpr::encodePostStatusFrame(/*seq=*/10, /*ok=*/false, /*statusCode=*/-3, wire);

    EspC5Frame decoded;
    size_t consumed = 0;
    mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);

    bool ok = true;
    int32_t statusCode = 0;
    mmpr::decodePostStatusPayload(decoded.payload, ok, statusCode);
    check(!ok, "POST_STATUS(failure) round-trips ok=false");
    check(statusCode == -3, "POST_STATUS(failure) round-trips negative statusCode");
  }

  // --- POST_STATUS bridge BUSY (backpressure) round-trip -------------------------
  {
    std::vector<uint8_t> wire;
    mmpr::encodePostStatusFrame(/*seq=*/11, /*ok=*/false, mmpr::kEspC5BridgeStatusBusy, wire);
    EspC5Frame decoded;
    size_t consumed = 0;
    mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed);
    bool ok = true;
    int32_t statusCode = 0;
    check(mmpr::decodePostStatusPayload(decoded.payload, ok, statusCode), "POST_STATUS(busy) parses");
    check(!ok && statusCode == mmpr::kEspC5BridgeStatusBusy, "POST_STATUS(busy) round-trips the bridge code");
  }

  // --- LINK_STATUS frame shape (stub: no handler wired up, but shape is fixed) --
  {
    mmpr::LinkStatusPayload status;
    status.linkUp = true;
    status.rssiDbm = -47;
    status.wifiChannel = 149;
    status.band5GHz = true;
    status.tcpRetransmitsTotal = 0x01020304u;
    status.bridgeBufferHighWaterBytes = 65536u;
    status.postsRefusedBusyTotal = 7u;

    std::vector<uint8_t> wire;
    mmpr::encodeLinkStatusFrame(/*seq=*/3, status, wire);

    EspC5Frame decoded;
    size_t consumed = 0;
    check(mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed) == EspC5DecodeStatus::kOk,
          "LINK_STATUS frame decodes as kOk");
    check(decoded.type == EspC5FrameType::kLinkStatus, "LINK_STATUS decoded type is kLinkStatus");

    mmpr::LinkStatusPayload decodedStatus;
    check(mmpr::decodeLinkStatusPayload(decoded.payload, decodedStatus), "LINK_STATUS payload parses");
    check(decodedStatus.linkUp == true, "LINK_STATUS linkUp round-trips");
    check(decodedStatus.rssiDbm == -47, "LINK_STATUS rssiDbm round-trips (negative dBm)");
    check(decodedStatus.wifiChannel == 149, "LINK_STATUS wifiChannel round-trips");
    check(decodedStatus.band5GHz, "LINK_STATUS band5GHz round-trips");
    check(decodedStatus.tcpRetransmitsTotal == 0x01020304u, "LINK_STATUS tcpRetransmitsTotal round-trips");
    check(decodedStatus.bridgeBufferHighWaterBytes == 65536u, "LINK_STATUS buffer high-water round-trips");
    check(decodedStatus.postsRefusedBusyTotal == 7u, "LINK_STATUS postsRefusedBusyTotal round-trips");
    std::vector<uint8_t> shortLink(decoded.payload.begin(), decoded.payload.end() - 1);
    check(!mmpr::decodeLinkStatusPayload(shortLink, decodedStatus), "truncated LINK_STATUS payload is rejected");
  }

  // --- WIFI_CONFIG frame shape (stub: no handler wired up, but shape is fixed) --
  {
    mmpr::WifiConfigPayload config;
    config.ssid = "sirith-planar-bench";
    config.psk = "correct-horse-battery-staple";

    std::vector<uint8_t> wire;
    mmpr::encodeWifiConfigFrame(/*seq=*/4, config, wire);

    EspC5Frame decoded;
    size_t consumed = 0;
    check(mmpr::decodeEspC5Frame(wire.data(), wire.size(), decoded, consumed) == EspC5DecodeStatus::kOk,
          "WIFI_CONFIG frame decodes as kOk");
    check(decoded.type == EspC5FrameType::kWifiConfig, "WIFI_CONFIG decoded type is kWifiConfig");

    mmpr::WifiConfigPayload decodedConfig;
    check(mmpr::decodeWifiConfigPayload(decoded.payload, decodedConfig), "WIFI_CONFIG payload parses");
    check(decodedConfig.ssid == config.ssid, "WIFI_CONFIG ssid round-trips");
    check(decodedConfig.psk == config.psk, "WIFI_CONFIG psk round-trips");
  }

  // --- Reserved/future frame types still have stable numeric values --------------
  {
    check(static_cast<uint8_t>(EspC5FrameType::kTimeQuery) == 0x05, "kTimeQuery reserved value is stable (0x05)");
    check(static_cast<uint8_t>(EspC5FrameType::kBtScanReport) == 0x06,
          "kBtScanReport reserved value is stable (0x06)");
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
