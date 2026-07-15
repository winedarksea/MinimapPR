#include "mmpr/EspC5Frame.h"

namespace mmpr {

namespace {

void appendLeU16(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void appendLeU32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

uint16_t readLeU16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t readLeU32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

}  // namespace

uint32_t crc32Ieee(const uint8_t* data, size_t len, uint32_t seed) {
  uint32_t crc = seed;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return crc;
}

void encodeEspC5Frame(const EspC5Frame& frame, std::vector<uint8_t>& outBytes) {
  const size_t start = outBytes.size();
  outBytes.reserve(outBytes.size() + 1 + kEspC5FrameHeaderSize + frame.payload.size() + kEspC5FrameCrcSize);

  outBytes.push_back(kEspC5FrameSyncByte);
  outBytes.push_back(static_cast<uint8_t>(frame.type));
  appendLeU16(outBytes, frame.seq);
  appendLeU32(outBytes, static_cast<uint32_t>(frame.payload.size()));
  outBytes.insert(outBytes.end(), frame.payload.begin(), frame.payload.end());

  // CRC covers type+seq+len+payload -- everything after the sync byte.
  const uint8_t* crcSpanStart = outBytes.data() + start + 1;
  const size_t crcSpanLen = outBytes.size() - start - 1;
  const uint32_t crc = crc32Ieee(crcSpanStart, crcSpanLen) ^ 0xFFFFFFFFu;
  appendLeU32(outBytes, crc);
}

EspC5DecodeStatus decodeEspC5Frame(
    const uint8_t* bytes,
    size_t len,
    EspC5Frame& outFrame,
    size_t& consumedBytes) {
  consumedBytes = 0;

  if (len < 1) {
    return EspC5DecodeStatus::kIncomplete;
  }
  if (bytes[0] != kEspC5FrameSyncByte) {
    consumedBytes = 1;
    return EspC5DecodeStatus::kBadSync;
  }
  if (len < 1 + kEspC5FrameHeaderSize) {
    return EspC5DecodeStatus::kIncomplete;
  }

  const uint8_t typeByte = bytes[1];
  const uint16_t seq = readLeU16(bytes + 2);
  const uint32_t payloadLen = readLeU32(bytes + 4);

  if (payloadLen > kEspC5FrameMaxPayloadBytes) {
    return EspC5DecodeStatus::kBadLength;
  }

  const size_t frameTotal = 1 + kEspC5FrameHeaderSize + static_cast<size_t>(payloadLen) + kEspC5FrameCrcSize;
  if (len < frameTotal) {
    return EspC5DecodeStatus::kIncomplete;
  }

  const uint8_t* crcSpanStart = bytes + 1;
  const size_t crcSpanLen = kEspC5FrameHeaderSize + static_cast<size_t>(payloadLen);
  const uint32_t computedCrc = crc32Ieee(crcSpanStart, crcSpanLen) ^ 0xFFFFFFFFu;
  const uint32_t wireCrc = readLeU32(bytes + 1 + kEspC5FrameHeaderSize + payloadLen);
  if (computedCrc != wireCrc) {
    return EspC5DecodeStatus::kCrcMismatch;
  }

  outFrame.type = static_cast<EspC5FrameType>(typeByte);
  outFrame.seq = seq;
  outFrame.payload.assign(bytes + 1 + kEspC5FrameHeaderSize, bytes + 1 + kEspC5FrameHeaderSize + payloadLen);
  consumedBytes = frameTotal;
  return EspC5DecodeStatus::kOk;
}

void encodeDataPostFrame(
    uint16_t seq,
    const std::string& endpointPath,
    const std::string& body,
    std::vector<uint8_t>& outBytes) {
  EspC5Frame frame;
  frame.type = EspC5FrameType::kDataPost;
  frame.seq = seq;
  frame.payload.reserve(2 + endpointPath.size() + body.size());
  appendLeU16(frame.payload, static_cast<uint16_t>(endpointPath.size()));
  frame.payload.insert(frame.payload.end(), endpointPath.begin(), endpointPath.end());
  frame.payload.insert(frame.payload.end(), body.begin(), body.end());
  encodeEspC5Frame(frame, outBytes);
}

bool decodeDataPostPayload(
    const std::vector<uint8_t>& payload,
    std::string& outPath,
    std::string& outBody) {
  if (payload.size() < 2) {
    return false;
  }
  const uint16_t pathLen = readLeU16(payload.data());
  if (static_cast<size_t>(pathLen) + 2 > payload.size()) {
    return false;
  }
  outPath.assign(reinterpret_cast<const char*>(payload.data() + 2), pathLen);
  outBody.assign(reinterpret_cast<const char*>(payload.data() + 2 + pathLen), payload.size() - 2 - pathLen);
  return true;
}

void encodePostStatusFrame(uint16_t seq, bool ok, int32_t statusCode, std::vector<uint8_t>& outBytes) {
  EspC5Frame frame;
  frame.type = EspC5FrameType::kPostStatus;
  frame.seq = seq;
  frame.payload.reserve(5);
  frame.payload.push_back(ok ? 1 : 0);
  appendLeU32(frame.payload, static_cast<uint32_t>(statusCode));
  encodeEspC5Frame(frame, outBytes);
}

bool decodePostStatusPayload(const std::vector<uint8_t>& payload, bool& outOk, int32_t& outStatusCode) {
  if (payload.size() < 5) {
    return false;
  }
  outOk = payload[0] != 0;
  outStatusCode = static_cast<int32_t>(readLeU32(payload.data() + 1));
  return true;
}

void encodeLinkStatusFrame(uint16_t seq, const LinkStatusPayload& status, std::vector<uint8_t>& outBytes) {
  EspC5Frame frame;
  frame.type = EspC5FrameType::kLinkStatus;
  frame.seq = seq;
  frame.payload.reserve(2);
  frame.payload.push_back(status.linkUp ? 1 : 0);
  frame.payload.push_back(static_cast<uint8_t>(status.rssiDbm));
  encodeEspC5Frame(frame, outBytes);
}

bool decodeLinkStatusPayload(const std::vector<uint8_t>& payload, LinkStatusPayload& outStatus) {
  if (payload.size() < 2) {
    return false;
  }
  outStatus.linkUp = payload[0] != 0;
  outStatus.rssiDbm = static_cast<int8_t>(payload[1]);
  return true;
}

void encodeWifiConfigFrame(uint16_t seq, const WifiConfigPayload& config, std::vector<uint8_t>& outBytes) {
  EspC5Frame frame;
  frame.type = EspC5FrameType::kWifiConfig;
  frame.seq = seq;
  const uint8_t ssidLen = static_cast<uint8_t>(config.ssid.size() > 255 ? 255 : config.ssid.size());
  const uint8_t pskLen = static_cast<uint8_t>(config.psk.size() > 255 ? 255 : config.psk.size());
  frame.payload.reserve(2 + ssidLen + pskLen);
  frame.payload.push_back(ssidLen);
  frame.payload.insert(frame.payload.end(), config.ssid.begin(), config.ssid.begin() + ssidLen);
  frame.payload.push_back(pskLen);
  frame.payload.insert(frame.payload.end(), config.psk.begin(), config.psk.begin() + pskLen);
  encodeEspC5Frame(frame, outBytes);
}

bool decodeWifiConfigPayload(const std::vector<uint8_t>& payload, WifiConfigPayload& outConfig) {
  if (payload.empty()) {
    return false;
  }
  size_t offset = 0;
  const uint8_t ssidLen = payload[offset++];
  if (offset + ssidLen > payload.size()) {
    return false;
  }
  outConfig.ssid.assign(reinterpret_cast<const char*>(payload.data() + offset), ssidLen);
  offset += ssidLen;

  if (offset >= payload.size()) {
    return false;
  }
  const uint8_t pskLen = payload[offset++];
  if (offset + pskLen > payload.size()) {
    return false;
  }
  outConfig.psk.assign(reinterpret_cast<const char*>(payload.data() + offset), pskLen);
  return true;
}

}  // namespace mmpr
