#include "mmpr/Base64.h"

namespace mmpr {
namespace {

constexpr char kAlphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

}  // namespace

String encodeBase64(const uint8_t* data, size_t length) {
  if (data == nullptr || length == 0) {
    return String();
  }

  const size_t outputLength = 4 * ((length + 2) / 3);
  String out;
  out.reserve(outputLength);

  size_t index = 0;
  while (index + 3 <= length) {
    const uint32_t chunk =
        (static_cast<uint32_t>(data[index]) << 16) |
        (static_cast<uint32_t>(data[index + 1]) << 8) |
        (static_cast<uint32_t>(data[index + 2]));

    out += kAlphabet[(chunk >> 18) & 0x3F];
    out += kAlphabet[(chunk >> 12) & 0x3F];
    out += kAlphabet[(chunk >> 6) & 0x3F];
    out += kAlphabet[chunk & 0x3F];
    index += 3;
  }

  const size_t remainder = length - index;
  if (remainder == 1) {
    const uint32_t chunk = static_cast<uint32_t>(data[index]) << 16;
    out += kAlphabet[(chunk >> 18) & 0x3F];
    out += kAlphabet[(chunk >> 12) & 0x3F];
    out += '=';
    out += '=';
  } else if (remainder == 2) {
    const uint32_t chunk =
        (static_cast<uint32_t>(data[index]) << 16) |
        (static_cast<uint32_t>(data[index + 1]) << 8);
    out += kAlphabet[(chunk >> 18) & 0x3F];
    out += kAlphabet[(chunk >> 12) & 0x3F];
    out += kAlphabet[(chunk >> 6) & 0x3F];
    out += '=';
  }

  return out;
}

}  // namespace mmpr
