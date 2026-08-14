#pragma once

// Left-justified serial-audio slot -> PCM16 decode.
//
// This is the single definition of how a raw captured audio word becomes the
// int16 sample that goes on the wire. It was extracted from
// SirithPicoTdmSource.cpp / PicoI2SMonoSource.cpp (which each carried an
// identical private copy) so that the RP2350 PIO capture path and the ESP32-S3
// I2S/TDM capture path cannot drift apart: the ESP node's whole value is being
// bit-comparable with the Pico node on the same acoustic input, and that only
// holds if both run this exact arithmetic.
//
// Pure C++ (<cstdint> only) — no pico-sdk, no ESP-IDF.
//
// Capture geometry it assumes: `validBits` of MSB-first PCM sitting at the top
// of a `slotBits`-wide word, i.e. left-justified, with the low
// (slotBits - validBits) bits being pad the source does not drive.
// `captureBitOffset` corrects for a capture path whose sampling point is off by
// a whole bit relative to that (the RP2350 PIO needs +1 for the ADAU7112's
// 1-BCLK I2S-style delay; the ESP32-S3 I2S peripheral absorbs the same delay in
// hardware via `bit_shift`, so it needs 0).

#include <cstdint>

namespace mmpr {

// Shifts a captured word so its PCM payload lands left-justified. A positive
// offset shifts left; a negative offset shifts right while preserving the sign
// bit (the payload is already signed and left-justified at that point).
inline uint32_t applyCaptureBitOffset(uint32_t rawWord, int8_t captureBitOffset) {
  if (captureBitOffset > 0) {
    return rawWord << static_cast<uint8_t>(captureBitOffset);
  }
  if (captureBitOffset < 0) {
    // Right-shifting a left-justified PCM word must preserve the sign bit.
    return static_cast<uint32_t>(
        static_cast<int32_t>(rawWord) >> static_cast<uint8_t>(-captureBitOffset));
  }
  return rawWord;
}

inline int16_t decodeLeftJustifiedSlotToPcm16(
    uint32_t rawWord,
    uint8_t slotBits,
    uint8_t validBits,
    int8_t captureBitOffset) {
  rawWord = applyCaptureBitOffset(rawWord, captureBitOffset);
  const uint8_t paddingBits = static_cast<uint8_t>(slotBits - validBits);
  // ADAU7112 drives 24 valid PCM bits within each 32-bit TDM slot, then SDATA
  // is no longer valid for the remaining clocks. Clamp the pad region low so
  // decode does not depend on bus float or external bias between active slots.
  const uint32_t maskedWord =
      paddingBits == 0 ? rawWord : (rawWord & (~((1u << paddingBits) - 1u)));
  int32_t signedPayload = static_cast<int32_t>(maskedWord) >> paddingBits;

  const int shiftToPcm16 = static_cast<int>(validBits) - 16;
  int32_t pcm16 = shiftToPcm16 > 0 ? (signedPayload >> shiftToPcm16) : (signedPayload << (-shiftToPcm16));
  if (pcm16 > 32767) {
    pcm16 = 32767;
  } else if (pcm16 < -32768) {
    pcm16 = -32768;
  }
  return static_cast<int16_t>(pcm16);
}

}  // namespace mmpr
