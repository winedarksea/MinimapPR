#pragma once

#include <cstdint>

namespace mmpr {

enum class PicoSerialSampleEdge : uint8_t {
  kRising = 0,
  kFalling = 1,
};

enum class PicoSerialDataPinBias : uint8_t {
  kDisabled = 0,
  kPullDown = 1,
};

}  // namespace mmpr
