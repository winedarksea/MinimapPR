#pragma once

#include <Arduino.h>

#include <cstddef>
#include <cstdint>

namespace mmpr {

String encodeBase64(const uint8_t* data, size_t length);

}  // namespace mmpr
