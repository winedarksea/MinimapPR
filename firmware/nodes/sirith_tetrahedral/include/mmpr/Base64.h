#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mmpr {

std::string encodeBase64(const uint8_t* data, size_t length);

}  // namespace mmpr
