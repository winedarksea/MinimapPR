#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mmpr {

void appendBase64(std::string& out, const uint8_t* data, size_t length);
std::string encodeBase64(const uint8_t* data, size_t length);

}  // namespace mmpr
