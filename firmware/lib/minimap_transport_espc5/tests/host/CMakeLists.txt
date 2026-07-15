# Standalone host test build for the pico-free ESP32-C5 SPI-bridge framing
# (EspC5Frame.h/.cpp -- encode/decode/CRC32 only; SpiHostLink/EspC5Publisher
# pull in hardware/spi.h and are not host-testable). Deliberately NOT wired
# into the firmware CMakeLists -- uses a plain host toolchain, same pattern as
# firmware/nodes/sirith_tetrahedral/tests/host.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
#
cmake_minimum_required(VERSION 3.16)
project(mmpr_espc5_frame_host_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_espc5_frame
        test_espc5_frame.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../../src/EspC5Frame.cpp
)
target_include_directories(test_espc5_frame PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
if (NOT MSVC)
  target_compile_options(test_espc5_frame PRIVATE -Wall -Wextra)
endif()

add_test(NAME espc5_frame COMMAND test_espc5_frame)
