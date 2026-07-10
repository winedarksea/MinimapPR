# Standalone host test build for the pico-free clock holdover models.
# This is deliberately NOT wired into the firmware CMakeLists — it uses a plain
# host toolchain so the models can be unit-tested off-target.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
#
cmake_minimum_required(VERSION 3.16)
project(mmpr_clock_holdover_host_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_clock_holdover_model test_clock_holdover_model.cpp)
target_include_directories(test_clock_holdover_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
if (NOT MSVC)
  target_compile_options(test_clock_holdover_model PRIVATE -Wall -Wextra)
endif()

add_test(NAME clock_holdover_model COMMAND test_clock_holdover_model)
