# Standalone host test build for the pico-free PDM CIC/halfband decimator
# math (Phase 2, design decision D3). Mirrors the sirith_tetrahedral host
# test pattern: a plain host toolchain, no pico-sdk, so the DSP core can be
# unit-tested off-target.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
#
cmake_minimum_required(VERSION 3.16)
project(mmpr_pdm_decimator_host_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

set(MMPR_AUDIO_PICO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../lib/minimap_audio_pico/include)
set(MMPR_AUDIO_PICO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../lib/minimap_audio_pico/src)

add_executable(test_pdm_cic_decimator
        test_pdm_cic_decimator.cpp
        ${MMPR_AUDIO_PICO_SRC}/PdmCicDecimator.cpp
)
target_include_directories(test_pdm_cic_decimator PRIVATE ${MMPR_AUDIO_PICO_INCLUDE})
if (NOT MSVC)
  target_compile_options(test_pdm_cic_decimator PRIVATE -Wall -Wextra)
endif()
add_test(NAME pdm_cic_decimator COMMAND test_pdm_cic_decimator)
