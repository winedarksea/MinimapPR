# sirith_planar — 5-mic PDM planar array node (RP2354A / bench: Pico 2).

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

# PICO_BOARD pico2 (NOT pico2_w): guarantees no CYW43 driver is pulled in. The
# bench board is a Pico 2 W with its radio unused; the final board is an
# RP2354A (firmware/boards/sirith_planar_rp2354a.h added at board bring-up:
# 2 MB flash, 12.288 MHz TCXO XOSC).
set(PICO_BOARD pico2 CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(sirith_planar C CXX ASM)

set(PICO_CXX_ENABLE_EXCEPTIONS 1)

pico_sdk_init()

set(FIRMWARE_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../../lib)
include(${FIRMWARE_LIB_DIR}/cmake/mmpr_nodecfg.cmake)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_node_runtime
        ${CMAKE_BINARY_DIR}/minimap_node_runtime)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_audio_pico
        ${CMAKE_BINARY_DIR}/minimap_audio_pico)
add_subdirectory(${FIRMWARE_LIB_DIR}/minimap_transport_espc5
        ${CMAKE_BINARY_DIR}/minimap_transport_espc5)

add_executable(sirith_planar
        sirith_planar.cpp
        )

# Generated PIO headers (need the concrete executable target).
minimap_audio_pico_generate_pio(sirith_planar)
minimap_node_runtime_generate_pio(sirith_planar)

pico_set_program_name(sirith_planar "sirith_planar")
pico_set_program_version(sirith_planar "0.1")

pico_enable_stdio_uart(sirith_planar 0)
pico_enable_stdio_usb(sirith_planar 1)

target_include_directories(sirith_planar PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
)

# No CYW43/lwIP/btstack transport: the Wi-Fi radio is off-board (ESP32-C5).
target_link_libraries(sirith_planar
        minimap_node_runtime
        minimap_audio_pico
        minimap_transport_espc5
        )

mmpr_apply_nodecfg(sirith_planar ${CMAKE_CURRENT_LIST_DIR}/include/node_config.h)

pico_add_extra_outputs(sirith_planar)
