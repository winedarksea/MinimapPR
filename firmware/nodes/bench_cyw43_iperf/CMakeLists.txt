# bench_cyw43_iperf — raw CYW43439 TCP goodput ceiling for the Pico 2 W.
#
# Stage 1 of the multichannel capacity plan (docs/pico2w_cyw43_capacity_results.md):
# a bare TCP blaster (no MMB3, no HTTP) answering "can the radio + gSPI bus
# carry N channels at all?". It shares lwipopts.h, WiFiSupport and
# BleRssiScanner with sirith_tetrahedral so the link config matches production.
# Driven over USB serial by scripts/bench/cyw43_link_ceiling.py.
#
# A custom blaster rather than lwiperf: lwiperf's client has a fixed 10 s
# duration and no per-second report, and Bluefin (ostree) has no iperf2.
#
#   -DBENCH_SYS_CLOCK_KHZ=150000|200000   (default 150000)
#   -DBENCH_ARCH=poll|background          (default poll)
#   -DBENCH_TCP_TMR_INTERVAL_MS=50        (default: production lwipopts value)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
set(PICO_BOARD pico2_w CACHE STRING "Board type")
include(pico_sdk_import.cmake)
project(bench_cyw43_iperf C CXX ASM)
pico_sdk_init()

set(BENCH_SYS_CLOCK_KHZ 150000 CACHE STRING "System clock in kHz")
set(BENCH_ARCH poll CACHE STRING "cyw43 arch: poll or background")
set(BENCH_TCP_TMR_INTERVAL_MS "" CACHE STRING "override lwIP TCP_TMR_INTERVAL (ms); empty = production value")

set(FIRMWARE_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../../lib)
set(TETRA_DIR ${CMAKE_CURRENT_LIST_DIR}/../sirith_tetrahedral)

add_executable(bench_cyw43_iperf
        bench_cyw43_iperf.cpp
        ${FIRMWARE_LIB_DIR}/minimap_transport_cyw43/src/WiFiSupport.cpp
        ${FIRMWARE_LIB_DIR}/minimap_transport_cyw43/src/BleRssiScanner.cpp
)
target_include_directories(bench_cyw43_iperf PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}         # lwipopts.h: wraps the production one, adds bench knobs
        ${TETRA_DIR}
        ${TETRA_DIR}/include              # node_config.h (Wi-Fi credentials, BLE scan timing)
        ${FIRMWARE_LIB_DIR}/minimap_node_runtime/include
        ${FIRMWARE_LIB_DIR}/minimap_transport_cyw43/include
)
target_compile_definitions(bench_cyw43_iperf PRIVATE
        BENCH_SYS_CLOCK_KHZ=${BENCH_SYS_CLOCK_KHZ}
        CYW43_PIO_CLOCK_DIV_DYNAMIC=1     # gSPI divider swept at runtime over serial
        MMPR_ENABLE_BLE_SCAN=1
        CYW43_ENABLE_BLUETOOTH=1
)
if(BENCH_TCP_TMR_INTERVAL_MS)
    target_compile_definitions(bench_cyw43_iperf PRIVATE BENCH_TCP_TMR_INTERVAL_MS=${BENCH_TCP_TMR_INTERVAL_MS})
endif()
if(BENCH_ARCH STREQUAL "background")
    set(BENCH_ARCH_LIB pico_cyw43_arch_lwip_threadsafe_background)
    target_compile_definitions(bench_cyw43_iperf PRIVATE BENCH_ARCH_BACKGROUND=1)
else()
    set(BENCH_ARCH_LIB pico_cyw43_arch_lwip_poll)
endif()
target_link_libraries(bench_cyw43_iperf
        pico_stdlib
        hardware_vreg
        hardware_clocks
        ${BENCH_ARCH_LIB}
        pico_btstack_ble
        pico_btstack_cyw43
)
pico_enable_stdio_uart(bench_cyw43_iperf 0)
pico_enable_stdio_usb(bench_cyw43_iperf 1)
pico_add_extra_outputs(bench_cyw43_iperf)
# Count cyw43 driver 1 ms waits (TX-credit stalls) without patching the SDK.
target_link_options(bench_cyw43_iperf PRIVATE -Wl,--wrap=cyw43_await_background_or_timeout_us)
