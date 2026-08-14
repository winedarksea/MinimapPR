# minimap_node_runtime — core node runtime (pico-sdk INTERFACE library)
#
# Node-type-agnostic runtime: clock/holdover, node protocol + runner, GPS/PPS
# capture, environmental + IMU/mag sensors, audio-transport QoS. Carries no
# uplink transport (see minimap_transport_cyw43 / minimap_transport_espc5) and
# no node_config.h — configuration flows in via structs supplied by each node
# main. Sources are compiled into the consuming executable target.

# --- ESP-IDF (ESP32-S3 / future C5) ---------------------------------------
# When this directory is named in an IDF project's EXTRA_COMPONENT_DIRS it is
# registered as a single component (tools/cmake/project.cmake), so the ESP build
# takes this branch and never sees the pico-sdk INTERFACE library below.
#
# Only the platform-portable subset is registered: the MMB3 serializer, the
# capture->publish runner, the clock/holdover model, and the synthetic/silence
# audio sources. Everything excluded here is RP2040/RP2350 hardware (PIO, DMA,
# hardware_i2c/uart/pwm/watchdog): GpsPpsTimerCapture, NmeaGpsSource,
# FailureSnapshot, Lis2mdlMagnetometer, Lsm6TemperatureSensor,
# Sht4xEnvironmentalSource, MagAutoOrientation, PiezoBuzzer,
# FallbackEnvironmentalSource, TemperatureEnvironmentalSource.
#
# NodeClock.h includes GpsPpsTimerCapture.h, but that header is pure structs
# (<cstdint> + IAudioSource.h) and NodeClock.cpp only takes GpsPpsCaptureEvent by
# const reference, so the GPS/PPS clock-discipline entry points link fine without
# the RP2350 capture implementation behind them.
if(ESP_PLATFORM)
    idf_component_register(
        SRCS ${CMAKE_CURRENT_LIST_DIR}/src/Base64.cpp
             ${CMAKE_CURRENT_LIST_DIR}/src/NodeProtocol.cpp
             ${CMAKE_CURRENT_LIST_DIR}/src/NodeRunner.cpp
             ${CMAKE_CURRENT_LIST_DIR}/src/NodeClock.cpp
             ${CMAKE_CURRENT_LIST_DIR}/src/SilenceAudioSource.cpp
             ${CMAKE_CURRENT_LIST_DIR}/src/SyntheticAudioSource.cpp
        INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
        REQUIRES esp_timer)
    return()
endif()

add_library(minimap_node_runtime INTERFACE)

target_sources(minimap_node_runtime INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/src/Base64.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/FailureSnapshot.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/FallbackEnvironmentalSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/GpsPpsTimerCapture.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Lis2mdlMagnetometer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Lsm6TemperatureSensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/MagAutoOrientation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NmeaGpsSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NodeClock.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NodeProtocol.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NodeRunner.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/PiezoBuzzer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Sht4xEnvironmentalSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SilenceAudioSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SyntheticAudioSource.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/TemperatureEnvironmentalSource.cpp
)

target_include_directories(minimap_node_runtime INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(minimap_node_runtime INTERFACE
        pico_stdlib
        hardware_dma
        hardware_i2c
        hardware_pwm
        hardware_pio
        hardware_uart
        hardware_clocks
        hardware_watchdog
        pico_unique_id
)

# PIO headers must be generated against a concrete (executable) target, so each
# node calls this after creating its target. Capture the lib dir at definition
# time (CMAKE_CURRENT_LIST_DIR inside a function resolves to the call site).
set(MINIMAP_NODE_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "minimap_node_runtime source dir")
function(minimap_node_runtime_generate_pio target)
    pico_generate_pio_header(${target} ${MINIMAP_NODE_RUNTIME_DIR}/src/gps_pps_capture.pio)
endfunction()
