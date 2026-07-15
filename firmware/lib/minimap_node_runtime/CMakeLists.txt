# minimap_node_runtime — core node runtime (pico-sdk INTERFACE library)
#
# Node-type-agnostic runtime: clock/holdover, node protocol + runner, GPS/PPS
# capture, environmental + IMU/mag sensors, audio-transport QoS. Carries no
# uplink transport (see minimap_transport_cyw43 / minimap_transport_espc5) and
# no node_config.h — configuration flows in via structs supplied by each node
# main. Sources are compiled into the consuming executable target.

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
