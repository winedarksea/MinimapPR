# minimap_transport_espc5 — ESP32-C5 SPI-bridge uplink transport (INTERFACE library)
#
# D6 (sirith_planar plan): the C5 co-processor receives framed requests over
# SPI (endpoint path + opaque MMB3 body), performs the HTTP POST itself, and
# returns a status frame. EspC5Publisher maps that 1:1 onto the existing
# beginBinaryStoreForwardPublish/pollPublish contract (IUplinkTransport), so
# NodeRunner/NodeProtocol/backend are unchanged versus the CYW43 direct-HTTP
# path. This is the transport used by nodes with no onboard Wi-Fi radio (e.g.
# sirith_planar on an RP2354A) -- it does NOT link CYW43/lwIP/btstack.
#
# EspC5Frame.{h,cpp} (frame encode/decode/CRC32) is deliberately pico-sdk-free
# so it can also be built by the host unit tests in tests/host/. SpiHostLink
# is the only pico-sdk-dependent (hardware/spi.h, hardware/gpio.h) piece.

add_library(minimap_transport_espc5 INTERFACE)

target_sources(minimap_transport_espc5 INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/src/EspC5Frame.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SpiHostLink.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/EspC5Publisher.cpp
)

target_include_directories(minimap_transport_espc5 INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(minimap_transport_espc5 INTERFACE
        hardware_spi
        hardware_gpio
        pico_stdlib
)
