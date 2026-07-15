# minimap_transport_cyw43 — CYW43/lwIP uplink + BLE transport (INTERFACE library)
#
# Wi-Fi (WiFiSupport), the HTTP store-forward frame publisher, NTP client, the
# tiny publish-target control server, and the BLE RSSI scanner/report publisher.
# This is the transport used by CYW43-equipped nodes (e.g. sirith_tetrahedral on
# a Pico 2 W). Planar/ESP32-C5 nodes do NOT link this library.
#
# btstack/BLE linkage is gated by MMPR_ENABLE_BLE_SCAN, which is a node-level
# decision (derived from node_config.h via mmpr_apply_nodecfg); the consuming
# node CMakeLists adds pico_btstack_ble/pico_btstack_cyw43 when enabled. This
# library only unconditionally requires the CYW43 lwIP arch.

add_library(minimap_transport_cyw43 INTERFACE)

target_sources(minimap_transport_cyw43 INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/src/BleReportPublisher.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/BleRssiScanner.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/HttpFramePublisher.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NodeControlServer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/NtpClient.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/WiFiSupport.cpp
)

# btstack_config.h lives at the include root (SDK searches for it by bare name).
target_include_directories(minimap_transport_cyw43 INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(minimap_transport_cyw43 INTERFACE
        pico_cyw43_arch_lwip_poll
)
