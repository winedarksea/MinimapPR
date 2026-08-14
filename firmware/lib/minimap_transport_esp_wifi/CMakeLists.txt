# minimap_transport_esp_wifi — ESP-IDF Wi-Fi + HTTP uplink (IDF component)
#
# Station-mode Wi-Fi, the blocking-socket IUplinkTransport, SNTP time discipline,
# and the read-only node stats listener. Deliberately free of any ESP32-S3
# peripheral code so the planned ESP32-C5 SPI->Wi-Fi bridge can link it as-is for
# its uplink half (see EspWiFiSupport.h).

idf_component_register(
    SRCS ${CMAKE_CURRENT_LIST_DIR}/src/EspWiFiSupport.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/EspHttpFramePublisher.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/EspSntpClient.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/EspNodeStatsServer.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
    REQUIRES minimap_node_runtime esp_wifi esp_netif esp_event nvs_flash lwip esp_timer)
