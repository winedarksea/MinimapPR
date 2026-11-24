#pragma once

// Template for the per-developer Wi-Fi credentials shared by every Wi-Fi node
// (sirith_tetrahedral, sirith_tetra_s3, bench_cyw43_iperf).
//
//   cp firmware/local_config/wifi_credentials.example.h \
//      firmware/local_config/wifi_credentials.h
//
// then edit the copy. wifi_credentials.h is gitignored so real credentials
// never reach the repository; node builds stop with #error if it is missing.

#define MMPR_WIFI_SSID "REPLACE_WIFI_SSID"
#define MMPR_WIFI_PASSWORD "REPLACE_WIFI_PASSWORD"
