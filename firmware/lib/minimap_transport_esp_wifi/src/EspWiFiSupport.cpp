#include "mmpr/EspWiFiSupport.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"

namespace mmpr {
namespace {

constexpr const char* kTag = "mmpr-wifi";
constexpr int kConnectedBit = BIT0;

EventGroupHandle_t gWifiEvents = nullptr;
esp_netif_t* gStaNetif = nullptr;
volatile bool gConnected = false;
std::atomic<uint32_t> gConnectCount{0};
char gIpAddress[16] = "0.0.0.0";

void onWifiEvent(void* /*arg*/, esp_event_base_t base, int32_t id, void* /*data*/) {
  if (base != WIFI_EVENT) {
    return;
  }
  if (id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
    return;
  }
  if (id == WIFI_EVENT_STA_DISCONNECTED) {
    gConnected = false;
    std::strcpy(gIpAddress, "0.0.0.0");
    if (gWifiEvents != nullptr) {
      xEventGroupClearBits(gWifiEvents, kConnectedBit);
    }
    // Immediate retry. A publish in flight will fail with kWiFiDisconnected and
    // NodeRunner's adaptive backoff handles the pacing; adding a delay here only
    // lengthens the outage.
    esp_wifi_connect();
  }
}

void onIpEvent(void* /*arg*/, esp_event_base_t base, int32_t id, void* data) {
  if (base != IP_EVENT || id != IP_EVENT_STA_GOT_IP || data == nullptr) {
    return;
  }
  const ip_event_got_ip_t* event = static_cast<const ip_event_got_ip_t*>(data);
  std::snprintf(gIpAddress, sizeof(gIpAddress), IPSTR, IP2STR(&event->ip_info.ip));
  gConnected = true;
  gConnectCount.fetch_add(1, std::memory_order_relaxed);
  if (gWifiEvents != nullptr) {
    xEventGroupSetBits(gWifiEvents, kConnectedBit);
  }
  ESP_LOGI(kTag, "connected ip=%s", gIpAddress);
}

bool ensureNvs() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    // A firmware change can leave the NVS partition in an incompatible state.
    // Erasing loses the persisted boot count, which is only a telemetry blip;
    // failing to init NVS would take Wi-Fi down entirely.
    ESP_LOGW(kTag, "erasing NVS partition (%s)", esp_err_to_name(err));
    if (nvs_flash_erase() != ESP_OK) {
      return false;
    }
    err = nvs_flash_init();
  }
  return err == ESP_OK;
}

}  // namespace

bool espWiFiBegin(const EspWiFiConfig& config) {
  if (config.ssid == nullptr || config.ssid[0] == '\0') {
    ESP_LOGE(kTag, "no SSID configured");
    return false;
  }
  if (!ensureNvs()) {
    ESP_LOGE(kTag, "NVS init failed");
    return false;
  }

  if (gWifiEvents == nullptr) {
    gWifiEvents = xEventGroupCreate();
    if (gWifiEvents == nullptr) {
      return false;
    }
  }

  if (esp_netif_init() != ESP_OK) {
    return false;
  }
  const esp_err_t loopErr = esp_event_loop_create_default();
  if (loopErr != ESP_OK && loopErr != ESP_ERR_INVALID_STATE) {
    return false;
  }
  if (gStaNetif == nullptr) {
    gStaNetif = esp_netif_create_default_wifi_sta();
    if (gStaNetif == nullptr) {
      return false;
    }
    if (config.hostname != nullptr && config.hostname[0] != '\0') {
      esp_netif_set_hostname(gStaNetif, config.hostname);
    }
  }

  wifi_init_config_t initConfig = WIFI_INIT_CONFIG_DEFAULT();
  if (esp_wifi_init(&initConfig) != ESP_OK) {
    return false;
  }

  if (esp_event_handler_instance_register(
          WIFI_EVENT, ESP_EVENT_ANY_ID, &onWifiEvent, nullptr, nullptr) != ESP_OK ||
      esp_event_handler_instance_register(
          IP_EVENT, IP_EVENT_STA_GOT_IP, &onIpEvent, nullptr, nullptr) != ESP_OK) {
    return false;
  }

  wifi_config_t staConfig = {};
  std::strncpy(
      reinterpret_cast<char*>(staConfig.sta.ssid), config.ssid, sizeof(staConfig.sta.ssid) - 1);
  if (config.password != nullptr) {
    std::strncpy(
        reinterpret_cast<char*>(staConfig.sta.password),
        config.password,
        sizeof(staConfig.sta.password) - 1);
  }
  staConfig.sta.threshold.authmode =
      (config.password != nullptr && config.password[0] != '\0') ? WIFI_AUTH_WPA2_PSK
                                                                 : WIFI_AUTH_OPEN;

  if (esp_wifi_set_mode(WIFI_MODE_STA) != ESP_OK ||
      esp_wifi_set_config(WIFI_IF_STA, &staConfig) != ESP_OK ||
      esp_wifi_start() != ESP_OK) {
    return false;
  }

  if (config.disablePowerSave) {
    esp_wifi_set_ps(WIFI_PS_NONE);
  }

  const EventBits_t bits = xEventGroupWaitBits(
      gWifiEvents,
      kConnectedBit,
      pdFALSE,
      pdTRUE,
      pdMS_TO_TICKS(config.connectTimeoutMs));
  if ((bits & kConnectedBit) == 0) {
    ESP_LOGE(kTag, "connect to '%s' timed out after %u ms",
             config.ssid, static_cast<unsigned>(config.connectTimeoutMs));
    return false;
  }
  return true;
}

bool espWiFiConnected() {
  return gConnected;
}

const char* espWiFiIpAddress() {
  return gIpAddress;
}

int8_t espWiFiRssiDbm() {
  if (!gConnected) {
    return 0;
  }
  wifi_ap_record_t apRecord = {};
  if (esp_wifi_sta_get_ap_info(&apRecord) != ESP_OK) {
    return 0;
  }
  return static_cast<int8_t>(apRecord.rssi);
}

uint32_t espWiFiConnectCount() {
  return gConnectCount.load(std::memory_order_relaxed);
}

}  // namespace mmpr
