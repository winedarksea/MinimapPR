#include "mmpr/BleRssiScanner.h"

#include <cstring>

#include "pico/time.h"

#if MMPR_ENABLE_BLE_SCAN
#include "btstack.h"
#include "pico/btstack_cyw43.h"
#include "pico/cyw43_arch.h"
#endif

namespace mmpr {
namespace {

BleRssiScanner* gActiveScanner = nullptr;

bool sameAddress(const uint8_t* lhs, const uint8_t* rhs) {
  return std::memcmp(lhs, rhs, 6) == 0;
}

void copyAddress(uint8_t* dest, const uint8_t* src) {
  std::memcpy(dest, src, 6);
}

uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

}  // namespace

BleRssiScanner::BleRssiScanner(uint16_t scanIntervalUnits, uint16_t scanWindowUnits)
    : scanIntervalUnits_(scanIntervalUnits), scanWindowUnits_(scanWindowUnits) {}

bool BleRssiScanner::begin() {
#if MMPR_ENABLE_BLE_SCAN
  if (eventHandlerRegistered_) {
    return true;
  }
  if (!btstack_cyw43_init(cyw43_arch_async_context())) {
    return false;
  }
  static btstack_packet_callback_registration_t eventCallbackRegistration = {};
  eventCallbackRegistration.callback = &BleRssiScanner::packetHandlerStatic;
  hci_add_event_handler(&eventCallbackRegistration);
  gActiveScanner = this;
  eventHandlerRegistered_ = true;
  stats_.initialized = true;
  hci_power_control(HCI_POWER_ON);
  return true;
#else
  return false;
#endif
}

void BleRssiScanner::poll() {
#if MMPR_ENABLE_BLE_SCAN
  if (stats_.initialized && !stats_.scanning) {
    startScanIfReady();
  }
#endif
}

size_t BleRssiScanner::snapshotAndClear(
    BleAdvertiserSnapshot* outSnapshots,
    size_t maxSnapshots,
    uint32_t nowMs) {
  size_t written = 0;
  for (AdvertiserEntry& entry : entries_) {
    if (!entry.occupied) {
      continue;
    }
    if (written < maxSnapshots && outSnapshots != nullptr) {
      outSnapshots[written] = entry.snapshot;
      ++written;
    } else {
      ++stats_.advertisementsDropped;
    }
    entry = {};
  }
  (void)nowMs;
  refreshActiveCount();
  return written;
}

void BleRssiScanner::packetHandlerStatic(
    uint8_t packetType,
    uint16_t channel,
    uint8_t* packet,
    uint16_t size) {
  if (gActiveScanner != nullptr) {
    gActiveScanner->handlePacket(packetType, channel, packet, size);
  }
}

void BleRssiScanner::handlePacket(
    uint8_t packetType,
    uint16_t channel,
    uint8_t* packet,
    uint16_t size) {
  (void)channel;
  (void)size;
#if MMPR_ENABLE_BLE_SCAN
  if (packetType != HCI_EVENT_PACKET) {
    return;
  }
  const uint8_t eventType = hci_event_packet_get_type(packet);
  if (eventType == BTSTACK_EVENT_STATE) {
    if (btstack_event_state_get_state(packet) == HCI_STATE_WORKING) {
      startScanIfReady();
    }
    return;
  }
  if (eventType != GAP_EVENT_ADVERTISING_REPORT) {
    return;
  }
  bd_addr_t address = {};
  gap_event_advertising_report_get_address(packet, address);
  const uint8_t addressType = gap_event_advertising_report_get_address_type(packet);
  const uint8_t advType = gap_event_advertising_report_get_advertising_event_type(packet);
  const int8_t rssiDbm = static_cast<int8_t>(gap_event_advertising_report_get_rssi(packet));
  const uint8_t dataLength = gap_event_advertising_report_get_data_length(packet);
  const uint8_t* data = gap_event_advertising_report_get_data(packet);
  observeAdvertisement(addressType, address, advType, rssiDbm, data, dataLength);
#endif
}

void BleRssiScanner::startScanIfReady() {
#if MMPR_ENABLE_BLE_SCAN
  if (stats_.scanning) {
    return;
  }
  gap_set_scan_parameters(0, scanIntervalUnits_, scanWindowUnits_);
  gap_start_scan();
  stats_.scanning = true;
#endif
}

void BleRssiScanner::observeAdvertisement(
    uint8_t addressType,
    const uint8_t* address,
    uint8_t advType,
    int8_t rssiDbm,
    const uint8_t* data,
    uint8_t dataLength) {
  AdvertiserEntry* entry = findOrAllocateEntry(addressType, address, millis32());
  if (entry == nullptr) {
    ++stats_.advertisementsDropped;
    return;
  }

  BleAdvertiserSnapshot& snapshot = entry->snapshot;
  if (!entry->occupied || snapshot.count == 0) {
    entry->occupied = true;
    copyAddress(snapshot.address, address);
    snapshot.addressType = addressType;
    snapshot.advType = advType;
    snapshot.rssiLast = rssiDbm;
    snapshot.rssiMin = rssiDbm;
    snapshot.rssiMax = rssiDbm;
    snapshot.rssiEwmaQ8 = static_cast<int16_t>(static_cast<int16_t>(rssiDbm) * 256);
    snapshot.count = 1;
    copyNameHint(snapshot.nameHint, sizeof(snapshot.nameHint), data, dataLength);
  } else {
    snapshot.advType = advType;
    snapshot.rssiLast = rssiDbm;
    if (rssiDbm < snapshot.rssiMin) {
      snapshot.rssiMin = rssiDbm;
    }
    if (rssiDbm > snapshot.rssiMax) {
      snapshot.rssiMax = rssiDbm;
    }
    snapshot.rssiEwmaQ8 =
        static_cast<int16_t>(((static_cast<int32_t>(snapshot.rssiEwmaQ8) * 3) +
                              (static_cast<int32_t>(rssiDbm) * 256) + 2) /
                             4);
    if (snapshot.count < 0xffffu) {
      ++snapshot.count;
    }
    if (snapshot.nameHint[0] == '\0') {
      copyNameHint(snapshot.nameHint, sizeof(snapshot.nameHint), data, dataLength);
    }
  }
  snapshot.lastSeenMs = millis32();
  ++stats_.advertisementsObserved;
  refreshActiveCount();
}

BleRssiScanner::AdvertiserEntry* BleRssiScanner::findOrAllocateEntry(
    uint8_t addressType,
    const uint8_t* address,
    uint32_t nowMs) {
  AdvertiserEntry* emptyEntry = nullptr;
  AdvertiserEntry* oldestEntry = nullptr;
  for (AdvertiserEntry& entry : entries_) {
    if (entry.occupied &&
        entry.snapshot.addressType == addressType &&
        sameAddress(entry.snapshot.address, address)) {
      return &entry;
    }
    if (!entry.occupied && emptyEntry == nullptr) {
      emptyEntry = &entry;
    }
    if (entry.occupied &&
        (oldestEntry == nullptr || entry.snapshot.lastSeenMs < oldestEntry->snapshot.lastSeenMs)) {
      oldestEntry = &entry;
    }
  }
  if (emptyEntry != nullptr) {
    return emptyEntry;
  }
  if (oldestEntry != nullptr) {
    oldestEntry->snapshot = {};
    oldestEntry->snapshot.lastSeenMs = nowMs;
    ++stats_.tableEvictions;
    return oldestEntry;
  }
  return nullptr;
}

void BleRssiScanner::copyNameHint(
    char* dest,
    size_t destBytes,
    const uint8_t* data,
    uint8_t dataLength) {
  if (dest == nullptr || destBytes == 0) {
    return;
  }
  dest[0] = '\0';
  size_t offset = 0;
  while (offset + 1 < dataLength) {
    const uint8_t fieldLen = data[offset];
    if (fieldLen == 0 || offset + 1u + fieldLen > dataLength) {
      break;
    }
    const uint8_t fieldType = data[offset + 1u];
    if ((fieldType == 0x08 || fieldType == 0x09) && fieldLen > 1u) {
      size_t copyLen = fieldLen - 1u;
      if (copyLen >= destBytes) {
        copyLen = destBytes - 1u;
      }
      std::memcpy(dest, data + offset + 2u, copyLen);
      dest[copyLen] = '\0';
      return;
    }
    offset += static_cast<size_t>(fieldLen) + 1u;
  }
}

void BleRssiScanner::refreshActiveCount() {
  uint32_t count = 0;
  for (const AdvertiserEntry& entry : entries_) {
    if (entry.occupied) {
      ++count;
    }
  }
  stats_.activeAdvertisers = count;
}

}  // namespace mmpr
