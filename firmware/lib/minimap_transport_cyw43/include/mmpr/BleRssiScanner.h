#pragma once

#include <cstddef>
#include <cstdint>

namespace mmpr {

struct BleScannerStats {
  uint64_t advertisementsObserved = 0;
  uint64_t advertisementsDropped = 0;
  uint64_t tableEvictions = 0;
  uint32_t activeAdvertisers = 0;
  bool initialized = false;
  bool scanning = false;
};

struct BleAdvertiserSnapshot {
  uint8_t address[6] = {};
  uint8_t addressType = 0;
  uint8_t advType = 0;
  int8_t rssiLast = 0;
  int8_t rssiMin = 0;
  int8_t rssiMax = 0;
  int16_t rssiEwmaQ8 = 0;
  uint16_t count = 0;
  uint32_t lastSeenMs = 0;
  char nameHint[9] = {};
};

class BleRssiScanner {
 public:
  BleRssiScanner(uint16_t scanIntervalUnits, uint16_t scanWindowUnits);

  bool begin();
  void poll();
  size_t snapshotAndClear(BleAdvertiserSnapshot* outSnapshots, size_t maxSnapshots, uint32_t nowMs);
  const BleScannerStats& stats() const { return stats_; }

 private:
  struct AdvertiserEntry {
    bool occupied = false;
    BleAdvertiserSnapshot snapshot = {};
  };

  static void packetHandlerStatic(uint8_t packetType, uint16_t channel, uint8_t* packet, uint16_t size);
  void handlePacket(uint8_t packetType, uint16_t channel, uint8_t* packet, uint16_t size);
  void startScanIfReady();
  void observeAdvertisement(
      uint8_t addressType,
      const uint8_t* address,
      uint8_t advType,
      int8_t rssiDbm,
      const uint8_t* data,
      uint8_t dataLength);
  AdvertiserEntry* findOrAllocateEntry(uint8_t addressType, const uint8_t* address, uint32_t nowMs);
  void copyNameHint(char* dest, size_t destBytes, const uint8_t* data, uint8_t dataLength);
  void refreshActiveCount();

  uint16_t scanIntervalUnits_ = 0;
  uint16_t scanWindowUnits_ = 0;
  bool eventHandlerRegistered_ = false;
  AdvertiserEntry entries_[48] = {};
  BleScannerStats stats_ = {};
};

}  // namespace mmpr
