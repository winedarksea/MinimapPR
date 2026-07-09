#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "mmpr/BleRssiScanner.h"
#include "mmpr/HttpFramePublisher.h"
#include "mmpr/Types.h"

namespace mmpr {

struct BleReportPublisherStats {
  uint64_t reportsSent = 0;
  uint64_t reportsDropped = 0;
  uint64_t reportPublishErrors = 0;
  uint32_t lastReportObservationCount = 0;
  uint32_t lastReportStatus = 0;
};

class BleReportPublisher {
 public:
  BleReportPublisher(
      HttpFramePublisher& publisher,
      BleRssiScanner& scanner,
      const NodeDescriptor& node,
      uint32_t reportIntervalMs,
      size_t maxObservations);

  void poll(uint32_t nowMs, uint32_t bootId);
  const BleReportPublisherStats& stats() const { return stats_; }

 private:
  bool buildReportJson(uint32_t nowMs, uint32_t bootId, std::string& outJson, uint32_t& observationCount);
  void appendEscapedJsonString(std::string& out, const char* value) const;
  void appendMac(std::string& out, const uint8_t* address) const;

  HttpFramePublisher& publisher_;
  BleRssiScanner& scanner_;
  const NodeDescriptor& node_;
  uint32_t reportIntervalMs_ = 0;
  size_t maxObservations_ = 0;
  uint32_t nextReportMs_ = 0;
  bool publishInFlight_ = false;
  BleReportPublisherStats stats_ = {};
};

}  // namespace mmpr
