#pragma once

#include <cstddef>
#include <cstdint>

#include "hardware/uart.h"

#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

struct NmeaGpsSourceConfig {
  uart_inst_t* uart = uart0;
  uint32_t baudRate = 9600;
  int txPin = -1;
  int rxPin = -1;
  int ppsPin = -1;
  GeoPoint fallbackGeoPosition = makeGeoPoint(0.0f, 0.0f, 0.0f);
  size_t maxBytesPerPoll = 128;
  uint32_t missingSentenceTimeoutMs = 5000;
  uint32_t staleFixTimeoutMs = 5000;
};

class NmeaGpsSource {
 public:
  explicit NmeaGpsSource(const NmeaGpsSourceConfig& config);

  bool begin();
  void poll(NodeDescriptor& descriptor, NodeClock* clock = nullptr);

  bool healthy() const { return healthy_; }
  bool hasFix() const { return hasFix_; }

 private:
  static void gpioIrqCallback(uint gpio, uint32_t events);

  struct ParsedSentence {
    bool hasFix = false;
    bool hasLocation = false;
    bool hasAltitude = false;
    bool hasUtcDateTime = false;
    bool hasFixDimension = false;
    float latitudeDeg = 0.0f;
    float longitudeDeg = 0.0f;
    float altitudeM = 0.0f;
    uint8_t fixDimension = 0;
    uint16_t year = 0;
    uint8_t month = 0;
    uint8_t day = 0;
    uint8_t hour = 0;
    uint8_t minute = 0;
    uint8_t second = 0;
    uint16_t millisecond = 0;
  };

  void updateDescriptor(NodeDescriptor& descriptor) const;
  void consumePendingPps(NodeClock* clock);
  void consumeLine(const char* line, NodeClock* clock);
  bool parseSentence(const char* line, ParsedSentence& outSentence) const;
  bool parseGgaSentence(const char* body, ParsedSentence& outSentence) const;
  bool parseRmcSentence(const char* body, ParsedSentence& outSentence) const;
  bool parseGllSentence(const char* body, ParsedSentence& outSentence) const;
  bool parseZdaSentence(const char* body, ParsedSentence& outSentence) const;

  static bool validateChecksum(const char* line);
  static bool sentenceTypeMatches(const char* body, const char* sentenceType);
  static bool parseLatitudeField(const char* field, const char* hemisphere, float& latitudeDeg);
  static bool parseLongitudeField(const char* field, const char* hemisphere, float& longitudeDeg);
  static bool parseNmeaCoordinate(const char* field, int degreeDigits, float& coordinateDeg);
  static bool parseFloatField(const char* field, float& value);
  static bool parseIntField(const char* field, int& value);
  static bool parseUtcTimeField(
      const char* field,
      uint8_t& hour,
      uint8_t& minute,
      uint8_t& second,
      uint16_t& millisecond);
  static bool parseRmcDateField(const char* field, uint16_t& year, uint8_t& month, uint8_t& day);
  static bool isFieldEmpty(const char* field);
  static const char* fieldAt(char* body, size_t fieldIndex);
  static uint64_t unixEpochNs(
      uint16_t year,
      uint8_t month,
      uint8_t day,
      uint8_t hour,
      uint8_t minute,
      uint8_t second,
      uint16_t millisecond);

  NmeaGpsSourceConfig config_;
  char lineBuffer_[128] = {};
  size_t lineLength_ = 0;
  bool uartStarted_ = false;
  bool healthy_ = false;
  bool hasFix_ = false;
  bool hasDateTime_ = false;
  bool hasAltitude_ = false;
  bool haveSeenSentences_ = false;
  bool ppsConfigured_ = false;
  bool haveUtcForNextPps_ = false;
  uint8_t activeFixDimension_ = 0;
  GeoPoint activeGeoPosition_ = {};
  uint64_t lastSentenceUs_ = 0;
  uint64_t lastFixUs_ = 0;
  uint64_t nextPpsUtcNs_ = 0;
  uint32_t processedPpsEdgeCount_ = 0;
  volatile uint32_t observedPpsEdgeCount_ = 0;
  volatile uint64_t latestPpsEdgeUs_ = 0;

  static NmeaGpsSource* activeInstance_;
};

}  // namespace mmpr
