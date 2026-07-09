#pragma once

#include <cstddef>
#include <cstdint>

#include "hardware/uart.h"

#include "mmpr/GpsPpsTimerCapture.h"
#include "mmpr/NodeClock.h"
#include "mmpr/Types.h"

namespace mmpr {

struct GpsRuntimeStats {
  bool uartStarted = false;
  bool nmeaHealthy = false;
  bool hasFix = false;
  bool hasDateTime = false;
  bool ppsConfigured = false;
  bool ppsObserved = false;
  bool ppsEpochAligned = false;
  TimeQuality clockQuality = TimeQuality::kFreeRunning;
  uint8_t fixDimension = 0;
  uint32_t currentBaudRate = 0;
  uint32_t sentenceAgeMs = 0;
  uint32_t fixAgeMs = 0;
  uint32_t ppsAgeMs = 0;
  uint64_t uartBytesReceived = 0;
  uint64_t validSentences = 0;
  uint64_t invalidChecksumSentences = 0;
  uint64_t unsupportedSentences = 0;
};

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
  uint32_t baudScanIntervalMs = 3000;
};

class NmeaGpsSource {
 public:
  explicit NmeaGpsSource(const NmeaGpsSourceConfig& config);

  bool begin();
  void bindAudioSource(IAudioSource* audioSource);
  void poll(NodeDescriptor& descriptor, NodeClock* clock = nullptr);

  bool healthy() const { return healthy_; }
  bool hasFix() const { return hasFix_; }
  GpsRuntimeStats stats() const;

 private:
  struct ParsedSentence {
    bool hasFix = false;
    bool hasLocation = false;
    bool hasAltitude = false;
    bool hasUtcDateTime = false;
    bool hasFixDimension = false;
    // GSA is the authoritative NMEA source for the 2D/3D classification.
    // Kept separate from hasFixDimension so a GSA sentence (which carries no
    // location) never trips the no-fix reset path used by GGA/RMC/GLL.
    bool hasGsaFixType = false;
    float latitudeDeg = 0.0f;
    float longitudeDeg = 0.0f;
    float altitudeM = 0.0f;
    uint8_t fixDimension = 0;
    uint8_t gsaFixType = 0;  // 1 = no fix, 2 = 2D, 3 = 3D
    uint16_t year = 0;
    uint8_t month = 0;
    uint8_t day = 0;
    uint8_t hour = 0;
    uint8_t minute = 0;
    uint8_t second = 0;
    uint16_t millisecond = 0;
  };

  // Pre-tokenized view of an NMEA sentence body. The tokenizer rewrites commas
  // in the source buffer to '\0' once, then exposes each field as a
  // null-terminated pointer into that buffer. Callers must keep the source
  // buffer alive for the lifetime of the view.
  struct TokenizedSentence {
    static constexpr size_t kMaxFields = 24;
    const char* fields[kMaxFields] = {};
    size_t fieldCount = 0;
  };

  void updateDescriptor(NodeDescriptor& descriptor) const;
  void ensurePpsCaptureStarted();
  void maybeScanBaudRate();
  void switchBaudRate(uint32_t baudRate);
  void consumePendingPps(NodeClock* clock);
  void consumeLine(const char* line, NodeClock* clock);
  bool parseSentence(const char* line, ParsedSentence& outSentence) const;
  bool parseGgaSentence(char* body, ParsedSentence& outSentence) const;
  bool parseRmcSentence(char* body, ParsedSentence& outSentence) const;
  bool parseGllSentence(char* body, ParsedSentence& outSentence) const;
  bool parseGsaSentence(char* body, ParsedSentence& outSentence) const;
  bool parseZdaSentence(char* body, ParsedSentence& outSentence) const;

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
  static void tokenizeBody(char* body, TokenizedSentence& outTokens);
  static const char* fieldAt(const TokenizedSentence& tokens, size_t fieldIndex);
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
  uint32_t currentBaudRate_ = 0;
  uint32_t baudScanIndex_ = 0;
  uint64_t lastBaudScanUs_ = 0;
  uint64_t uartBytesReceived_ = 0;
  uint64_t validSentences_ = 0;
  uint64_t invalidChecksumSentences_ = 0;
  uint64_t unsupportedSentences_ = 0;
  bool ppsConfigured_ = false;
  bool haveUtcForNextPps_ = false;
  bool haveGsaFixType_ = false;
  uint8_t activeFixDimension_ = 0;
  uint8_t gsaFixType_ = 0;
  GeoPoint activeGeoPosition_ = {};
  uint64_t lastSentenceUs_ = 0;
  uint64_t lastFixUs_ = 0;
  uint64_t nextPpsUtcNs_ = 0;
  GpsPpsTimerCapture ppsCapture_;
  IAudioSource* audioSourceForPps_ = nullptr;
  bool loggedFirstSentence_ = false;
  bool loggedPpsCaptureState_ = false;
  bool loggedHealthyState_ = false;
  bool loggedFixState_ = false;
  bool loggedNoPpsFallback_ = false;
  bool loggedUartBytes_ = false;
  bool loggedInvalidNmeaChecksum_ = false;
  bool loggedUnsupportedNmeaSentence_ = false;
  bool haveAlignedPpsEpoch_ = false;
  bool ppsSignalCurrentlyObserved_ = false;
  uint32_t lastAppliedPpsEdgeCount_ = 0;
  uint64_t lastPpsEdgeUs_ = 0;
  uint64_t ppsUtcLabelValidAfterMonotonicUs_ = 0;
};

}  // namespace mmpr
