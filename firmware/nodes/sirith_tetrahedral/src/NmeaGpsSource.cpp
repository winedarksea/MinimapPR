#include "mmpr/NmeaGpsSource.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/gpio.h"
#include "pico/time.h"

namespace mmpr {
namespace {

constexpr const char* kGpsSignalMissing = "missing";
constexpr const char* kGpsSignalNoFix = "no_fix";
constexpr const char* kGpsSignalFix2d = "fix_2d";
constexpr const char* kGpsSignalFix3d = "fix_3d";
constexpr const char* kGpsPositionSourceFallback = "fallback_static";
constexpr const char* kGpsPositionSourceUart = "gps_nmea_uart";

constexpr uint64_t kUsPerMs = 1000ULL;
constexpr uint64_t kNsPerSecond = 1000000000ULL;
constexpr uint64_t kNsPerMillisecond = 1000000ULL;
constexpr uint64_t kPpsFallbackTimeoutUs = 3000000ULL;
constexpr int64_t kUtcSentenceSanityLimitNs = 2LL * 1000000000LL;
// UART-delivered NMEA time-of-day trails the actual UTC instant because the
// sentence has to be serialized on-wire before we parse it. Use a fixed
// correction in no-PPS mode so the fallback discipline does not carry a
// systematic ~100-200 ms late bias. PPS remains the high-precision path.
constexpr uint64_t kNmeaSerialLatencyUs = 150000ULL;

int64_t daysFromCivil(int year, unsigned month, unsigned day) {
  year -= month <= 2;
  const int era = (year >= 0 ? year : year - 399) / 400;
  const unsigned yearOfEra = static_cast<unsigned>(year - era * 400);
  const unsigned dayOfYear = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const unsigned dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return era * 146097 + static_cast<int>(dayOfEra) - 719468;
}

}  // namespace

NmeaGpsSource::NmeaGpsSource(const NmeaGpsSourceConfig& config)
    : config_(config), activeGeoPosition_(config.fallbackGeoPosition) {}

bool NmeaGpsSource::begin() {
  uart_init(config_.uart, config_.baudRate);
  gpio_set_function(config_.txPin, GPIO_FUNC_UART);
  gpio_set_function(config_.rxPin, GPIO_FUNC_UART);
  gpio_pull_up(config_.rxPin);

  // Do not claim PPS PIO/IRQ resources until a GPS receiver is actually
  // speaking NMEA. With no receiver attached, an exposed PPS pin can be noisy
  // enough to create an IRQ storm before the node reaches the audio loop.
  ppsConfigured_ = false;

  uartStarted_ = true;
  healthy_ = false;
  hasFix_ = false;
  hasDateTime_ = false;
  hasAltitude_ = false;
  haveSeenSentences_ = false;
  haveUtcForNextPps_ = false;
  activeFixDimension_ = 0;
  activeGeoPosition_ = config_.fallbackGeoPosition;
  lineLength_ = 0;
  lastSentenceUs_ = 0;
  lastFixUs_ = 0;
  nextPpsUtcNs_ = 0;
  loggedFirstSentence_ = false;
  loggedPpsCaptureState_ = false;
  loggedHealthyState_ = false;
  loggedFixState_ = false;
  loggedNoPpsFallback_ = false;
  haveAlignedPpsEpoch_ = false;
  ppsSignalCurrentlyObserved_ = false;
  lastAppliedPpsEdgeCount_ = 0;
  lastPpsEdgeUs_ = 0;
  ppsUtcLabelValidAfterMonotonicUs_ = 0;
  return true;
}

void NmeaGpsSource::bindAudioSource(IAudioSource* audioSource) {
  audioSourceForPps_ = audioSource;
  if (!ppsConfigured_) {
    return;
  }
  ppsCapture_.bindAudioSource(audioSource);
}

void NmeaGpsSource::ensurePpsCaptureStarted() {
  if (ppsConfigured_ || config_.ppsPin < 0) {
    return;
  }

  ppsConfigured_ = ppsCapture_.begin(config_.ppsPin);
  if (ppsConfigured_ && audioSourceForPps_ != nullptr) {
    ppsCapture_.bindAudioSource(audioSourceForPps_);
  }

  if (!loggedPpsCaptureState_) {
    loggedPpsCaptureState_ = true;
    if (ppsConfigured_) {
      std::printf("[gps] PPS capture armed on GP%d after NMEA detected\n", config_.ppsPin);
    } else {
      std::printf("[gps] PPS capture unavailable on GP%d; using NMEA/NTP fallback\n", config_.ppsPin);
    }
  }
}

void NmeaGpsSource::poll(NodeDescriptor& descriptor, NodeClock* clock) {
  if (!uartStarted_) {
    updateDescriptor(descriptor);
    return;
  }

  consumePendingPps(clock);

  size_t bytesConsumed = 0;
  while (uart_is_readable(config_.uart) && bytesConsumed < config_.maxBytesPerPoll) {
    const char byte = static_cast<char>(uart_getc(config_.uart));
    ++bytesConsumed;
    if (byte == '\r') {
      continue;
    }
    if (byte == '\n') {
      lineBuffer_[lineLength_] = '\0';
      if (lineLength_ > 0) {
        consumeLine(lineBuffer_, clock);
      }
      lineLength_ = 0;
      continue;
    }

    if (lineLength_ + 1 < sizeof(lineBuffer_)) {
      lineBuffer_[lineLength_++] = byte;
    } else {
      lineLength_ = 0;
    }
  }

  const uint64_t nowUs = time_us_64();
  const uint64_t sentenceAgeUs = (lastSentenceUs_ > 0) ? (nowUs - lastSentenceUs_) : UINT64_MAX;
  const uint64_t fixAgeUs = (lastFixUs_ > 0) ? (nowUs - lastFixUs_) : UINT64_MAX;
  const bool ppsRecentlyObserved =
      ppsConfigured_ &&
      lastPpsEdgeUs_ > 0 &&
      nowUs >= lastPpsEdgeUs_ &&
      (nowUs - lastPpsEdgeUs_) <= kPpsFallbackTimeoutUs;
  if (ppsConfigured_ && ppsSignalCurrentlyObserved_ && !ppsRecentlyObserved) {
    ppsSignalCurrentlyObserved_ = false;
    haveAlignedPpsEpoch_ = false;
    std::printf("[gps] PPS signal stale; falling back to NMEA/NTP discipline until edges resume\n");
  }

  if (!haveSeenSentences_ || sentenceAgeUs > (static_cast<uint64_t>(config_.missingSentenceTimeoutMs) * kUsPerMs)) {
    healthy_ = false;
    hasFix_ = false;
    hasAltitude_ = false;
    hasDateTime_ = false;
    haveUtcForNextPps_ = false;
    haveAlignedPpsEpoch_ = false;
    ppsUtcLabelValidAfterMonotonicUs_ = 0;
    activeFixDimension_ = 0;
    activeGeoPosition_ = config_.fallbackGeoPosition;
  } else if (hasFix_ && fixAgeUs > (static_cast<uint64_t>(config_.staleFixTimeoutMs) * kUsPerMs)) {
    hasFix_ = false;
    hasAltitude_ = false;
    haveUtcForNextPps_ = false;
    haveAlignedPpsEpoch_ = false;
    ppsUtcLabelValidAfterMonotonicUs_ = 0;
    activeFixDimension_ = 0;
    activeGeoPosition_ = config_.fallbackGeoPosition;
  }

  if (haveSeenSentences_ && !loggedFirstSentence_) {
    std::printf("[gps] received first NMEA sentence on uart\n");
    loggedFirstSentence_ = true;
  }

  if (healthy_ != loggedHealthyState_) {
    loggedHealthyState_ = healthy_;
    if (healthy_) {
      std::printf("[gps] sentence stream healthy\n");
    } else {
      std::printf("[gps] sentence stream missing; using fallback position\n");
    }
  }

  if (hasFix_ != loggedFixState_) {
    loggedFixState_ = hasFix_;
    if (hasFix_) {
      std::printf(
          "[gps] fix acquired dim=%u lat=%.6f lon=%.6f alt=%.1f\n",
          static_cast<unsigned>(activeFixDimension_),
          static_cast<double>(activeGeoPosition_.lat),
          static_cast<double>(activeGeoPosition_.lon),
          static_cast<double>(activeGeoPosition_.altM));
    } else {
      std::printf("[gps] no active fix; position source reverted to fallback\n");
    }
  }

  updateDescriptor(descriptor);
}

void NmeaGpsSource::consumePendingPps(NodeClock* clock) {
  if (!ppsConfigured_) {
    return;
  }

  GpsPpsCaptureEvent ppsEvent = {};
  if (clock == nullptr || !hasFix_ || !haveUtcForNextPps_) {
    while (ppsCapture_.consumeNext(ppsEvent)) {
      lastPpsEdgeUs_ = time_us_64();
      if (!ppsSignalCurrentlyObserved_) {
        ppsSignalCurrentlyObserved_ = true;
        loggedNoPpsFallback_ = false;
        std::printf("[gps] PPS edges observed; waiting for UTC sentence before locking\n");
      }
    }
    haveAlignedPpsEpoch_ = false;
    ppsUtcLabelValidAfterMonotonicUs_ = 0;
    return;
  }

  while (ppsCapture_.consumeNext(ppsEvent)) {
    if (!ppsSignalCurrentlyObserved_) {
      ppsSignalCurrentlyObserved_ = true;
      loggedNoPpsFallback_ = false;
      std::printf("[gps] PPS edges resumed; attempting GPS PPS discipline\n");
    }
    if (ppsUtcLabelValidAfterMonotonicUs_ > 0 &&
        ppsEvent.monotonicUs <= ppsUtcLabelValidAfterMonotonicUs_) {
      lastPpsEdgeUs_ = time_us_64();
      lastAppliedPpsEdgeCount_ = ppsEvent.edgeCount;
      haveAlignedPpsEpoch_ = false;
      std::printf(
          "[gps] discarded PPS edge=%u captured before UTC sentence; waiting for next PPS boundary\n",
          static_cast<unsigned>(ppsEvent.edgeCount));
      continue;
    }
    if (haveAlignedPpsEpoch_ &&
        ppsEvent.edgeCount > lastAppliedPpsEdgeCount_ &&
        ppsEvent.edgeCount != (lastAppliedPpsEdgeCount_ + 1u)) {
      const uint32_t skippedEdges = ppsEvent.edgeCount - lastAppliedPpsEdgeCount_ - 1u;
      nextPpsUtcNs_ += static_cast<uint64_t>(skippedEdges) * kNsPerSecond;
      std::printf(
          "[gps] skipped %u PPS edges before=%u current=%u; advancing UTC labeling\n",
          static_cast<unsigned>(skippedEdges),
          static_cast<unsigned>(lastAppliedPpsEdgeCount_),
          static_cast<unsigned>(ppsEvent.edgeCount));
    }
    clock->applyGpsPpsObservation(nextPpsUtcNs_, ppsEvent);
    haveAlignedPpsEpoch_ = true;
    lastAppliedPpsEdgeCount_ = ppsEvent.edgeCount;
    lastPpsEdgeUs_ = time_us_64();
    nextPpsUtcNs_ += kNsPerSecond;
  }
}

void NmeaGpsSource::updateDescriptor(NodeDescriptor& descriptor) const {
  if (hasFix_) {
    descriptor.hasGeoPosition = true;
    descriptor.geoPosition = activeGeoPosition_;
    descriptor.positionSource = kGpsPositionSourceUart;
    descriptor.gpsSignalStatus = (activeFixDimension_ >= 3) ? kGpsSignalFix3d : kGpsSignalFix2d;
    return;
  }

  descriptor.hasGeoPosition = true;
  descriptor.geoPosition = config_.fallbackGeoPosition;
  descriptor.positionSource = kGpsPositionSourceFallback;
  descriptor.gpsSignalStatus = healthy_ ? kGpsSignalNoFix : kGpsSignalMissing;
}

void NmeaGpsSource::consumeLine(const char* line, NodeClock* clock) {
  ParsedSentence parsed = {};
  if (!parseSentence(line, parsed)) {
    return;
  }

  haveSeenSentences_ = true;
  healthy_ = true;
  lastSentenceUs_ = time_us_64();
  if (parsed.hasFix && parsed.hasLocation) {
    hasFix_ = true;
    activeGeoPosition_.lat = parsed.latitudeDeg;
    activeGeoPosition_.lon = parsed.longitudeDeg;
    if (parsed.hasAltitude) {
      activeGeoPosition_.altM = parsed.altitudeM;
      hasAltitude_ = true;
    } else if (!hasAltitude_) {
      activeGeoPosition_.altM = config_.fallbackGeoPosition.altM;
    }
    if (parsed.hasFixDimension) {
      activeFixDimension_ = parsed.fixDimension;
    } else if (activeFixDimension_ == 0) {
      activeFixDimension_ = 2;
    }
    lastFixUs_ = lastSentenceUs_;
  } else if (parsed.hasFixDimension && !parsed.hasFix) {
    hasFix_ = false;
    activeFixDimension_ = 0;
    hasAltitude_ = false;
    haveUtcForNextPps_ = false;
    haveAlignedPpsEpoch_ = false;
    ppsUtcLabelValidAfterMonotonicUs_ = 0;
    activeGeoPosition_ = config_.fallbackGeoPosition;
  }

  // Only arm PPS once the receiver reports an active fix. Modules commonly
  // emit UART time/date before their PPS output is trustworthy; starting PPS
  // capture earlier lets a floating or pre-lock PPS line suppress the NMEA
  // fallback path and destabilize packet timestamps.
  if (hasFix_) {
    ensurePpsCaptureStarted();
  }

  if (parsed.hasUtcDateTime) {
    hasDateTime_ = true;
    const uint64_t sentenceMonotonicUs =
        lastSentenceUs_ > kNmeaSerialLatencyUs ? (lastSentenceUs_ - kNmeaSerialLatencyUs) : 0ULL;
    const bool ppsRecentlyObserved =
        hasFix_ &&
        ppsConfigured_ &&
        lastPpsEdgeUs_ > 0 &&
        lastSentenceUs_ >= lastPpsEdgeUs_ &&
        (lastSentenceUs_ - lastPpsEdgeUs_) <= kPpsFallbackTimeoutUs;
    const uint64_t parsedUtcNs = unixEpochNs(
        parsed.year,
        parsed.month,
        parsed.day,
        parsed.hour,
        parsed.minute,
        parsed.second,
        parsed.millisecond);
    bool utcSentencePlausible = true;
    if (clock != nullptr && clock->timeQuality() != TimeQuality::kFreeRunning) {
      const uint64_t estimatedNowNs = clock->nowUtcNs();
      const int64_t sentenceAgeNs =
          static_cast<int64_t>(estimatedNowNs) - static_cast<int64_t>(parsedUtcNs);
      if (std::llabs(sentenceAgeNs) > kUtcSentenceSanityLimitNs) {
        utcSentencePlausible = false;
        haveUtcForNextPps_ = false;
        haveAlignedPpsEpoch_ = false;
        ppsUtcLabelValidAfterMonotonicUs_ = 0;
        std::printf(
            "[gps] ignored stale/implausible UTC sentence age_ns=%lld while clock=%u\n",
            static_cast<long long>(sentenceAgeNs),
            static_cast<unsigned>(clock->timeQuality()));
      }
    }
    if (ppsConfigured_ && hasFix_) {
      if (utcSentencePlausible) {
        nextPpsUtcNs_ = (parsedUtcNs / kNsPerSecond + 1ULL) * kNsPerSecond;
        haveUtcForNextPps_ = true;
        ppsUtcLabelValidAfterMonotonicUs_ = lastSentenceUs_;
      }
    } else {
      haveUtcForNextPps_ = false;
      haveAlignedPpsEpoch_ = false;
      ppsUtcLabelValidAfterMonotonicUs_ = 0;
    }

    // Only trust NMEA UTC as an absolute time source once the receiver has an
    // active fix. In no-fix mode many modules still emit syntactically valid
    // RMC/ZDA date/time fields, but they can be stale or otherwise not tied to
    // the current epoch. Using them here makes fresh audio look "stale" on the
    // server because packets are timestamped minutes in the past.
    if (clock != nullptr && hasFix_ && utcSentencePlausible && !ppsRecentlyObserved) {
      if (ppsConfigured_ && !loggedNoPpsFallback_) {
        loggedNoPpsFallback_ = true;
        std::printf("[gps] PPS configured but no recent edge observed; using NMEA UTC fallback\n");
      }
      clock->applyNmeaUtcObservation(parsedUtcNs, sentenceMonotonicUs);
    }
  }
}

bool NmeaGpsSource::parseSentence(const char* line, ParsedSentence& outSentence) const {
  if (line == nullptr || line[0] != '$' || !validateChecksum(line)) {
    return false;
  }

  char body[96] = {};
  size_t bodyLength = 0;
  for (const char* cursor = line + 1; *cursor != '\0' && *cursor != '*'; ++cursor) {
    if (bodyLength + 1 >= sizeof(body)) {
      return false;
    }
    body[bodyLength++] = *cursor;
  }
  body[bodyLength] = '\0';

  if (sentenceTypeMatches(body, "GGA")) {
    return parseGgaSentence(body, outSentence);
  }
  if (sentenceTypeMatches(body, "RMC")) {
    return parseRmcSentence(body, outSentence);
  }
  if (sentenceTypeMatches(body, "GLL")) {
    return parseGllSentence(body, outSentence);
  }
  if (sentenceTypeMatches(body, "ZDA")) {
    return parseZdaSentence(body, outSentence);
  }
  return false;
}

bool NmeaGpsSource::parseGgaSentence(char* body, ParsedSentence& outSentence) const {
  TokenizedSentence tokens;
  tokenizeBody(body, tokens);

  const char* latitudeField = fieldAt(tokens, 2);
  const char* latitudeHemisphere = fieldAt(tokens, 3);
  const char* longitudeField = fieldAt(tokens, 4);
  const char* longitudeHemisphere = fieldAt(tokens, 5);
  const char* fixQualityField = fieldAt(tokens, 6);
  const char* altitudeField = fieldAt(tokens, 9);

  int fixQuality = 0;
  if (!parseIntField(fixQualityField, fixQuality) || fixQuality <= 0) {
    outSentence.hasFixDimension = true;
    outSentence.fixDimension = 0;
    return true;
  }

  if (!parseLatitudeField(latitudeField, latitudeHemisphere, outSentence.latitudeDeg) ||
      !parseLongitudeField(longitudeField, longitudeHemisphere, outSentence.longitudeDeg)) {
    return false;
  }

  outSentence.hasFix = true;
  outSentence.hasLocation = true;
  outSentence.hasFixDimension = true;
  outSentence.fixDimension = (fixQuality == 1) ? 2 : 3;

  float altitudeM = 0.0f;
  if (parseFloatField(altitudeField, altitudeM)) {
    outSentence.hasAltitude = true;
    outSentence.altitudeM = altitudeM;
  }

  return true;
}

bool NmeaGpsSource::parseRmcSentence(char* body, ParsedSentence& outSentence) const {
  TokenizedSentence tokens;
  tokenizeBody(body, tokens);

  const char* timeField = fieldAt(tokens, 1);
  const char* statusField = fieldAt(tokens, 2);
  const char* latitudeField = fieldAt(tokens, 3);
  const char* latitudeHemisphere = fieldAt(tokens, 4);
  const char* longitudeField = fieldAt(tokens, 5);
  const char* longitudeHemisphere = fieldAt(tokens, 6);
  const char* dateField = fieldAt(tokens, 9);
  const char* modeField = fieldAt(tokens, 12);

  const bool active = (statusField != nullptr && statusField[0] == 'A');
  outSentence.hasFixDimension = true;
  outSentence.fixDimension = active ? 2 : 0;

  if (active &&
      parseLatitudeField(latitudeField, latitudeHemisphere, outSentence.latitudeDeg) &&
      parseLongitudeField(longitudeField, longitudeHemisphere, outSentence.longitudeDeg)) {
    outSentence.hasFix = true;
    outSentence.hasLocation = true;
    if (modeField != nullptr && (modeField[0] == 'D' || modeField[0] == 'R' || modeField[0] == 'F')) {
      outSentence.fixDimension = 3;
    }
  }

  if (parseUtcTimeField(
          timeField,
          outSentence.hour,
          outSentence.minute,
          outSentence.second,
          outSentence.millisecond) &&
      parseRmcDateField(dateField, outSentence.year, outSentence.month, outSentence.day)) {
    outSentence.hasUtcDateTime = true;
  }

  return true;
}

bool NmeaGpsSource::parseGllSentence(char* body, ParsedSentence& outSentence) const {
  TokenizedSentence tokens;
  tokenizeBody(body, tokens);

  const char* latitudeField = fieldAt(tokens, 1);
  const char* latitudeHemisphere = fieldAt(tokens, 2);
  const char* longitudeField = fieldAt(tokens, 3);
  const char* longitudeHemisphere = fieldAt(tokens, 4);
  const char* statusField = fieldAt(tokens, 6);

  const bool active = (statusField != nullptr && statusField[0] == 'A');
  outSentence.hasFixDimension = true;
  outSentence.fixDimension = active ? 2 : 0;
  if (!active) {
    return true;
  }

  if (!parseLatitudeField(latitudeField, latitudeHemisphere, outSentence.latitudeDeg) ||
      !parseLongitudeField(longitudeField, longitudeHemisphere, outSentence.longitudeDeg)) {
    return false;
  }

  outSentence.hasFix = true;
  outSentence.hasLocation = true;
  return true;
}

bool NmeaGpsSource::parseZdaSentence(char* body, ParsedSentence& outSentence) const {
  TokenizedSentence tokens;
  tokenizeBody(body, tokens);

  const char* timeField = fieldAt(tokens, 1);
  const char* dayField = fieldAt(tokens, 2);
  const char* monthField = fieldAt(tokens, 3);
  const char* yearField = fieldAt(tokens, 4);

  int day = 0;
  int month = 0;
  int year = 0;
  if (!parseUtcTimeField(
          timeField,
          outSentence.hour,
          outSentence.minute,
          outSentence.second,
          outSentence.millisecond) ||
      !parseIntField(dayField, day) ||
      !parseIntField(monthField, month) ||
      !parseIntField(yearField, year)) {
    return false;
  }

  outSentence.hasUtcDateTime = true;
  outSentence.day = static_cast<uint8_t>(day);
  outSentence.month = static_cast<uint8_t>(month);
  outSentence.year = static_cast<uint16_t>(year);
  return true;
}

bool NmeaGpsSource::validateChecksum(const char* line) {
  const char* checksumDelimiter = std::strchr(line, '*');
  if (checksumDelimiter == nullptr || checksumDelimiter[1] == '\0' || checksumDelimiter[2] == '\0') {
    return false;
  }

  uint8_t checksum = 0;
  for (const char* cursor = line + 1; cursor < checksumDelimiter; ++cursor) {
    checksum ^= static_cast<uint8_t>(*cursor);
  }

  char expected[3] = {checksumDelimiter[1], checksumDelimiter[2], '\0'};
  char* end = nullptr;
  const long parsed = std::strtol(expected, &end, 16);
  return end != nullptr && *end == '\0' && parsed == checksum;
}

bool NmeaGpsSource::sentenceTypeMatches(const char* body, const char* sentenceType) {
  if (body == nullptr || sentenceType == nullptr) {
    return false;
  }

  const char* firstComma = std::strchr(body, ',');
  const size_t headerLength = (firstComma != nullptr) ? static_cast<size_t>(firstComma - body) : std::strlen(body);
  const size_t typeLength = std::strlen(sentenceType);
  return headerLength >= typeLength &&
      std::strncmp(body + headerLength - typeLength, sentenceType, typeLength) == 0;
}

bool NmeaGpsSource::parseLatitudeField(const char* field, const char* hemisphere, float& latitudeDeg) {
  if (!parseNmeaCoordinate(field, 2, latitudeDeg)) {
    return false;
  }
  if (hemisphere != nullptr && hemisphere[0] == 'S') {
    latitudeDeg = -latitudeDeg;
  }
  return true;
}

bool NmeaGpsSource::parseLongitudeField(const char* field, const char* hemisphere, float& longitudeDeg) {
  if (!parseNmeaCoordinate(field, 3, longitudeDeg)) {
    return false;
  }
  if (hemisphere != nullptr && hemisphere[0] == 'W') {
    longitudeDeg = -longitudeDeg;
  }
  return true;
}

bool NmeaGpsSource::parseNmeaCoordinate(const char* field, int degreeDigits, float& coordinateDeg) {
  if (isFieldEmpty(field) || static_cast<int>(std::strlen(field)) <= degreeDigits) {
    return false;
  }

  char degreesBuffer[4] = {};
  std::memcpy(degreesBuffer, field, static_cast<size_t>(degreeDigits));
  char* end = nullptr;
  const long degrees = std::strtol(degreesBuffer, &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }

  const float minutes = std::strtof(field + degreeDigits, &end);
  if (end == nullptr || *end != '\0') {
    return false;
  }

  coordinateDeg = static_cast<float>(degrees) + (minutes / 60.0f);
  return true;
}

bool NmeaGpsSource::parseFloatField(const char* field, float& value) {
  if (isFieldEmpty(field)) {
    return false;
  }
  char* end = nullptr;
  value = std::strtof(field, &end);
  return end != nullptr && *end == '\0' && !std::isnan(value);
}

bool NmeaGpsSource::parseIntField(const char* field, int& value) {
  if (isFieldEmpty(field)) {
    return false;
  }
  char* end = nullptr;
  const long parsed = std::strtol(field, &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

bool NmeaGpsSource::parseUtcTimeField(
    const char* field,
    uint8_t& hour,
    uint8_t& minute,
    uint8_t& second,
    uint16_t& millisecond) {
  if (isFieldEmpty(field) || std::strlen(field) < 6) {
    return false;
  }

  char buffer[16] = {};
  std::snprintf(buffer, sizeof(buffer), "%s", field);
  char* fraction = std::strchr(buffer, '.');
  if (fraction != nullptr) {
    *fraction++ = '\0';
  }

  if (std::strlen(buffer) < 6) {
    return false;
  }

  const int parsedHour = (buffer[0] - '0') * 10 + (buffer[1] - '0');
  const int parsedMinute = (buffer[2] - '0') * 10 + (buffer[3] - '0');
  const int parsedSecond = (buffer[4] - '0') * 10 + (buffer[5] - '0');
  if (parsedHour < 0 || parsedHour > 23 || parsedMinute < 0 || parsedMinute > 59 ||
      parsedSecond < 0 || parsedSecond > 60) {
    return false;
  }

  hour = static_cast<uint8_t>(parsedHour);
  minute = static_cast<uint8_t>(parsedMinute);
  second = static_cast<uint8_t>(parsedSecond);
  millisecond = 0;

  if (fraction != nullptr && *fraction != '\0') {
    int scale = 100;
    while (*fraction != '\0' && scale > 0) {
      if (*fraction < '0' || *fraction > '9') {
        return false;
      }
      millisecond = static_cast<uint16_t>(millisecond + ((*fraction - '0') * scale));
      scale /= 10;
      ++fraction;
    }
  }

  return true;
}

bool NmeaGpsSource::parseRmcDateField(const char* field, uint16_t& year, uint8_t& month, uint8_t& day) {
  if (isFieldEmpty(field) || std::strlen(field) != 6) {
    return false;
  }

  const int parsedDay = (field[0] - '0') * 10 + (field[1] - '0');
  const int parsedMonth = (field[2] - '0') * 10 + (field[3] - '0');
  const int parsedYear = (field[4] - '0') * 10 + (field[5] - '0');
  if (parsedDay <= 0 || parsedDay > 31 || parsedMonth <= 0 || parsedMonth > 12) {
    return false;
  }

  day = static_cast<uint8_t>(parsedDay);
  month = static_cast<uint8_t>(parsedMonth);
  year = static_cast<uint16_t>((parsedYear >= 80) ? (1900 + parsedYear) : (2000 + parsedYear));
  return true;
}

bool NmeaGpsSource::isFieldEmpty(const char* field) {
  return field == nullptr || field[0] == '\0';
}

void NmeaGpsSource::tokenizeBody(char* body, TokenizedSentence& outTokens) {
  outTokens.fieldCount = 0;
  if (body == nullptr) {
    return;
  }
  outTokens.fields[outTokens.fieldCount++] = body;
  for (char* cursor = body; *cursor != '\0'; ++cursor) {
    if (*cursor == ',') {
      *cursor = '\0';
      if (outTokens.fieldCount >= TokenizedSentence::kMaxFields) {
        return;
      }
      outTokens.fields[outTokens.fieldCount++] = cursor + 1;
    }
  }
}

const char* NmeaGpsSource::fieldAt(const TokenizedSentence& tokens, size_t fieldIndex) {
  return (fieldIndex < tokens.fieldCount) ? tokens.fields[fieldIndex] : nullptr;
}

uint64_t NmeaGpsSource::unixEpochNs(
    uint16_t year,
    uint8_t month,
    uint8_t day,
    uint8_t hour,
    uint8_t minute,
    uint8_t second,
    uint16_t millisecond) {
  const int64_t days = daysFromCivil(year, month, day);
  const uint64_t seconds = static_cast<uint64_t>(days) * 86400ULL +
      static_cast<uint64_t>(hour) * 3600ULL +
      static_cast<uint64_t>(minute) * 60ULL +
      static_cast<uint64_t>(second);
  return seconds * kNsPerSecond + static_cast<uint64_t>(millisecond) * kNsPerMillisecond;
}

}  // namespace mmpr
