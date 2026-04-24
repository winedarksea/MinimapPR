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

  if (config_.ppsPin >= 0) {
    ppsConfigured_ = ppsCapture_.begin(config_.ppsPin);
  } else {
    ppsConfigured_ = false;
  }

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
  loggedHealthyState_ = false;
  loggedFixState_ = false;
  loggedPpsEdgeCount_ = 0;
  haveAlignedPpsEpoch_ = false;
  lastAppliedPpsEdgeCount_ = 0;
  return true;
}

void NmeaGpsSource::bindAudioSource(IAudioSource* audioSource) {
  if (!ppsConfigured_) {
    return;
  }
  ppsCapture_.bindAudioSource(audioSource);
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

  if (!haveSeenSentences_ || sentenceAgeUs > (static_cast<uint64_t>(config_.missingSentenceTimeoutMs) * kUsPerMs)) {
    healthy_ = false;
    hasFix_ = false;
    hasAltitude_ = false;
    hasDateTime_ = false;
    haveUtcForNextPps_ = false;
    haveAlignedPpsEpoch_ = false;
    activeFixDimension_ = 0;
    activeGeoPosition_ = config_.fallbackGeoPosition;
  } else if (hasFix_ && fixAgeUs > (static_cast<uint64_t>(config_.staleFixTimeoutMs) * kUsPerMs)) {
    hasFix_ = false;
    hasAltitude_ = false;
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
  if (clock == nullptr || !haveUtcForNextPps_) {
    bool discardedEvent = false;
    uint32_t latestDiscardedEdgeCount = 0;
    while (ppsCapture_.consumeNext(ppsEvent)) {
      discardedEvent = true;
      latestDiscardedEdgeCount = ppsEvent.edgeCount;
    }
    if (discardedEvent && latestDiscardedEdgeCount > loggedPpsEdgeCount_) {
      loggedPpsEdgeCount_ = latestDiscardedEdgeCount;
      std::printf(
          "[gps] discarded unlabeled PPS edges through=%u while awaiting UTC\n",
          static_cast<unsigned>(loggedPpsEdgeCount_));
    }
    haveAlignedPpsEpoch_ = false;
    return;
  }

  while (ppsCapture_.consumeNext(ppsEvent)) {
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
    nextPpsUtcNs_ += kNsPerSecond;
    if (ppsEvent.edgeCount > loggedPpsEdgeCount_) {
      loggedPpsEdgeCount_ = ppsEvent.edgeCount;
      std::printf("[gps] observed PPS edges=%u\n", static_cast<unsigned>(loggedPpsEdgeCount_));
    }
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
    activeGeoPosition_ = config_.fallbackGeoPosition;
  }

  if (parsed.hasUtcDateTime) {
    hasDateTime_ = true;
    if (ppsConfigured_) {
      const uint64_t parsedUtcNs = unixEpochNs(
          parsed.year,
          parsed.month,
          parsed.day,
          parsed.hour,
          parsed.minute,
          parsed.second,
          0);
      nextPpsUtcNs_ = parsedUtcNs + kNsPerSecond;
      haveUtcForNextPps_ = true;
    } else if (clock != nullptr) {
      clock->applyNtpObservation(
          unixEpochNs(
              parsed.year,
              parsed.month,
              parsed.day,
              parsed.hour,
              parsed.minute,
              parsed.second,
              parsed.millisecond),
          time_us_64(),
          0);
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

bool NmeaGpsSource::parseGgaSentence(const char* body, ParsedSentence& outSentence) const {
  char mutableBody[96] = {};
  std::snprintf(mutableBody, sizeof(mutableBody), "%s", body);

  const char* latitudeField = fieldAt(mutableBody, 2);
  const char* latitudeHemisphere = fieldAt(mutableBody, 3);
  const char* longitudeField = fieldAt(mutableBody, 4);
  const char* longitudeHemisphere = fieldAt(mutableBody, 5);
  const char* fixQualityField = fieldAt(mutableBody, 6);
  const char* altitudeField = fieldAt(mutableBody, 9);

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

bool NmeaGpsSource::parseRmcSentence(const char* body, ParsedSentence& outSentence) const {
  char mutableBody[96] = {};
  std::snprintf(mutableBody, sizeof(mutableBody), "%s", body);

  const char* timeField = fieldAt(mutableBody, 1);
  const char* statusField = fieldAt(mutableBody, 2);
  const char* latitudeField = fieldAt(mutableBody, 3);
  const char* latitudeHemisphere = fieldAt(mutableBody, 4);
  const char* longitudeField = fieldAt(mutableBody, 5);
  const char* longitudeHemisphere = fieldAt(mutableBody, 6);
  const char* dateField = fieldAt(mutableBody, 9);
  const char* modeField = fieldAt(mutableBody, 12);

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

bool NmeaGpsSource::parseGllSentence(const char* body, ParsedSentence& outSentence) const {
  char mutableBody[96] = {};
  std::snprintf(mutableBody, sizeof(mutableBody), "%s", body);

  const char* latitudeField = fieldAt(mutableBody, 1);
  const char* latitudeHemisphere = fieldAt(mutableBody, 2);
  const char* longitudeField = fieldAt(mutableBody, 3);
  const char* longitudeHemisphere = fieldAt(mutableBody, 4);
  const char* statusField = fieldAt(mutableBody, 6);

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

bool NmeaGpsSource::parseZdaSentence(const char* body, ParsedSentence& outSentence) const {
  char mutableBody[96] = {};
  std::snprintf(mutableBody, sizeof(mutableBody), "%s", body);

  const char* timeField = fieldAt(mutableBody, 1);
  const char* dayField = fieldAt(mutableBody, 2);
  const char* monthField = fieldAt(mutableBody, 3);
  const char* yearField = fieldAt(mutableBody, 4);

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

const char* NmeaGpsSource::fieldAt(char* body, size_t fieldIndex) {
  size_t currentField = 0;
  char* field = body;
  for (char* cursor = body; ; ++cursor) {
    if (*cursor == ',' || *cursor == '\0') {
      if (currentField == fieldIndex) {
        if (*cursor != '\0') {
          *cursor = '\0';
        }
        return field;
      }
      if (*cursor == '\0') {
        return nullptr;
      }
      ++currentField;
      field = cursor + 1;
    }
  }
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
