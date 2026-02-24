#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <inttypes.h>
#include <new>
#include <string>

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/i2c.h"
#include "hardware/pio.h"
#include "hardware/pio_instructions.h"
#include "hardware/uart.h"

#include "cyw43.h"
#include "lwip/dns.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/tcp.h"

#include "node_config.h"

#ifndef MMPR_FW_VERSION
#define MMPR_FW_VERSION "dev"
#endif

namespace mmpr {

enum class NodeType {
  kPoint,
  kSirithTetra,
  kUnknown,
};

struct Vec3 {
  float x;
  float y;
  float z;
};

struct NodeDescriptor {
  const char* id;
  NodeType type;
  Vec3 positionM;

  const Vec3* sensorOffsetsM;
  size_t sensorCount;

  const char* const* capabilities;
  size_t capabilityCount;

  const char* hardwareName;
  const char* firmwareVersion;
};

struct AudioFrame {
  uint64_t startTimeNs;
  uint32_t sampleRateHz;
  uint8_t channels;
  uint64_t sequence;

  const int16_t* interleavedSamples;
  size_t samplesPerChannel;
};

struct EnvironmentalSample {
  bool hasTemperatureC = false;
  float temperatureC = 0.0f;
  const char* temperatureSource = nullptr;
};

struct PublishResult {
  bool ok = false;
  int statusCode = -1;
  std::string responseBody;
};

class IAudioSource {
 public:
  virtual ~IAudioSource() = default;

  virtual bool begin() = 0;
  virtual uint32_t sampleRateHz() const = 0;
  virtual uint8_t channels() const = 0;
  virtual size_t frameSamples() const = 0;
  virtual bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) = 0;
};

class IEnvironmentalSource {
 public:
  virtual ~IEnvironmentalSource() = default;

  virtual bool begin() = 0;
  virtual bool read(EnvironmentalSample& outSample) = 0;
};

static uint32_t millis32() {
  return to_ms_since_boot(get_absolute_time());
}

static bool isWiFiConnected() {
  return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA) == CYW43_LINK_UP;
}

namespace {

constexpr char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

void appendEscapedString(std::string& out, const char* text) {
  if (text == nullptr) {
    return;
  }

  while (*text != '\0') {
    const char c = *text;
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
    ++text;
  }
}

void appendQuoted(std::string& out, const char* text) {
  out.push_back('"');
  appendEscapedString(out, text);
  out.push_back('"');
}

void appendFloat(std::string& out, float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
  out += buffer;
}

void appendUint64(std::string& out, uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%" PRIu64, value);
  out += buffer;
}

void appendUint32(std::string& out, uint32_t value) {
  char buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%" PRIu32, value);
  out += buffer;
}

const char* nodeTypeToWire(NodeType type) {
  switch (type) {
    case NodeType::kPoint:
      return "point";
    case NodeType::kSirithTetra:
      return "sirith_tetra";
    default:
      return "unknown";
  }
}

}  // namespace

std::string encodeBase64(const uint8_t* data, size_t length) {
  if (data == nullptr || length == 0) {
    return std::string();
  }

  const size_t outputLength = 4 * ((length + 2) / 3);
  std::string out;
  out.reserve(outputLength);

  size_t index = 0;
  while (index + 3 <= length) {
    const uint32_t chunk =
        (static_cast<uint32_t>(data[index]) << 16) |
        (static_cast<uint32_t>(data[index + 1]) << 8) |
        static_cast<uint32_t>(data[index + 2]);

    out.push_back(kBase64Alphabet[(chunk >> 18) & 0x3F]);
    out.push_back(kBase64Alphabet[(chunk >> 12) & 0x3F]);
    out.push_back(kBase64Alphabet[(chunk >> 6) & 0x3F]);
    out.push_back(kBase64Alphabet[chunk & 0x3F]);
    index += 3;
  }

  const size_t remainder = length - index;
  if (remainder == 1) {
    const uint32_t chunk = static_cast<uint32_t>(data[index]) << 16;
    out.push_back(kBase64Alphabet[(chunk >> 18) & 0x3F]);
    out.push_back(kBase64Alphabet[(chunk >> 12) & 0x3F]);
    out.push_back('=');
    out.push_back('=');
  } else if (remainder == 2) {
    const uint32_t chunk =
        (static_cast<uint32_t>(data[index]) << 16) |
        (static_cast<uint32_t>(data[index + 1]) << 8);
    out.push_back(kBase64Alphabet[(chunk >> 18) & 0x3F]);
    out.push_back(kBase64Alphabet[(chunk >> 12) & 0x3F]);
    out.push_back(kBase64Alphabet[(chunk >> 6) & 0x3F]);
    out.push_back('=');
  }

  return out;
}

bool buildIngestPayload(
    const NodeDescriptor& node,
    const AudioFrame& frame,
    const EnvironmentalSample* environment,
    std::string& outPayload) {
  if (node.id == nullptr || node.sensorOffsetsM == nullptr || node.sensorCount == 0 ||
      frame.interleavedSamples == nullptr) {
    return false;
  }

  const size_t bytes = frame.samplesPerChannel * static_cast<size_t>(frame.channels) * sizeof(int16_t);
  const std::string encoded = encodeBase64(reinterpret_cast<const uint8_t*>(frame.interleavedSamples), bytes);

  outPayload.clear();
  outPayload.reserve(640 + encoded.size());

  outPayload += "{\"node\":{";

  outPayload += "\"id\":";
  appendQuoted(outPayload, node.id);

  outPayload += ",\"node_type\":";
  appendQuoted(outPayload, nodeTypeToWire(node.type));

  outPayload += ",\"position_m\":[";
  appendFloat(outPayload, node.positionM.x);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.y);
  outPayload += ',';
  appendFloat(outPayload, node.positionM.z);
  outPayload += ']';

  outPayload += ",\"sensor_offsets_m\":[";
  for (size_t i = 0; i < node.sensorCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    outPayload += '[';
    appendFloat(outPayload, node.sensorOffsetsM[i].x);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].y);
    outPayload += ',';
    appendFloat(outPayload, node.sensorOffsetsM[i].z);
    outPayload += ']';
  }
  outPayload += ']';

  outPayload += ",\"capabilities\":[";
  for (size_t i = 0; i < node.capabilityCount; ++i) {
    if (i > 0) {
      outPayload += ',';
    }
    appendQuoted(outPayload, node.capabilities[i]);
  }
  outPayload += ']';

  outPayload += ",\"metadata\":{";
  outPayload += "\"hardware\":";
  appendQuoted(outPayload, node.hardwareName != nullptr ? node.hardwareName : "unknown");
  outPayload += ",\"firmware\":";
  appendQuoted(outPayload, node.firmwareVersion != nullptr ? node.firmwareVersion : "dev");
  if (environment != nullptr && environment->hasTemperatureC) {
    outPayload += ",\"temperature_c\":";
    appendFloat(outPayload, environment->temperatureC);
    if (environment->temperatureSource != nullptr) {
      outPayload += ",\"temperature_source\":";
      appendQuoted(outPayload, environment->temperatureSource);
    }
  }
  outPayload += "}}";

  outPayload += ",\"frame\":{";
  outPayload += "\"start_time_ns\":";
  appendUint64(outPayload, frame.startTimeNs);

  outPayload += ",\"sample_rate_hz\":";
  appendUint32(outPayload, frame.sampleRateHz);

  outPayload += ",\"channels\":";
  outPayload += std::to_string(static_cast<unsigned>(frame.channels));

  outPayload += ",\"encoding\":\"pcm16le\"";
  outPayload += ",\"samples_b64\":";
  appendQuoted(outPayload, encoded.c_str());

  outPayload += ",\"sequence\":";
  appendUint64(outPayload, frame.sequence);
  outPayload += "}}";

  return true;
}

class I2cBus {
 public:
  bool begin(i2c_inst_t* inst, uint sdaPin, uint sclPin, uint32_t baudHz) {
    if (inst == nullptr) {
      return false;
    }

    inst_ = inst;
    i2c_init(inst_, baudHz);
    gpio_set_function(sdaPin, GPIO_FUNC_I2C);
    gpio_set_function(sclPin, GPIO_FUNC_I2C);
    gpio_pull_up(sdaPin);
    gpio_pull_up(sclPin);
    initialized_ = true;
    return true;
  }

  bool initialized() const { return initialized_; }

  bool write(uint8_t address7Bit, const uint8_t* data, size_t length, bool noStop) {
    if (!initialized_ || data == nullptr || length == 0) {
      return false;
    }
    const int written = i2c_write_blocking(inst_, address7Bit, data, length, noStop);
    return written == static_cast<int>(length);
  }

  bool read(uint8_t address7Bit, uint8_t* data, size_t length, bool noStop) {
    if (!initialized_ || data == nullptr || length == 0) {
      return false;
    }
    const int readCount = i2c_read_blocking(inst_, address7Bit, data, length, noStop);
    return readCount == static_cast<int>(length);
  }

  bool readReg(uint8_t address7Bit, uint8_t reg, uint8_t* outData, size_t outLen) {
    if (!write(address7Bit, &reg, 1, true)) {
      return false;
    }
    return read(address7Bit, outData, outLen, false);
  }

  bool writeReg(uint8_t address7Bit, uint8_t reg, uint8_t value) {
    uint8_t payload[2] = {reg, value};
    return write(address7Bit, payload, 2, false);
  }

 private:
  i2c_inst_t* inst_ = nullptr;
  bool initialized_ = false;
};

struct Lis2mdlAutoOrientationConfig {
  uint8_t i2cAddress7Bit = 0x1E;
  uint8_t outputDataRateBits = 0;

  bool enableTempComp = true;
  bool lowPowerMode = false;

  uint32_t sampleIntervalMs = 500;
  float smoothingAlpha = 0.03f;
  float headingOffsetDeg = 0.0f;
  float minHorizontalFieldLsb = 50.0f;

  uint16_t stableSamplesRequired = 18;
};

class Lis2mdlAutoOrientation {
 public:
  bool begin(I2cBus& bus, const Lis2mdlAutoOrientationConfig& config, uint8_t initialRotationSteps) {
    bus_ = &bus;
    config_ = config;

    if (config_.sampleIntervalMs == 0) {
      config_.sampleIntervalMs = 500;
    }
    if (!(config_.smoothingAlpha > 0.0f && config_.smoothingAlpha <= 1.0f)) {
      config_.smoothingAlpha = 0.03f;
    }
    if (config_.stableSamplesRequired == 0) {
      config_.stableSamplesRequired = 18;
    }

    rotationSteps_ = static_cast<uint8_t>(initialRotationSteps % 3u);
    candidateRotationSteps_ = rotationSteps_;

    uint8_t whoAmI = 0;
    if (!readReg(kRegWhoAmI, whoAmI) || whoAmI != kWhoAmIValue) {
      healthy_ = false;
      started_ = false;
      return false;
    }

    uint8_t cfgA = 0;
    cfgA |= static_cast<uint8_t>((config_.outputDataRateBits & 0x03u) << 2u);
    if (config_.lowPowerMode) {
      cfgA |= 0x10u;
    }
    if (config_.enableTempComp) {
      cfgA |= 0x80u;
    }

    const uint8_t cfgC = 0x10u;
    if (!writeReg(kRegCfgA, cfgA) || !writeReg(kRegCfgC, cfgC)) {
      healthy_ = false;
      started_ = false;
      return false;
    }

    healthy_ = true;
    started_ = true;
    sampleCount_ = 0;
    stableSampleCount_ = 0;
    lastSampleMs_ = millis32();
    return true;
  }

  bool poll(uint8_t* changedRotationSteps = nullptr) {
    if (!started_ || !healthy_) {
      return false;
    }

    const uint32_t nowMs = millis32();
    if ((nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
      return false;
    }
    lastSampleMs_ = nowMs;

    int16_t x = 0;
    int16_t y = 0;
    int16_t z = 0;
    (void)z;
    if (!readMagRaw(x, y, z)) {
      healthy_ = false;
      return false;
    }

    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    const float mag = std::sqrt((fx * fx) + (fy * fy));
    if (!(mag >= config_.minHorizontalFieldLsb)) {
      return false;
    }

    const float nx = fx / mag;
    const float ny = fy / mag;

    if (sampleCount_ == 0) {
      filtX_ = nx;
      filtY_ = ny;
    } else {
      const float alpha = config_.smoothingAlpha;
      filtX_ = ((1.0f - alpha) * filtX_) + (alpha * nx);
      filtY_ = ((1.0f - alpha) * filtY_) + (alpha * ny);

      const float n = std::sqrt((filtX_ * filtX_) + (filtY_ * filtY_));
      if (n > 1e-6f) {
        filtX_ /= n;
        filtY_ /= n;
      }
    }

    ++sampleCount_;

    headingDeg_ = wrap360(std::atan2(filtY_, filtX_) * kRadToDeg);
    const uint8_t candidate = headingToRotationSteps(headingDeg_);

    if (candidate != candidateRotationSteps_) {
      candidateRotationSteps_ = candidate;
      stableSampleCount_ = 1;
      return false;
    }

    if (stableSampleCount_ < 0xFFFFu) {
      ++stableSampleCount_;
    }

    if (candidateRotationSteps_ != rotationSteps_ &&
        stableSampleCount_ >= config_.stableSamplesRequired) {
      rotationSteps_ = candidateRotationSteps_;
      if (changedRotationSteps != nullptr) {
        *changedRotationSteps = rotationSteps_;
      }
      stableSampleCount_ = 0;
      return true;
    }

    return false;
  }

  bool healthy() const { return healthy_; }
  float headingDeg() const { return headingDeg_; }

 private:
  static constexpr uint8_t kRegWhoAmI = 0x4F;
  static constexpr uint8_t kRegCfgA = 0x60;
  static constexpr uint8_t kRegCfgC = 0x62;
  static constexpr uint8_t kRegOutXL = 0x68;
  static constexpr uint8_t kWhoAmIValue = 0x40;
  static constexpr float kRadToDeg = 57.29577951308232f;

  bool readReg(uint8_t reg, uint8_t& value) {
    if (bus_ == nullptr) {
      return false;
    }
    uint8_t buf = 0;
    if (!bus_->readReg(config_.i2cAddress7Bit, reg, &buf, 1)) {
      return false;
    }
    value = buf;
    return true;
  }

  bool writeReg(uint8_t reg, uint8_t value) {
    if (bus_ == nullptr) {
      return false;
    }
    return bus_->writeReg(config_.i2cAddress7Bit, reg, value);
  }

  bool readMagRaw(int16_t& x, int16_t& y, int16_t& z) {
    if (bus_ == nullptr) {
      return false;
    }
    uint8_t raw[6] = {0};
    if (!bus_->readReg(config_.i2cAddress7Bit, kRegOutXL, raw, sizeof(raw))) {
      return false;
    }
    x = static_cast<int16_t>((static_cast<uint16_t>(raw[1]) << 8u) | raw[0]);
    y = static_cast<int16_t>((static_cast<uint16_t>(raw[3]) << 8u) | raw[2]);
    z = static_cast<int16_t>((static_cast<uint16_t>(raw[5]) << 8u) | raw[4]);
    return true;
  }

  static float wrap360(float deg) {
    float out = std::fmod(deg, 360.0f);
    if (out < 0.0f) {
      out += 360.0f;
    }
    return out;
  }

  uint8_t headingToRotationSteps(float headingDeg) const {
    const float adjusted = wrap360(headingDeg - config_.headingOffsetDeg);
    const int sector = static_cast<int>(std::lround(adjusted / 120.0f));
    const int mod = sector % 3;
    return static_cast<uint8_t>((mod < 0) ? (mod + 3) : mod);
  }

  I2cBus* bus_ = nullptr;
  Lis2mdlAutoOrientationConfig config_ = {};
  bool started_ = false;
  bool healthy_ = false;
  uint8_t rotationSteps_ = 0;
  uint8_t candidateRotationSteps_ = 0;
  uint16_t stableSampleCount_ = 0;
  uint32_t lastSampleMs_ = 0;
  uint32_t sampleCount_ = 0;
  float filtX_ = 0.0f;
  float filtY_ = 0.0f;
  float headingDeg_ = 0.0f;
};

struct Lsm6TemperatureSourceConfig {
  uint8_t primaryAddress7Bit = 0x6A;
  uint8_t secondaryAddress7Bit = 0x6B;
  uint32_t sampleIntervalMs = 2000;
};

class Lsm6TemperatureSource : public IEnvironmentalSource {
 public:
  Lsm6TemperatureSource(I2cBus& bus, const Lsm6TemperatureSourceConfig& config)
      : bus_(&bus), config_(config) {}

  bool begin() override {
    started_ = false;
    healthy_ = false;
    hasReading_ = false;
    address7Bit_ = 0;
    sourceName_ = "lsm6";

    if (bus_ == nullptr || !bus_->initialized()) {
      return false;
    }
    if (config_.sampleIntervalMs == 0) {
      config_.sampleIntervalMs = 2000;
    }

    if (probeAddress(config_.primaryAddress7Bit)) {
      started_ = true;
      healthy_ = true;
      return true;
    }
    if (config_.secondaryAddress7Bit != config_.primaryAddress7Bit &&
        probeAddress(config_.secondaryAddress7Bit)) {
      started_ = true;
      healthy_ = true;
      return true;
    }
    return false;
  }

  bool read(EnvironmentalSample& outSample) override {
    if (!started_) {
      return false;
    }

    const uint32_t nowMs = millis32();
    if (hasReading_ && (nowMs - lastSampleMs_) < config_.sampleIntervalMs) {
      emitLast(outSample);
      return true;
    }

    int16_t rawTemperature = 0;
    if (!readTemperatureRaw(rawTemperature)) {
      healthy_ = false;
      if (hasReading_) {
        emitLast(outSample);
        return true;
      }
      return false;
    }

    const float temperatureC =
        (static_cast<float>(rawTemperature) / kTemperatureScaleLsbPerC) + kTemperatureOffsetC;
    if (!(std::isfinite(temperatureC) && temperatureC >= -55.0f && temperatureC <= 125.0f)) {
      if (hasReading_) {
        emitLast(outSample);
        return true;
      }
      return false;
    }

    lastTemperatureC_ = temperatureC;
    lastSampleMs_ = nowMs;
    hasReading_ = true;
    healthy_ = true;

    emitLast(outSample);
    return true;
  }

 private:
  static constexpr uint8_t kRegWhoAmI = 0x0F;
  static constexpr uint8_t kRegCtrl1Xl = 0x10;
  static constexpr uint8_t kRegOutTempL = 0x20;
  static constexpr uint8_t kWhoAmILsm6dsox = 0x6C;
  static constexpr uint8_t kWhoAmILsm6dsv16x = 0x70;
  static constexpr float kTemperatureScaleLsbPerC = 256.0f;
  static constexpr float kTemperatureOffsetC = 25.0f;

  bool probeAddress(uint8_t address7Bit) {
    if (address7Bit == 0) {
      return false;
    }
    address7Bit_ = address7Bit;

    uint8_t whoAmI = 0;
    if (!readReg(kRegWhoAmI, whoAmI)) {
      return false;
    }
    if (whoAmI == kWhoAmILsm6dsox) {
      sourceName_ = "lsm6dsox";
    } else if (whoAmI == kWhoAmILsm6dsv16x) {
      sourceName_ = "lsm6dsv16x";
    } else {
      return false;
    }

    uint8_t ctrl1Xl = 0;
    if (readReg(kRegCtrl1Xl, ctrl1Xl)) {
      const uint8_t odrMask = 0xF0u;
      if ((ctrl1Xl & odrMask) == 0) {
        const uint8_t odr12_5Hz = 0x10u;
        (void)writeReg(kRegCtrl1Xl, static_cast<uint8_t>((ctrl1Xl & 0x0Fu) | odr12_5Hz));
      }
    }
    return true;
  }

  bool readReg(uint8_t reg, uint8_t& value) {
    if (bus_ == nullptr || address7Bit_ == 0) {
      return false;
    }
    uint8_t out = 0;
    if (!bus_->readReg(address7Bit_, reg, &out, 1)) {
      return false;
    }
    value = out;
    return true;
  }

  bool writeReg(uint8_t reg, uint8_t value) {
    if (bus_ == nullptr || address7Bit_ == 0) {
      return false;
    }
    return bus_->writeReg(address7Bit_, reg, value);
  }

  bool readTemperatureRaw(int16_t& rawTemperature) {
    if (bus_ == nullptr || address7Bit_ == 0) {
      return false;
    }
    uint8_t raw[2] = {0};
    if (!bus_->readReg(address7Bit_, kRegOutTempL, raw, sizeof(raw))) {
      return false;
    }
    rawTemperature = static_cast<int16_t>((static_cast<uint16_t>(raw[1]) << 8u) | raw[0]);
    return true;
  }

  void emitLast(EnvironmentalSample& outSample) const {
    outSample = EnvironmentalSample();
    outSample.hasTemperatureC = hasReading_;
    outSample.temperatureC = lastTemperatureC_;
    outSample.temperatureSource = sourceName_;
  }

  I2cBus* bus_ = nullptr;
  Lsm6TemperatureSourceConfig config_ = {};
  bool started_ = false;
  bool healthy_ = false;
  bool hasReading_ = false;
  uint8_t address7Bit_ = 0;
  const char* sourceName_ = "lsm6";
  float lastTemperatureC_ = 0.0f;
  uint32_t lastSampleMs_ = 0;
};

class NodeClock {
 public:
  void begin(
      uint32_t sampleRateHz,
      size_t frameSamples,
      bool syncNtp,
      const char* ntpServer,
      long gmtOffsetSeconds,
      int daylightOffsetSeconds) {
    (void)ntpServer;
    (void)gmtOffsetSeconds;
    (void)daylightOffsetSeconds;

    if (syncNtp) {
      std::printf("[node] NTP sync requested but not enabled in this build; using monotonic clock\n");
    }

    streamStartNs_ = nowUtcNs();
    frameDurationNs_ = static_cast<uint64_t>(
        (static_cast<double>(frameSamples) * 1000000000.0) / static_cast<double>(sampleRateHz));
    frameIndex_ = 0;
  }

  uint64_t nextFrameStartNs() {
    const uint64_t out = streamStartNs_ + (frameIndex_ * frameDurationNs_);
    ++frameIndex_;
    return out;
  }

  uint64_t nowUtcNs() const {
    if (hasWallClock_) {
      return wallClockNs_ + (time_us_64() - wallClockSetUs_) * 1000ULL;
    }
    return static_cast<uint64_t>(time_us_64()) * 1000ULL;
  }

  void setUtcNs(uint64_t utcNs) {
    wallClockNs_ = utcNs;
    wallClockSetUs_ = time_us_64();
    hasWallClock_ = true;
    streamStartNs_ = utcNs;
    frameIndex_ = 0;
  }

 private:
  uint64_t streamStartNs_ = 0;
  uint64_t frameDurationNs_ = 0;
  uint64_t frameIndex_ = 0;

  bool hasWallClock_ = false;
  uint64_t wallClockNs_ = 0;
  uint64_t wallClockSetUs_ = 0;
};

struct SirithPicoTdmPins {
  uint8_t dataIn;
  uint8_t bclk;
  uint8_t ws;
};

struct SirithPicoTdmConfig {
  uint32_t sampleRateHz = 16000;
  size_t frameSamples = 1024;
  int32_t sampleShiftBits = 16;
  uint8_t tdmSlots = 4;
  uint8_t slotBits = 32;
  uint8_t outputChannelToSlot[4] = {0, 1, 3, 2};
  bool useSafeDriveStrength = true;
};

class SirithPicoTdmSource final : public IAudioSource {
 public:
  SirithPicoTdmSource(const SirithPicoTdmPins& pins, const SirithPicoTdmConfig& config)
      : pins_(pins), config_(config) {}

  ~SirithPicoTdmSource() override {
    deinitPioStateMachine();
  }

  bool begin() override {
    deinitPioStateMachine();
    if (!validateConfig()) {
      std::printf("[sirith-pico] invalid TDM config\n");
      return false;
    }
    if (!initPioStateMachine()) {
      return false;
    }
    initialized_ = true;
    return true;
  }

  uint32_t sampleRateHz() const override { return config_.sampleRateHz; }
  uint8_t channels() const override { return 4; }
  size_t frameSamples() const override { return config_.frameSamples; }

  bool readFrame(int16_t* interleavedOut, size_t samplesPerChannel) override {
    if (!initialized_ || interleavedOut == nullptr || samplesPerChannel != config_.frameSamples ||
        pio_ == nullptr || sm_ < 0) {
      return false;
    }

    PIO pio = reinterpret_cast<PIO>(pio_);
    for (size_t i = 0; i < config_.frameSamples; ++i) {
      int32_t slotWords[4] = {0, 0, 0, 0};
      for (uint8_t slot = 0; slot < 4; ++slot) {
        slotWords[slot] = static_cast<int32_t>(pio_sm_get_blocking(pio, static_cast<uint>(sm_)));
      }
      for (uint8_t channel = 0; channel < 4; ++channel) {
        const uint8_t slot = config_.outputChannelToSlot[channel];
        interleavedOut[(i * 4) + channel] = toPcm16(slotWords[slot]);
      }
    }
    return true;
  }

 private:
  inline static const uint16_t kSirithTdmMasterRxInstructions[] = {
      static_cast<uint16_t>(pio_encode_set(pio_x, 127) | pio_encode_sideset(2, 0x2)),
      static_cast<uint16_t>(pio_encode_in(pio_pins, 1) | pio_encode_sideset(2, 0x3)),
      static_cast<uint16_t>(pio_encode_jmp_x_dec(3) | pio_encode_sideset(2, 0x0)),
      static_cast<uint16_t>(pio_encode_in(pio_pins, 1) | pio_encode_sideset(2, 0x1)),
      static_cast<uint16_t>(pio_encode_jmp_x_dec(3) | pio_encode_sideset(2, 0x0)),
  };

  inline static const pio_program kSirithTdmMasterRxProgram = {
      kSirithTdmMasterRxInstructions,
      static_cast<uint>(sizeof(kSirithTdmMasterRxInstructions) / sizeof(kSirithTdmMasterRxInstructions[0])),
      -1,
  };

  static bool pinIsValid(uint8_t pin) {
    return pin < NUM_BANK0_GPIOS;
  }

  bool validateConfig() const {
    if (config_.sampleRateHz == 0 || config_.frameSamples == 0) {
      return false;
    }
    if (config_.tdmSlots != 4 || config_.slotBits != 32) {
      return false;
    }
    if (!pinIsValid(pins_.dataIn) || !pinIsValid(pins_.bclk) || !pinIsValid(pins_.ws)) {
      return false;
    }
    if (pins_.dataIn == pins_.bclk || pins_.dataIn == pins_.ws || pins_.bclk == pins_.ws) {
      return false;
    }
    if (pins_.ws != static_cast<uint8_t>(pins_.bclk + 1u)) {
      return false;
    }
    for (uint8_t i = 0; i < 4; ++i) {
      if (config_.outputChannelToSlot[i] >= config_.tdmSlots) {
        return false;
      }
    }
    return true;
  }

  bool initPioStateMachine() {
    PIO selectedPio = pio0;
    int selectedSm = pio_claim_unused_sm(selectedPio, false);
    if (selectedSm < 0) {
      selectedPio = pio1;
      selectedSm = pio_claim_unused_sm(selectedPio, false);
      if (selectedSm < 0) {
        std::printf("[sirith-pico] no free PIO state machine\n");
        return false;
      }
    }

    if (!pio_can_add_program(selectedPio, &kSirithTdmMasterRxProgram)) {
      pio_sm_unclaim(selectedPio, static_cast<uint>(selectedSm));
      std::printf("[sirith-pico] no room for PIO TDM program\n");
      return false;
    }

    const uint32_t bitsPerFrame =
        static_cast<uint32_t>(config_.slotBits) * static_cast<uint32_t>(config_.tdmSlots);
    const uint32_t bitRateHz = config_.sampleRateHz * bitsPerFrame;
    const uint32_t smCyclesPerFrame = (bitsPerFrame * 2u) + 1u;
    const float smClockHz = static_cast<float>(config_.sampleRateHz) * static_cast<float>(smCyclesPerFrame);
    const float clkSysHz = static_cast<float>(clock_get_hz(clk_sys));
    const float clkDiv = clkSysHz / smClockHz;

    if (!(clkDiv >= 1.0f && clkDiv <= 65535.0f)) {
      pio_sm_unclaim(selectedPio, static_cast<uint>(selectedSm));
      std::printf("[sirith-pico] unsupported sample rate for PIO clock divider\n");
      return false;
    }

    const uint programOffset = pio_add_program(selectedPio, &kSirithTdmMasterRxProgram);

    pio_gpio_init(selectedPio, pins_.dataIn);
    pio_gpio_init(selectedPio, pins_.bclk);
    pio_gpio_init(selectedPio, pins_.ws);

    gpio_set_dir(pins_.dataIn, GPIO_IN);
    gpio_pull_down(pins_.dataIn);

    gpio_set_dir(pins_.bclk, GPIO_OUT);
    gpio_set_dir(pins_.ws, GPIO_OUT);
    gpio_put(pins_.bclk, 0);
    gpio_put(pins_.ws, 0);

    if (config_.useSafeDriveStrength) {
#if defined(GPIO_DRIVE_STRENGTH_2MA)
      gpio_set_drive_strength(pins_.bclk, GPIO_DRIVE_STRENGTH_2MA);
      gpio_set_drive_strength(pins_.ws, GPIO_DRIVE_STRENGTH_2MA);
#endif
#if defined(GPIO_SLEW_RATE_SLOW)
      gpio_set_slew_rate(pins_.bclk, GPIO_SLEW_RATE_SLOW);
      gpio_set_slew_rate(pins_.ws, GPIO_SLEW_RATE_SLOW);
#endif
    }

    pio_sm_config smCfg = pio_get_default_sm_config();
    sm_config_set_wrap(&smCfg, programOffset + 0u, programOffset + 4u);
    sm_config_set_sideset(&smCfg, 2, false, false);
    sm_config_set_sideset_pins(&smCfg, pins_.bclk);
    sm_config_set_in_pins(&smCfg, pins_.dataIn);
    sm_config_set_in_shift(&smCfg, false, true, config_.slotBits);
    sm_config_set_fifo_join(&smCfg, PIO_FIFO_JOIN_RX);
    sm_config_set_clkdiv(&smCfg, clkDiv);

    pio_sm_set_consecutive_pindirs(selectedPio, static_cast<uint>(selectedSm), pins_.dataIn, 1, false);
    pio_sm_set_consecutive_pindirs(selectedPio, static_cast<uint>(selectedSm), pins_.bclk, 2, true);

    pio_sm_init(selectedPio, static_cast<uint>(selectedSm), programOffset, &smCfg);
    pio_sm_clear_fifos(selectedPio, static_cast<uint>(selectedSm));
    pio_sm_restart(selectedPio, static_cast<uint>(selectedSm));
    pio_sm_set_enabled(selectedPio, static_cast<uint>(selectedSm), true);

    pio_ = selectedPio;
    sm_ = selectedSm;
    offset_ = programOffset;
    programInstalled_ = true;

    std::printf(
        "[sirith-pico] TDM started pio=%d sm=%d sr=%lu bclk=%luHz ws=%luHz\n",
        (selectedPio == pio0) ? 0 : 1,
        selectedSm,
        static_cast<unsigned long>(config_.sampleRateHz),
        static_cast<unsigned long>(bitRateHz),
        static_cast<unsigned long>(config_.sampleRateHz));
    return true;
  }

  void deinitPioStateMachine() {
    if (!initialized_ && pio_ == nullptr) {
      return;
    }

    PIO pio = reinterpret_cast<PIO>(pio_);
    if (pio != nullptr && sm_ >= 0) {
      pio_sm_set_enabled(pio, static_cast<uint>(sm_), false);
      pio_sm_unclaim(pio, static_cast<uint>(sm_));
    }
    if (pio != nullptr && programInstalled_) {
      pio_remove_program(pio, &kSirithTdmMasterRxProgram, offset_);
    }

    gpio_put(pins_.bclk, 0);
    gpio_put(pins_.ws, 0);
    gpio_set_dir(pins_.bclk, GPIO_IN);
    gpio_set_dir(pins_.ws, GPIO_IN);

    pio_ = nullptr;
    sm_ = -1;
    offset_ = 0;
    programInstalled_ = false;
    initialized_ = false;
  }

  int16_t toPcm16(int32_t raw) const {
    int32_t shifted = raw >> config_.sampleShiftBits;
    if (shifted > 32767) {
      shifted = 32767;
    } else if (shifted < -32768) {
      shifted = -32768;
    }
    return static_cast<int16_t>(shifted);
  }

  SirithPicoTdmPins pins_ = {};
  SirithPicoTdmConfig config_ = {};
  bool initialized_ = false;
  void* pio_ = nullptr;
  int sm_ = -1;
  uint32_t offset_ = 0;
  bool programInstalled_ = false;
};

bool connectWiFiBlocking(const char* ssid, const char* password, uint32_t timeoutMs, uint32_t retryDelayMs = 500) {
  if (ssid == nullptr || ssid[0] == '\0') {
    return false;
  }

  if (isWiFiConnected()) {
    return true;
  }

  cyw43_arch_enable_sta_mode();

  const uint32_t auth = (password == nullptr || password[0] == '\0') ? CYW43_AUTH_OPEN : CYW43_AUTH_WPA2_AES_PSK;
  const int rc = cyw43_arch_wifi_connect_timeout_ms(ssid, password, auth, timeoutMs);
  if (rc == PICO_OK) {
    return true;
  }

  std::printf("[wifi] connect failed rc=%d\n", rc);
  sleep_ms(retryDelayMs);
  return false;
}

void ensureWiFiConnected(
    const char* ssid,
    const char* password,
    uint32_t timeoutMs,
    uint32_t checkIntervalMs = 5000) {
  static uint32_t lastCheckMs = 0;
  const uint32_t nowMs = millis32();
  if ((nowMs - lastCheckMs) < checkIntervalMs) {
    return;
  }
  lastCheckMs = nowMs;

  if (!isWiFiConnected()) {
    (void)connectWiFiBlocking(ssid, password, timeoutMs);
  }
}

class RawHttpPostClient {
 public:
  static PublishResult post(
      const std::string& host,
      uint16_t port,
      const std::string& path,
      const std::string& payload,
      uint32_t timeoutMs,
      bool keepResponseBody) {
    PublishResult result = {};
    result.ok = false;
    result.statusCode = -3;

    if (host.empty() || path.empty()) {
      return result;
    }

    State state = {};
    state.keepResponse = keepResponseBody;
    state.responseCap = keepResponseBody ? 0 : 2048;
    state.request = buildRequest(host, port, path, payload);

    if (!resolveHost(host, timeoutMs, state)) {
      result.statusCode = -3;
      return result;
    }

    state.pcb = tcp_new_ip_type(IP_GET_TYPE(&state.remoteAddr));
    if (state.pcb == nullptr) {
      result.statusCode = -3;
      return result;
    }

    tcp_arg(state.pcb, &state);
    tcp_recv(state.pcb, &RawHttpPostClient::onRecv);
    tcp_sent(state.pcb, &RawHttpPostClient::onSent);
    tcp_poll(state.pcb, &RawHttpPostClient::onPoll, 2);
    tcp_err(state.pcb, &RawHttpPostClient::onErr);

    const err_t connectErr = tcp_connect(state.pcb, &state.remoteAddr, port, &RawHttpPostClient::onConnected);
    if (connectErr != ERR_OK) {
      closeConnection(state);
      result.statusCode = -3;
      return result;
    }

    const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
    while (!state.done && !time_reached(deadline)) {
      cyw43_arch_poll();
      sleep_ms(1);
    }

    if (!state.done) {
      state.err = ERR_TIMEOUT;
      closeConnection(state);
      result.statusCode = -4;
    } else {
      parseStatusLine(state);
      if (state.statusCode > 0) {
        result.statusCode = state.statusCode;
      } else {
        result.statusCode = (state.err == ERR_OK) ? -4 : -3;
      }
    }

    if (keepResponseBody || result.statusCode < 200 || result.statusCode >= 300) {
      result.responseBody = state.response;
    }

    result.ok = (result.statusCode >= 200 && result.statusCode < 300);
    return result;
  }

 private:
  struct State {
    tcp_pcb* pcb = nullptr;
    ip_addr_t remoteAddr = {};

    bool dnsDone = false;
    bool dnsOk = false;

    bool done = false;
    err_t err = ERR_OK;

    std::string request;
    size_t requestOffset = 0;

    std::string response;
    bool keepResponse = false;
    size_t responseCap = 2048;
    int statusCode = -4;
  };

  static std::string buildRequest(
      const std::string& host,
      uint16_t port,
      const std::string& path,
      const std::string& payload) {
    std::string request;
    request.reserve(payload.size() + 256);

    request += "POST ";
    request += path;
    request += " HTTP/1.1\r\nHost: ";
    request += host;
    if (port != 80) {
      request += ':';
      request += std::to_string(port);
    }
    request += "\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: ";
    request += std::to_string(payload.size());
    request += "\r\n\r\n";
    request += payload;
    return request;
  }

  static void onDnsFound(const char* name, const ip_addr_t* ipaddr, void* arg) {
    (void)name;
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return;
    }
    state->dnsDone = true;
    if (ipaddr != nullptr) {
      state->remoteAddr = *ipaddr;
      state->dnsOk = true;
    } else {
      state->dnsOk = false;
    }
  }

  static bool resolveHost(const std::string& host, uint32_t timeoutMs, State& state) {
    if (ipaddr_aton(host.c_str(), &state.remoteAddr)) {
      return true;
    }

    const err_t dnsErr = dns_gethostbyname(host.c_str(), &state.remoteAddr, &RawHttpPostClient::onDnsFound, &state);
    if (dnsErr == ERR_OK) {
      state.dnsDone = true;
      state.dnsOk = true;
      return true;
    }
    if (dnsErr != ERR_INPROGRESS) {
      return false;
    }

    const absolute_time_t deadline = make_timeout_time_ms(timeoutMs);
    while (!state.dnsDone && !time_reached(deadline)) {
      cyw43_arch_poll();
      sleep_ms(1);
    }

    return state.dnsDone && state.dnsOk;
  }

  static void appendResponseChunk(State& state, const char* data, size_t len) {
    if (data == nullptr || len == 0) {
      return;
    }

    if (state.keepResponse) {
      state.response.append(data, len);
      return;
    }

    if (state.response.size() >= state.responseCap) {
      return;
    }
    size_t appendLen = len;
    const size_t remaining = state.responseCap - state.response.size();
    if (appendLen > remaining) {
      appendLen = remaining;
    }
    state.response.append(data, appendLen);
  }

  static void parseStatusLine(State& state) {
    if (state.statusCode > 0) {
      return;
    }

    const size_t lineEnd = state.response.find("\r\n");
    if (lineEnd == std::string::npos) {
      return;
    }

    const std::string line = state.response.substr(0, lineEnd);
    if (line.rfind("HTTP/", 0) != 0) {
      return;
    }

    const size_t firstSpace = line.find(' ');
    if (firstSpace == std::string::npos) {
      return;
    }
    const size_t secondSpace = line.find(' ', firstSpace + 1);
    const std::string codeText =
        (secondSpace == std::string::npos)
            ? line.substr(firstSpace + 1)
            : line.substr(firstSpace + 1, secondSpace - (firstSpace + 1));

    const long parsed = std::strtol(codeText.c_str(), nullptr, 10);
    if (parsed > 0 && parsed <= 999) {
      state.statusCode = static_cast<int>(parsed);
    }
  }

  static void closeConnection(State& state) {
    if (state.pcb == nullptr) {
      return;
    }

    tcp_arg(state.pcb, nullptr);
    tcp_recv(state.pcb, nullptr);
    tcp_sent(state.pcb, nullptr);
    tcp_poll(state.pcb, nullptr, 0);
    tcp_err(state.pcb, nullptr);

    const err_t closeErr = tcp_close(state.pcb);
    if (closeErr != ERR_OK) {
      tcp_abort(state.pcb);
    }
    state.pcb = nullptr;
  }

  static void flushTx(State& state, tcp_pcb* tpcb) {
    while (state.requestOffset < state.request.size()) {
      const uint16_t sndbuf = tcp_sndbuf(tpcb);
      if (sndbuf == 0) {
        break;
      }

      size_t remaining = state.request.size() - state.requestOffset;
      size_t chunk = remaining;
      if (chunk > sndbuf) {
        chunk = sndbuf;
      }
      if (chunk > 2048) {
        chunk = 2048;
      }

      const err_t writeErr = tcp_write(
          tpcb,
          state.request.data() + state.requestOffset,
          static_cast<u16_t>(chunk),
          TCP_WRITE_FLAG_COPY);
      if (writeErr == ERR_OK) {
        state.requestOffset += chunk;
      } else if (writeErr == ERR_MEM) {
        break;
      } else {
        state.err = writeErr;
        state.done = true;
        closeConnection(state);
        return;
      }
    }
    (void)tcp_output(tpcb);
  }

  static err_t onConnected(void* arg, tcp_pcb* tpcb, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_ARG;
    }
    if (err != ERR_OK) {
      state->err = err;
      state->done = true;
      closeConnection(*state);
      return err;
    }
    flushTx(*state, tpcb);
    return ERR_OK;
  }

  static err_t onSent(void* arg, tcp_pcb* tpcb, uint16_t len) {
    (void)len;
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_ARG;
    }
    flushTx(*state, tpcb);
    return ERR_OK;
  }

  static err_t onPoll(void* arg, tcp_pcb* tpcb) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return ERR_OK;
    }
    if (!state->done) {
      flushTx(*state, tpcb);
    }
    return ERR_OK;
  }

  static err_t onRecv(void* arg, tcp_pcb* tpcb, pbuf* p, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      if (p != nullptr) {
        pbuf_free(p);
      }
      return ERR_ARG;
    }

    if (err != ERR_OK) {
      state->err = err;
      state->done = true;
      if (p != nullptr) {
        pbuf_free(p);
      }
      closeConnection(*state);
      return err;
    }

    if (p == nullptr) {
      parseStatusLine(*state);
      state->done = true;
      closeConnection(*state);
      return ERR_OK;
    }

    for (pbuf* q = p; q != nullptr; q = q->next) {
      appendResponseChunk(*state, static_cast<const char*>(q->payload), q->len);
    }
    parseStatusLine(*state);
    tcp_recved(tpcb, p->tot_len);
    pbuf_free(p);
    return ERR_OK;
  }

  static void onErr(void* arg, err_t err) {
    State* state = static_cast<State*>(arg);
    if (state == nullptr) {
      return;
    }
    state->pcb = nullptr;
    state->err = err;
    state->done = true;
  }
};

class HttpFramePublisher {
 public:
  HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs)
      : timeoutMs_(timeoutMs) {
    endpointUrl_ = (serverBaseUrl != nullptr ? std::string(serverBaseUrl) : std::string()) +
                   (ingestPath != nullptr ? std::string(ingestPath) : std::string("/api/v1/ingest/frame"));
    endpointValid_ = parseEndpoint();
  }

  PublishResult publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody = false) {
    return publish(node, frame, nullptr, keepResponseBody);
  }

  PublishResult publish(
      const NodeDescriptor& node,
      const AudioFrame& frame,
      const EnvironmentalSample* environment,
      bool keepResponseBody = false) {
    PublishResult result = {};
    result.ok = false;
    result.statusCode = -1;

    if (!endpointValid_) {
      result.statusCode = -3;
      return result;
    }
    if (!isWiFiConnected()) {
      result.statusCode = -1;
      return result;
    }

    std::string payload;
    if (!buildIngestPayload(node, frame, environment, payload)) {
      result.statusCode = -2;
      return result;
    }

    return RawHttpPostClient::post(host_, port_, path_, payload, timeoutMs_, keepResponseBody);
  }

  const std::string& endpointUrl() const { return endpointUrl_; }

 private:
  static void trimAsciiWhitespace(std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
      ++start;
    }

    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
      --end;
    }

    if (start == 0 && end == s.size()) {
      return;
    }
    s = s.substr(start, end - start);
  }

  bool parseEndpoint() {
    host_.clear();
    path_.clear();
    port_ = 80;

    std::string url = endpointUrl_;
    trimAsciiWhitespace(url);

    constexpr const char* kHttpPrefix = "http://";
    if (url.rfind(kHttpPrefix, 0) != 0) {
      return false;
    }
    url.erase(0, std::strlen(kHttpPrefix));

    const size_t pathStart = url.find('/');
    const std::string hostPort = (pathStart == std::string::npos) ? url : url.substr(0, pathStart);
    path_ = (pathStart == std::string::npos) ? "/" : url.substr(pathStart);
    if (hostPort.empty()) {
      return false;
    }

    if (hostPort.front() == '[') {
      const size_t closeBracket = hostPort.find(']');
      if (closeBracket == std::string::npos) {
        return false;
      }
      host_ = hostPort.substr(1, closeBracket - 1);
      if (closeBracket + 1 < hostPort.size()) {
        if (hostPort[closeBracket + 1] != ':') {
          return false;
        }
        const std::string portText = hostPort.substr(closeBracket + 2);
        const long parsedPort = std::strtol(portText.c_str(), nullptr, 10);
        if (parsedPort <= 0 || parsedPort > 65535) {
          return false;
        }
        port_ = static_cast<uint16_t>(parsedPort);
      }
    } else {
      const size_t colonIndex = hostPort.rfind(':');
      if (colonIndex != std::string::npos) {
        host_ = hostPort.substr(0, colonIndex);
        const std::string portText = hostPort.substr(colonIndex + 1);
        const long parsedPort = std::strtol(portText.c_str(), nullptr, 10);
        if (parsedPort <= 0 || parsedPort > 65535) {
          return false;
        }
        port_ = static_cast<uint16_t>(parsedPort);
      } else {
        host_ = hostPort;
      }
    }

    if (host_.empty()) {
      return false;
    }
    if (path_.empty() || path_[0] != '/') {
      path_ = "/" + path_;
    }
    return true;
  }

  std::string endpointUrl_;
  std::string host_;
  std::string path_;
  uint16_t port_ = 80;
  bool endpointValid_ = false;
  uint32_t timeoutMs_ = 0;
};

struct RunnerStats {
  uint64_t framesCaptured = 0;
  uint64_t framesPublished = 0;
  uint64_t framesDropped = 0;
  uint64_t publishErrors = 0;
};

class NodeRunner {
 public:
  NodeRunner(
      const NodeDescriptor& descriptor,
      IAudioSource& audioSource,
      HttpFramePublisher& publisher,
      NodeClock& clock,
      uint32_t logEveryFrames = 100,
      IEnvironmentalSource* environmentalSource = nullptr)
      : descriptor_(descriptor),
        audioSource_(audioSource),
        publisher_(publisher),
        environmentalSource_(environmentalSource),
        clock_(clock),
        logEveryFrames_(logEveryFrames) {}

  bool begin(bool syncNtp, const char* ntpServer, long gmtOffsetSeconds, int daylightOffsetSeconds) {
    if (!audioSource_.begin()) {
      std::printf("[node] audio source init failed\n");
      return false;
    }

    frameBufferSamples_ = audioSource_.frameSamples() * static_cast<size_t>(audioSource_.channels());
    frameBuffer_ = new (std::nothrow) int16_t[frameBufferSamples_];
    if (frameBuffer_ == nullptr) {
      std::printf("[node] unable to allocate frame buffer\n");
      return false;
    }

    clock_.begin(
        audioSource_.sampleRateHz(),
        audioSource_.frameSamples(),
        syncNtp,
        ntpServer,
        gmtOffsetSeconds,
        daylightOffsetSeconds);

    if (environmentalSource_ != nullptr) {
      environmentalSourceReady_ = environmentalSource_->begin();
      if (!environmentalSourceReady_) {
        std::printf("[node] environmental source unavailable; continuing without telemetry\n");
      } else {
        std::printf("[node] environmental source enabled\n");
      }
    }

    std::printf(
        "[node] started id=%s channels=%u sample_rate=%lu frame_samples=%u endpoint=%s\n",
        descriptor_.id,
        static_cast<unsigned>(audioSource_.channels()),
        static_cast<unsigned long>(audioSource_.sampleRateHz()),
        static_cast<unsigned>(audioSource_.frameSamples()),
        publisher_.endpointUrl().c_str());
    return true;
  }

  void loopOnce() {
    if (frameBuffer_ == nullptr) {
      return;
    }

    const bool frameOk = audioSource_.readFrame(frameBuffer_, audioSource_.frameSamples());
    if (!frameOk) {
      ++stats_.framesDropped;
      sleep_ms(1);
      return;
    }

    AudioFrame frame = {
        clock_.nextFrameStartNs(),
        audioSource_.sampleRateHz(),
        audioSource_.channels(),
        sequence_,
        frameBuffer_,
        audioSource_.frameSamples(),
    };

    EnvironmentalSample environmental = {};
    const EnvironmentalSample* environmentalPtr = nullptr;
    if (environmentalSourceReady_ && environmentalSource_ != nullptr) {
      if (environmentalSource_->read(environmental)) {
        environmentalPtr = &environmental;
      }
    }

    const PublishResult result = publisher_.publish(descriptor_, frame, environmentalPtr, false);

    ++stats_.framesCaptured;
    if (result.ok) {
      ++stats_.framesPublished;
    } else {
      ++stats_.publishErrors;
    }

    ++sequence_;

    if (logEveryFrames_ > 0 && (stats_.framesCaptured % logEveryFrames_) == 0) {
      std::printf(
          "[node] frames=%" PRIu64 " published=%" PRIu64 " dropped=%" PRIu64
          " errors=%" PRIu64 " last_status=%d\n",
          stats_.framesCaptured,
          stats_.framesPublished,
          stats_.framesDropped,
          stats_.publishErrors,
          result.statusCode);
    }
  }

  const RunnerStats& stats() const { return stats_; }

 private:
  const NodeDescriptor& descriptor_;
  IAudioSource& audioSource_;
  HttpFramePublisher& publisher_;
  IEnvironmentalSource* environmentalSource_ = nullptr;
  bool environmentalSourceReady_ = false;
  NodeClock& clock_;

  int16_t* frameBuffer_ = nullptr;
  size_t frameBufferSamples_ = 0;
  uint64_t sequence_ = 0;
  uint32_t logEveryFrames_ = 100;
  RunnerStats stats_ = {};
};

}  // namespace mmpr

namespace {

constexpr uint8_t kMicCount = 4;

mmpr::Vec3 gSensorOffsetsOrdered[4] = {};
uint8_t gActiveBaseRotationSteps = static_cast<uint8_t>(nodecfg::kBasePlaneRotationSteps % 3u);

mmpr::I2cBus gI2c;
mmpr::Lis2mdlAutoOrientation gAutoOrientation;
bool gAutoOrientationEnabled = false;

mmpr::Lsm6TemperatureSourceConfig gImuTempConfig = {
    nodecfg::kImuI2cAddressPrimary7Bit,
    nodecfg::kImuI2cAddressSecondary7Bit,
    nodecfg::kImuTemperatureSampleIntervalMs,
};
mmpr::Lsm6TemperatureSource gImuTempSource(gI2c, gImuTempConfig);

uint8_t rotateBaseMic(uint8_t micIndex, uint8_t baseRotationSteps) {
  if (micIndex >= 3) {
    return micIndex;
  }
  const uint8_t rot = static_cast<uint8_t>(baseRotationSteps % 3u);
  return static_cast<uint8_t>((micIndex + rot) % 3u);
}

void buildOrderedOffsetsFromSlotMap(uint8_t baseRotationSteps) {
  for (uint8_t channel = 0; channel < kMicCount; ++channel) {
    uint8_t slot = nodecfg::kOutputChannelToSlot[channel];
    if (slot >= kMicCount) {
      std::printf(
          "[sirith-pico] invalid slot map outputChannelToSlot[%u]=%u, using slot 0\n",
          static_cast<unsigned>(channel),
          static_cast<unsigned>(slot));
      slot = 0;
    }

    uint8_t rawMicIndex = nodecfg::kSlotToPhysicalMic[slot];
    if (rawMicIndex >= kMicCount) {
      std::printf(
          "[sirith-pico] invalid mic map slotToPhysicalMic[%u]=%u, using mic 0\n",
          static_cast<unsigned>(slot),
          static_cast<unsigned>(rawMicIndex));
      rawMicIndex = 0;
    }

    const uint8_t rotatedMicIndex = rotateBaseMic(rawMicIndex, baseRotationSteps);
    gSensorOffsetsOrdered[channel] = mmpr::Vec3{
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][0],
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][1],
        nodecfg::kPhysicalSensorOffsetsM[rotatedMicIndex][2],
    };

    std::printf(
        "[sirith-pico] ch%u <- slot%u <- mic%u (rot=%u base=%u)\n",
        static_cast<unsigned>(channel),
        static_cast<unsigned>(slot + 1),
        static_cast<unsigned>(rawMicIndex),
        static_cast<unsigned>(rotatedMicIndex),
        static_cast<unsigned>(baseRotationSteps));
  }
}

mmpr::NodeDescriptor gNodeDescriptor = {
    nodecfg::kNodeId,
    mmpr::NodeType::kSirithTetra,
    {nodecfg::kNodePositionM[0], nodecfg::kNodePositionM[1], nodecfg::kNodePositionM[2]},
    gSensorOffsetsOrdered,
    sizeof(gSensorOffsetsOrdered) / sizeof(gSensorOffsetsOrdered[0]),
    nodecfg::kCapabilities,
    sizeof(nodecfg::kCapabilities) / sizeof(nodecfg::kCapabilities[0]),
    nodecfg::kHardwareName,
    MMPR_FW_VERSION,
};

mmpr::SirithPicoTdmPins gTdmPins = {
    nodecfg::kTdmDataPin,
    nodecfg::kTdmBclkPin,
    nodecfg::kTdmWsPin,
};

mmpr::SirithPicoTdmConfig gTdmConfig = {
    nodecfg::kAudioSampleRateHz,
    nodecfg::kAudioFrameSamples,
    nodecfg::kAudioSampleShiftBits,
    nodecfg::kAudioTdmSlots,
    nodecfg::kAudioSlotBits,
    {
        nodecfg::kOutputChannelToSlot[0],
        nodecfg::kOutputChannelToSlot[1],
        nodecfg::kOutputChannelToSlot[2],
        nodecfg::kOutputChannelToSlot[3],
    },
    nodecfg::kUseSafeDriveStrength,
};

mmpr::SirithPicoTdmSource gAudioSource(gTdmPins, gTdmConfig);
mmpr::HttpFramePublisher gPublisher(nodecfg::kServerBaseUrl, nodecfg::kIngestPath, nodecfg::kHttpTimeoutMs);
mmpr::NodeClock gClock;
mmpr::NodeRunner gRunner(
    gNodeDescriptor,
    gAudioSource,
    gPublisher,
    gClock,
    nodecfg::kLogEveryFrames,
    nodecfg::kEnableImuTemperature ? static_cast<mmpr::IEnvironmentalSource*>(&gImuTempSource) : nullptr);

void setupOptionalPeripherals() {
  const bool useI2c = nodecfg::kEnableCompassAutoOrientation || nodecfg::kEnableImuTemperature;
  bool i2cReady = false;
  if (useI2c) {
    i2cReady = gI2c.begin(i2c1, nodecfg::kI2cSdaPin, nodecfg::kI2cSclPin, nodecfg::kI2cBaudHz);
    if (!i2cReady) {
      std::printf("[sirith-pico] I2C init failed\n");
    }
  }

  if (nodecfg::kEnableGpsUart) {
    uart_init(uart0, nodecfg::kGpsUartBaud);
    gpio_set_function(nodecfg::kGpsTxPin, GPIO_FUNC_UART);
    gpio_set_function(nodecfg::kGpsRxPin, GPIO_FUNC_UART);

    gpio_init(nodecfg::kGpsPpsPin);
    gpio_set_dir(nodecfg::kGpsPpsPin, GPIO_IN);
    gpio_pull_down(nodecfg::kGpsPpsPin);
    std::printf("[sirith-pico] GPS UART enabled\n");
  }

  if (nodecfg::kEnableCompassAutoOrientation && i2cReady) {
    mmpr::Lis2mdlAutoOrientationConfig cfg = {};
    cfg.i2cAddress7Bit = nodecfg::kCompassI2cAddress7Bit;
    cfg.outputDataRateBits = nodecfg::kCompassOutputDataRateBits;
    cfg.sampleIntervalMs = nodecfg::kCompassSampleIntervalMs;
    cfg.smoothingAlpha = nodecfg::kCompassSmoothingAlpha;
    cfg.headingOffsetDeg = nodecfg::kCompassHeadingOffsetDeg;
    cfg.minHorizontalFieldLsb = nodecfg::kCompassMinHorizontalFieldLsb;
    cfg.stableSamplesRequired = nodecfg::kCompassStableSamplesRequired;

    gAutoOrientationEnabled = gAutoOrientation.begin(gI2c, cfg, gActiveBaseRotationSteps);
    if (gAutoOrientationEnabled) {
      std::printf("[sirith-pico] LIS2MDLTR auto-orientation enabled\n");
    } else {
      std::printf("[sirith-pico] LIS2MDLTR auto-orientation unavailable; manual rotation retained\n");
    }
  }

  if (nodecfg::kEnableImuTemperature) {
    std::printf("[sirith-pico] IMU temperature telemetry enabled (optional)\n");
  }
}

}  // namespace

int main() {
  stdio_init_all();
  sleep_ms(300);

  std::printf("[sirith-pico] booting\n");

  if (cyw43_arch_init()) {
    std::printf("[sirith-pico] fatal: Wi-Fi init failed\n");
    while (true) {
      sleep_ms(1000);
    }
  }

  buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
  setupOptionalPeripherals();

  const bool wifiConnected = mmpr::connectWiFiBlocking(
      nodecfg::kWifiSsid,
      nodecfg::kWifiPassword,
      nodecfg::kWiFiConnectTimeoutMs);

  if (!wifiConnected) {
    std::printf("[sirith-pico] WiFi not connected at startup, will keep retrying\n");
  } else {
    std::printf("[sirith-pico] WiFi connected\n");
  }

  const bool started = gRunner.begin(
      nodecfg::kEnableNtpSync,
      nodecfg::kNtpServer,
      nodecfg::kGmtOffsetSeconds,
      nodecfg::kDaylightOffsetSeconds);

  if (!started) {
    std::printf("[sirith-pico] fatal: runner failed to start\n");
    while (true) {
      cyw43_arch_poll();
      sleep_ms(1000);
    }
  }

  while (true) {
    if (gAutoOrientationEnabled) {
      uint8_t changedRotation = 0;
      if (gAutoOrientation.poll(&changedRotation)) {
        gActiveBaseRotationSteps = changedRotation;
        std::printf(
            "[sirith-pico] auto-rotation -> step=%u heading=%.1f deg\n",
            static_cast<unsigned>(gActiveBaseRotationSteps),
            static_cast<double>(gAutoOrientation.headingDeg()));
        buildOrderedOffsetsFromSlotMap(gActiveBaseRotationSteps);
      } else if (!gAutoOrientation.healthy()) {
        gAutoOrientationEnabled = false;
        std::printf("[sirith-pico] LIS2MDLTR read fault; holding current manual rotation\n");
      }
    }

    mmpr::ensureWiFiConnected(
        nodecfg::kWifiSsid,
        nodecfg::kWifiPassword,
        nodecfg::kWiFiConnectTimeoutMs);

    gRunner.loopOnce();
    cyw43_arch_poll();
  }
}
