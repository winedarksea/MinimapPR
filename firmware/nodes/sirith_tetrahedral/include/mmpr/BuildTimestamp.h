#pragma once

// Parses the compiler-provided __DATE__ / __TIME__ macros into a Unix-epoch
// nanosecond value at compile time.  This gives freerunning mode an approximate
// wall-clock anchor so that frame timestamps are "vaguely correct" even without
// GPS or NTP.  Accuracy is limited to:
//   - The difference between compile time and actual boot time (usually minutes
//     to hours for a freshly flashed device, potentially days/weeks for stale
//     firmware).
//   - RP2350 oscillator drift once the device is running (~100–200 ppm ROSC).
//
// The value is therefore never used when a better source (NTP, GPS) is
// available, and the backend should treat "build_timestamp" quality as
// coarse-grained wall time only.

#include <cstdint>

namespace mmpr {
namespace build_ts_detail {

// Gregorian civil date -> days since Unix epoch (1970-01-01).
// Howard Hinnant's algorithm: https://howardhinnant.github.io/date_algorithms.html
constexpr int64_t daysFromCivil(int y, int m, int d) {
  y -= (m <= 2) ? 1 : 0;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const int yoe = y - era * 400;
  const int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return static_cast<int64_t>(era) * 146097LL + doe - 719468LL;
}

// __DATE__ = "Mmm DD YYYY"  (day may have a leading space, e.g. " 5")
constexpr int buildMonth() {
  const char* d = __DATE__;
  switch (d[0]) {
    case 'J': return d[1] == 'a' ? 1 : (d[2] == 'n' ? 6 : 7);
    case 'F': return 2;
    case 'M': return d[2] == 'r' ? 3 : 5;
    case 'A': return d[1] == 'p' ? 4 : 8;
    case 'S': return 9;
    case 'O': return 10;
    case 'N': return 11;
    case 'D': return 12;
    default:  return 1;
  }
}

constexpr int buildDay() {
  const char* d = __DATE__;
  return (d[4] == ' ' ? 0 : (d[4] - '0') * 10) + (d[5] - '0');
}

constexpr int buildYear() {
  const char* d = __DATE__;
  return (d[7] - '0') * 1000 + (d[8] - '0') * 100 +
         (d[9] - '0') * 10  + (d[10] - '0');
}

// __TIME__ = "HH:MM:SS"
constexpr int buildHour()   { return (__TIME__[0] - '0') * 10 + (__TIME__[1] - '0'); }
constexpr int buildMinute() { return (__TIME__[3] - '0') * 10 + (__TIME__[4] - '0'); }
constexpr int buildSecond() { return (__TIME__[6] - '0') * 10 + (__TIME__[7] - '0'); }

}  // namespace build_ts_detail

// Unix-epoch nanoseconds at compile time.
inline constexpr uint64_t kBuildEpochNs = static_cast<uint64_t>(
    (build_ts_detail::daysFromCivil(
         build_ts_detail::buildYear(),
         build_ts_detail::buildMonth(),
         build_ts_detail::buildDay()) * 86400LL +
     static_cast<int64_t>(build_ts_detail::buildHour())   * 3600LL +
     static_cast<int64_t>(build_ts_detail::buildMinute()) * 60LL +
     static_cast<int64_t>(build_ts_detail::buildSecond())) * 1000000000LL);

}  // namespace mmpr
