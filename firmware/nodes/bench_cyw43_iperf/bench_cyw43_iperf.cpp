// bench_cyw43_iperf — sustained TCP goodput from a Pico 2 W over the CYW43439.
//
// Serial protocol (USB CDC, one command per line):
//   cfg <sys_khz> <pio_div_int> <ble 0|1>   persist link config, reboot to apply
//   run <ipv4> <port> <seconds> [copy 0|1] [rate_kbps] [inflight_cap_bytes]
//                                            blast TCP (paced if rate>0), report per second
//   info                                     print the active config
//
// The gSPI divider is read by cyw43_spi_init() once, and the sys clock is best
// set before any peripheral is up, so both are applied at boot from watchdog
// scratch registers (they survive a watchdog reboot) instead of at runtime.
//
// Output lines are machine-parsed by scripts/bench/cyw43_link_ceiling.py:
//   [tput] sec=<n> bytes=<acked> mbps=<x> rssi=<dBm>
//   [tput] done secs=<n> bytes=<total> mbps=<avg> min_sec_mbps=<x> rssi=<dBm> err=<lwip>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "hardware/watchdog.h"
#include "lwip/ip_addr.h"
#include "lwip/priv/tcp_priv.h"
#include "lwip/tcp.h"
#include "pico/cyw43_arch.h"
#include "pico/cyw43_driver.h"
#include "pico/stdlib.h"

#include "node_config.h"
#include "mmpr/BleRssiScanner.h"
#include "mmpr/WiFiSupport.h"

namespace {

constexpr uint32_t kScratchMagic = 0xB3C43A11u;
constexpr uint32_t kDefaultPioDivInt = 2;  // SDK default: 37.5 MHz gSPI at 150 MHz sys
constexpr uint32_t kRssiSampleIntervalMs = 2000;
// One MSS-multiple chunk per tcp_write keeps segment count low without
// holding more than the send buffer.
constexpr size_t kBlastChunkBytes = 4 * TCP_MSS;

struct LinkConfig {
  uint32_t sysKhz = BENCH_SYS_CLOCK_KHZ;
  uint32_t pioDivInt = kDefaultPioDivInt;
  bool bleScan = false;
};

LinkConfig gConfig;
mmpr::BleRssiScanner gBleScanner(nodecfg::kBleScanIntervalUnits, nodecfg::kBleScanWindowUnits);
uint8_t gBlastBuffer[kBlastChunkBytes];

struct BlastState {
  tcp_pcb* pcb = nullptr;
  bool connected = false;
  bool finished = false;
  bool copyWrites = false;
  err_t err = ERR_OK;
  uint64_t ackedBytes = 0;
  // Pacing (0 = saturate). Real audio is constant-rate, so a paced run at the
  // config's byte rate is the relevant pass/fail; saturation finds the ceiling.
  uint32_t rateKbps = 0;
  // In-flight cap (0 = TCP_SND_BUF). Bounds how many TX frames can sit in the
  // CYW43's packet pool, which it shares with RX (rxnobuf drops ACKs).
  uint32_t inflightCapBytes = 0;
  double allowanceBytes = 0.0;
  uint64_t lastRefillUs = 0;
};
BlastState gBlast;

// Every 1 ms wait inside the cyw43 driver (TX credit stall or ioctl wait) goes
// through this hook. During a credit stall the driver discards received
// Ethernet frames (cyw43_ll.c, cyw43_sdpcm_send_common), i.e. our TCP ACKs, so
// stall time is the firmware-visible proxy for "ACKs lost locally".
volatile uint32_t gDriverWaitCount = 0;

}  // namespace

extern "C" void __real_cyw43_await_background_or_timeout_us(uint32_t timeout_us);
extern "C" void __wrap_cyw43_await_background_or_timeout_us(uint32_t timeout_us) {
  ++gDriverWaitCount;
  __real_cyw43_await_background_or_timeout_us(timeout_us);
}

namespace {

void loadConfigFromScratch() {
  if (watchdog_hw->scratch[0] != kScratchMagic) {
    return;
  }
  gConfig.sysKhz = watchdog_hw->scratch[1];
  gConfig.pioDivInt = watchdog_hw->scratch[2];
  gConfig.bleScan = watchdog_hw->scratch[3] != 0;
}

void persistConfigAndReboot(const LinkConfig& config) {
  watchdog_hw->scratch[0] = kScratchMagic;
  watchdog_hw->scratch[1] = config.sysKhz;
  watchdog_hw->scratch[2] = config.pioDivInt;
  watchdog_hw->scratch[3] = config.bleScan ? 1u : 0u;
  std::printf("[bench] rebooting to apply cfg\n");
  sleep_ms(50);
  watchdog_reboot(0, 0, 10);
  while (true) {
    tight_loop_contents();
  }
}

int8_t sampleRssiDbm() {
  int32_t rssi = 0;
  if (cyw43_wifi_get_rssi(&cyw43_state, &rssi) != 0) {
    return 0;
  }
  return static_cast<int8_t>(rssi);
}

void printInfo() {
  std::printf(
      "[bench] info sys_khz=%lu pio_div=%lu ble=%u arch=%s ip=%s rssi=%d\n",
      static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000u),
      static_cast<unsigned long>(gConfig.pioDivInt),
      gConfig.bleScan ? 1u : 0u,
#if BENCH_ARCH_BACKGROUND
      "background",
#else
      "poll",
#endif
      ip4addr_ntoa(netif_ip4_addr(netif_list)),
      static_cast<int>(sampleRssiDbm()));
}

void fillSendBuffer(tcp_pcb* pcb) {
  if (pcb == nullptr || gBlast.finished) {
    return;
  }
  bool wrote = false;
  if (gBlast.rateKbps > 0) {
    const uint64_t nowUs = time_us_64();
    if (gBlast.lastRefillUs != 0) {
      gBlast.allowanceBytes +=
          static_cast<double>(nowUs - gBlast.lastRefillUs) * gBlast.rateKbps / 8000.0;
    }
    gBlast.lastRefillUs = nowUs;
    // Cap the burst at 20 ms of payload so pacing resembles a capture cadence.
    const double burstCapBytes = gBlast.rateKbps * 20.0 / 8.0;
    if (gBlast.allowanceBytes > burstCapBytes) {
      gBlast.allowanceBytes = burstCapBytes;
    }
  }
  while (true) {
    const uint16_t space = tcp_sndbuf(pcb);
    if (space < TCP_MSS || tcp_sndqueuelen(pcb) >= TCP_SND_QUEUELEN - 4) {
      break;
    }
    uint16_t chunk = space < kBlastChunkBytes ? space : static_cast<uint16_t>(kBlastChunkBytes);
    if (gBlast.inflightCapBytes > 0) {
      const uint32_t inflight = static_cast<uint32_t>(pcb->snd_lbb - pcb->lastack);
      if (inflight + TCP_MSS > gBlast.inflightCapBytes) {
        break;
      }
      const uint32_t room = gBlast.inflightCapBytes - inflight;
      if (chunk > room) {
        chunk = static_cast<uint16_t>(room);
      }
    }
    if (gBlast.rateKbps > 0) {
      if (gBlast.allowanceBytes < TCP_MSS) {
        break;
      }
      if (chunk > gBlast.allowanceBytes) {
        chunk = static_cast<uint16_t>(gBlast.allowanceBytes);
      }
    }
    const err_t writeErr = tcp_write(
        pcb, gBlastBuffer, chunk, gBlast.copyWrites ? TCP_WRITE_FLAG_COPY : 0);
    if (writeErr != ERR_OK) {
      break;
    }
    if (gBlast.rateKbps > 0) {
      gBlast.allowanceBytes -= chunk;
    }
    wrote = true;
  }
  if (wrote) {
    tcp_output(pcb);
  }
}

err_t onBlastSent(void*, tcp_pcb* pcb, u16_t len) {
  gBlast.ackedBytes += len;
  fillSendBuffer(pcb);
  return ERR_OK;
}

err_t onBlastConnected(void*, tcp_pcb* pcb, err_t err) {
  if (err != ERR_OK) {
    gBlast.err = err;
    gBlast.finished = true;
    return err;
  }
  gBlast.connected = true;
  tcp_nagle_disable(pcb);
  fillSendBuffer(pcb);
  return ERR_OK;
}

err_t onBlastRecv(void*, tcp_pcb* pcb, pbuf* p, err_t) {
  if (p == nullptr) {
    gBlast.finished = true;  // receiver closed
    return ERR_OK;
  }
  tcp_recved(pcb, p->tot_len);
  pbuf_free(p);
  return ERR_OK;
}

void onBlastErr(void*, err_t err) {
  gBlast.pcb = nullptr;  // pcb already freed by lwIP
  gBlast.err = err;
  gBlast.finished = true;
}

void serviceNetwork() {
#if !BENCH_ARCH_BACKGROUND
  cyw43_arch_poll();
#endif
}

void runBlast(const char* ipText, uint16_t port, uint32_t seconds, bool copyWrites, uint32_t rateKbps,
              uint32_t inflightCapBytes) {
  ip_addr_t remote = {};
  if (!ipaddr_aton(ipText, &remote)) {
    std::printf("[tput] error bad_ip=%s\n", ipText);
    return;
  }
  gBlast = {};
  gBlast.copyWrites = copyWrites;
  gBlast.rateKbps = rateKbps;
  gBlast.inflightCapBytes = inflightCapBytes;

  cyw43_arch_lwip_begin();
  gBlast.pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
  tcp_sent(gBlast.pcb, onBlastSent);
  tcp_recv(gBlast.pcb, onBlastRecv);
  tcp_err(gBlast.pcb, onBlastErr);
  const err_t connectErr = tcp_connect(gBlast.pcb, &remote, port, onBlastConnected);
  cyw43_arch_lwip_end();
  if (connectErr != ERR_OK) {
    std::printf("[tput] error connect=%d\n", static_cast<int>(connectErr));
    return;
  }

  const uint32_t connectDeadlineMs = to_ms_since_boot(get_absolute_time()) + 5000;
  while (!gBlast.connected && !gBlast.finished &&
         to_ms_since_boot(get_absolute_time()) < connectDeadlineMs) {
    serviceNetwork();
  }
  if (!gBlast.connected) {
    std::printf("[tput] error connect_timeout err=%d\n", static_cast<int>(gBlast.err));
    if (gBlast.pcb != nullptr) {
      cyw43_arch_lwip_begin();
      tcp_abort(gBlast.pcb);
      cyw43_arch_lwip_end();
    }
    return;
  }

  const uint32_t startMs = to_ms_since_boot(get_absolute_time());
  uint32_t nextReportMs = startMs + 1000;
  uint32_t nextRssiMs = startMs;
  uint64_t lastReportBytes = 0;
  uint32_t second = 0;
  int8_t rssi = 0;
  double minSecondMbps = 1e9;
  // Loss-recovery signatures, sampled every iteration: an RTO collapses cwnd
  // to one MSS and bumps nrtx; fast retransmit sets TF_INFR. Whole-second
  // stalls with rto_events > 0 mean air loss recovered by timeout, not a
  // bandwidth ceiling.
  uint32_t rtoEventsTotal = 0;
  uint32_t fastRexmitTotal = 0;
  uint32_t secondRtoEvents = 0;
  uint32_t secondFastRexmit = 0;
  uint8_t lastNrtx = 0;
  bool lastInFastRecovery = false;
  uint32_t secondMaxStallMs = 0;
  uint64_t lastAckedBytes = 0;
  uint32_t lastAckProgressMs = startMs;
  const uint32_t driverWaitsAtStart = gDriverWaitCount;
  uint32_t driverWaitsAtSecond = gDriverWaitCount;
  while (second < seconds && !gBlast.finished) {
    serviceNetwork();
    cyw43_arch_lwip_begin();
    fillSendBuffer(gBlast.pcb);
    if (gBlast.pcb != nullptr) {
      const uint8_t nrtx = gBlast.pcb->nrtx;
      if (nrtx > lastNrtx) {
        rtoEventsTotal += nrtx - lastNrtx;
        secondRtoEvents += nrtx - lastNrtx;
      }
      lastNrtx = nrtx;
      const bool inFastRecovery = (gBlast.pcb->flags & TF_INFR) != 0;
      if (inFastRecovery && !lastInFastRecovery) {
        ++fastRexmitTotal;
        ++secondFastRexmit;
      }
      lastInFastRecovery = inFastRecovery;
    }
    cyw43_arch_lwip_end();
    {
      const uint32_t nowMsStall = to_ms_since_boot(get_absolute_time());
      if (gBlast.ackedBytes != lastAckedBytes) {
        lastAckedBytes = gBlast.ackedBytes;
        lastAckProgressMs = nowMsStall;
      } else if (nowMsStall - lastAckProgressMs > secondMaxStallMs) {
        secondMaxStallMs = nowMsStall - lastAckProgressMs;
      }
    }
    gBleScanner.poll();
    const uint32_t nowMs = to_ms_since_boot(get_absolute_time());
    if (static_cast<int32_t>(nowMs - nextRssiMs) >= 0) {
      // Blocking SPI ioctl; sampled sparsely so it costs < 0.1% of link time.
      rssi = sampleRssiDbm();
      nextRssiMs = nowMs + kRssiSampleIntervalMs;
    }
    if (static_cast<int32_t>(nowMs - nextReportMs) >= 0) {
      ++second;
      const uint64_t bytes = gBlast.ackedBytes - lastReportBytes;
      lastReportBytes = gBlast.ackedBytes;
      const double mbps = static_cast<double>(bytes) * 8.0 / 1e6;
      if (mbps < minSecondMbps) {
        minSecondMbps = mbps;
      }
      uint32_t cwnd = 0;
      int rtoMs = 0;
      if (gBlast.pcb != nullptr) {
        cwnd = gBlast.pcb->cwnd;
        rtoMs = gBlast.pcb->rto * TCP_SLOW_INTERVAL;
      }
      std::printf(
          "[tput] sec=%lu bytes=%llu mbps=%.3f rssi=%d rto_ev=%lu fast_rx=%lu max_stall_ms=%lu cwnd=%lu rto_ms=%d drv_waits=%lu\n",
          static_cast<unsigned long>(second),
          static_cast<unsigned long long>(bytes),
          mbps,
          static_cast<int>(rssi),
          static_cast<unsigned long>(secondRtoEvents),
          static_cast<unsigned long>(secondFastRexmit),
          static_cast<unsigned long>(secondMaxStallMs),
          static_cast<unsigned long>(cwnd),
          rtoMs,
          static_cast<unsigned long>(gDriverWaitCount - driverWaitsAtSecond));
      driverWaitsAtSecond = gDriverWaitCount;
      secondRtoEvents = 0;
      secondFastRexmit = 0;
      secondMaxStallMs = 0;
      nextReportMs += 1000;
    }
  }
  const uint32_t elapsedMs = to_ms_since_boot(get_absolute_time()) - startMs;
  cyw43_arch_lwip_begin();
  if (gBlast.pcb != nullptr) {
    tcp_arg(gBlast.pcb, nullptr);
    tcp_sent(gBlast.pcb, nullptr);
    tcp_recv(gBlast.pcb, nullptr);
    tcp_err(gBlast.pcb, nullptr);
    if (tcp_close(gBlast.pcb) != ERR_OK) {
      tcp_abort(gBlast.pcb);
    }
    gBlast.pcb = nullptr;
  }
  cyw43_arch_lwip_end();
  std::printf(
      "[tput] done secs=%lu bytes=%llu mbps=%.3f min_sec_mbps=%.3f rssi=%d err=%d sys_khz=%lu pio_div=%lu ble=%u copy=%u rto_ev=%lu fast_rx=%lu rate_kbps=%lu drv_waits=%lu cap=%lu\n",
      static_cast<unsigned long>(second),
      static_cast<unsigned long long>(gBlast.ackedBytes),
      elapsedMs > 0 ? static_cast<double>(gBlast.ackedBytes) * 8.0 / (elapsedMs * 1000.0) : 0.0,
      second > 0 ? minSecondMbps : 0.0,
      static_cast<int>(rssi),
      static_cast<int>(gBlast.err),
      static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000u),
      static_cast<unsigned long>(gConfig.pioDivInt),
      gConfig.bleScan ? 1u : 0u,
      copyWrites ? 1u : 0u,
      static_cast<unsigned long>(rtoEventsTotal),
      static_cast<unsigned long>(fastRexmitTotal),
      static_cast<unsigned long>(rateKbps),
      static_cast<unsigned long>(gDriverWaitCount - driverWaitsAtStart),
      static_cast<unsigned long>(inflightCapBytes));
}

// Broadcom wl ioctl passthrough: cyw43_ioctl encodes (WLC_cmd << 1) | is_set.
// Used for MAC retry limits (WLC_GET/SET_SRL 31/32, LRL 33/34) and the
// current TX rate (WLC_GET_RATE 12, 500 kbit/s units).
int wlIoctlU32(uint32_t wlcCmd, bool isSet, uint32_t* value) {
  uint8_t buf[4];
  std::memcpy(buf, value, 4);
  const int rc = cyw43_ioctl(&cyw43_state, (wlcCmd << 1) | (isSet ? 1u : 0u), 4, buf, CYW43_ITF_STA);
  std::memcpy(value, buf, 4);
  return rc;
}

int wlIovarU32(const char* name, bool isSet, uint32_t* value) {
  constexpr uint32_t kWlcGetVar = 262;
  constexpr uint32_t kWlcSetVar = 263;
  uint8_t buf[64] = {};
  const size_t nameLen = std::strlen(name) + 1;
  if (nameLen + 4 > sizeof(buf)) {
    return -1;
  }
  std::memcpy(buf, name, nameLen);
  std::memcpy(buf + nameLen, value, 4);
  const int rc = cyw43_ioctl(
      &cyw43_state, ((isSet ? kWlcSetVar : kWlcGetVar) << 1) | (isSet ? 1u : 0u),
      isSet ? nameLen + 4 : sizeof(buf), buf, CYW43_ITF_STA);
  if (!isSet) {
    std::memcpy(value, buf, 4);
  }
  return rc;
}

// Raw dump of the firmware's wl_cnt_t ("counters" iovar) as u32 words; the
// layout is firmware-version dependent, so it is decoded host-side.
void dumpWlCounters() {
  constexpr uint32_t kWlcGetVar = 262;
  static uint8_t buf[1024];
  std::memset(buf, 0, sizeof(buf));
  std::memcpy(buf, "counters", 9);
  const int rc = cyw43_ioctl(&cyw43_state, kWlcGetVar << 1, sizeof(buf), buf, CYW43_ITF_STA);
  std::printf("[cnt] rc=%d words=", rc);
  for (size_t i = 0; i < sizeof(buf) / 4; ++i) {
    uint32_t word;
    std::memcpy(&word, buf + i * 4, 4);
    std::printf("%lx,", static_cast<unsigned long>(word));
  }
  std::printf("\n");
}

void handleCommand(char* line) {
  char* save = nullptr;
  const char* verb = strtok_r(line, " \t", &save);
  if (verb == nullptr) {
    return;
  }
  if (std::strcmp(verb, "info") == 0) {
    printInfo();
  } else if (std::strcmp(verb, "counters") == 0) {
    dumpWlCounters();
  } else if (std::strcmp(verb, "cfg") == 0) {
    const char* sysText = strtok_r(nullptr, " \t", &save);
    const char* divText = strtok_r(nullptr, " \t", &save);
    const char* bleText = strtok_r(nullptr, " \t", &save);
    if (sysText == nullptr || divText == nullptr || bleText == nullptr) {
      std::printf("[bench] usage: cfg <sys_khz> <pio_div_int> <ble>\n");
      return;
    }
    LinkConfig next;
    next.sysKhz = static_cast<uint32_t>(std::strtoul(sysText, nullptr, 10));
    next.pioDivInt = static_cast<uint32_t>(std::strtoul(divText, nullptr, 10));
    next.bleScan = std::atoi(bleText) != 0;
    persistConfigAndReboot(next);
  } else if (std::strcmp(verb, "wl") == 0 || std::strcmp(verb, "iovar") == 0) {
    // wl <get|set> <wlc_cmd> [value]   /   iovar <get|set> <name> [value]
    const char* op = strtok_r(nullptr, " \t", &save);
    const char* target = strtok_r(nullptr, " \t", &save);
    const char* valueText = strtok_r(nullptr, " \t", &save);
    if (op == nullptr || target == nullptr) {
      std::printf("[bench] usage: wl|iovar get|set <cmd|name> [value]\n");
      return;
    }
    const bool isSet = std::strcmp(op, "set") == 0;
    uint32_t value = valueText != nullptr ? static_cast<uint32_t>(std::strtoul(valueText, nullptr, 0)) : 0u;
    const int rc = std::strcmp(verb, "wl") == 0
        ? wlIoctlU32(static_cast<uint32_t>(std::strtoul(target, nullptr, 0)), isSet, &value)
        : wlIovarU32(target, isSet, &value);
    std::printf("[bench] %s %s %s rc=%d value=%lu\n", verb, op, target, rc, static_cast<unsigned long>(value));
  } else if (std::strcmp(verb, "run") == 0) {
    const char* ipText = strtok_r(nullptr, " \t", &save);
    const char* portText = strtok_r(nullptr, " \t", &save);
    const char* secText = strtok_r(nullptr, " \t", &save);
    const char* copyText = strtok_r(nullptr, " \t", &save);
    const char* rateText = strtok_r(nullptr, " \t", &save);
    const char* capText = strtok_r(nullptr, " \t", &save);
    if (ipText == nullptr || portText == nullptr || secText == nullptr) {
      std::printf("[bench] usage: run <ip> <port> <seconds> [copy]\n");
      return;
    }
    runBlast(ipText,
             static_cast<uint16_t>(std::atoi(portText)),
             static_cast<uint32_t>(std::strtoul(secText, nullptr, 10)),
             copyText != nullptr && std::atoi(copyText) != 0,
             rateText != nullptr ? static_cast<uint32_t>(std::strtoul(rateText, nullptr, 10)) : 0u,
             capText != nullptr ? static_cast<uint32_t>(std::strtoul(capText, nullptr, 10)) : 0u);
  } else {
    std::printf("[bench] unknown command '%s'\n", verb);
  }
}

}  // namespace

int main() {
  loadConfigFromScratch();
  if (gConfig.sysKhz > 150000) {
    // RP2350 is specified to 150 MHz at the default 1.10 V; give overclock headroom.
    vreg_set_voltage(VREG_VOLTAGE_1_20);
    sleep_ms(2);
  }
  if (!set_sys_clock_khz(gConfig.sysKhz, false)) {
    gConfig.sysKhz = BENCH_SYS_CLOCK_KHZ;
  }
  stdio_init_all();
  for (int i = 0; i < 20 && !stdio_usb_connected(); ++i) {
    sleep_ms(100);
  }
  for (size_t i = 0; i < sizeof(gBlastBuffer); ++i) {
    gBlastBuffer[i] = static_cast<uint8_t>(i * 131u);
  }

  cyw43_set_pio_clkdiv_int_frac8(gConfig.pioDivInt, 0);
  if (cyw43_arch_init()) {
    std::printf("[bench] fatal: cyw43 init failed\n");
    while (true) {
      sleep_ms(1000);
    }
  }
  if (gConfig.bleScan && !gBleScanner.begin()) {
    std::printf("[bench] warning: BLE scanner init failed\n");
  }
  while (!mmpr::connectWiFiBlocking(nodecfg::kWifiSsid, nodecfg::kWifiPassword, 15000)) {
  }
  printInfo();
  std::printf("[bench] ready\n");

  char line[128];
  size_t lineLength = 0;
  while (true) {
    serviceNetwork();
    gBleScanner.poll();
    const int ch = getchar_timeout_us(1000);
    if (ch == PICO_ERROR_TIMEOUT) {
      continue;
    }
    if (ch == '\n' || ch == '\r') {
      line[lineLength] = '\0';
      if (lineLength > 0) {
        handleCommand(line);
      }
      lineLength = 0;
    } else if (lineLength + 1 < sizeof(line)) {
      line[lineLength++] = static_cast<char>(ch);
    }
  }
}
