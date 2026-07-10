// Host unit tests for mmpr::ClockHoldoverModel (pico-free, plain toolchain).
//
// Minimal assert-based harness — no external test framework so this builds with
// any C++17 compiler. Returns non-zero on first failure.

#include "mmpr/ClockHoldoverModel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* expr, const char* file, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s\n", file, line, expr);
  }
}

void checkNear(double a, double b, double tol, const char* expr, const char* file, int line) {
  ++g_checks;
  if (!(std::fabs(a - b) <= tol)) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s  (%.6f vs %.6f, tol %.6f)\n", file, line, expr, a, b, tol);
  }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) checkNear((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

using mmpr::LongTermFrequencyModel;
using mmpr::TempFrequencyModel;

// Feed one full 9-sample window of a constant ppm and return the last result.
mmpr::LtSampleResult feedConstWindow(LongTermFrequencyModel& m, double ppm) {
  mmpr::LtSampleResult r;
  for (int i = 0; i < LongTermFrequencyModel::kWindow; ++i) {
    r = m.addLockedSample(ppm);
  }
  return r;
}

// --- LTFM: seeding + validity cadence -----------------------------------------
void test_ltfm_seed_and_valid() {
  LongTermFrequencyModel m;
  CHECK(!m.seeded());
  CHECK(!m.valid());

  mmpr::LtSampleResult r = feedConstWindow(m, 2.0);
  CHECK(r.medianReady);
  CHECK(r.accepted);
  CHECK(m.seeded());
  CHECK_NEAR(m.ltPpm(), 2.0, 1e-9);
  CHECK(!m.valid());  // only 1 update so far

  // 4 more medians -> 5 total -> valid.
  for (int k = 0; k < 4; ++k) {
    feedConstWindow(m, 2.0);
  }
  CHECK(m.valid());
  CHECK(m.updateCount() == 5);
  CHECK_NEAR(m.ltPpm(), 2.0, 1e-6);
}

// --- LTFM: median rejects a minority of degraded samples ----------------------
void test_ltfm_median_outlier_rejection() {
  LongTermFrequencyModel m;
  // Window: five good (1.0) + four wild (1000.0). Median of 9 = the 5th order
  // statistic = 1.0, so the degraded pulses are rejected entirely.
  double vals[9] = {1.0, 1000.0, 1.0, -1000.0, 1.0, 1000.0, 1.0, -1000.0, 1.0};
  mmpr::LtSampleResult r;
  for (int i = 0; i < 9; ++i) {
    r = m.addLockedSample(vals[i]);
  }
  CHECK(r.medianReady);
  CHECK_NEAR(r.medianPpm, 1.0, 1e-9);
  CHECK_NEAR(m.ltPpm(), 1.0, 1e-9);
}

// --- LTFM: a single outlier median is gated out, not absorbed -----------------
void test_ltfm_outlier_gate() {
  LongTermFrequencyModel m;
  // Converge tightly at 0 ppm.
  for (int k = 0; k < 10; ++k) {
    feedConstWindow(m, 0.0);
  }
  CHECK(m.valid());
  const double before = m.ltPpm();
  // One median far outside the gate: must be skipped (not accepted).
  mmpr::LtSampleResult r = feedConstWindow(m, 50.0);
  CHECK(r.medianReady);
  CHECK(!r.accepted);
  CHECK(!r.reseeded);
  CHECK_NEAR(m.ltPpm(), before, 1e-9);
  CHECK(m.consecutiveOutliers() == 1);
}

// --- LTFM: learned frequency survives resetWindow() ---------------------------
void test_ltfm_survives_reset() {
  LongTermFrequencyModel m;
  for (int k = 0; k < 8; ++k) {
    feedConstWindow(m, 3.25);
  }
  CHECK(m.valid());
  const double ltBefore = m.ltPpm();
  const double varBefore = m.varPpm2();
  const uint32_t countBefore = m.updateCount();

  // Simulate lock loss mid-window: push a few samples then reset.
  m.addLockedSample(999.0);
  m.addLockedSample(999.0);
  CHECK(m.windowFill() == 2);
  m.resetWindow();

  CHECK(m.windowFill() == 0);
  CHECK_NEAR(m.ltPpm(), ltBefore, 1e-12);
  CHECK_NEAR(m.varPpm2(), varBefore, 1e-12);
  CHECK(m.updateCount() == countBefore);
  CHECK(m.valid());
}

// --- LTFM: sustained shift reseeds after 5 consecutive outlier medians --------
void test_ltfm_reseed_on_shift() {
  LongTermFrequencyModel m;
  for (int k = 0; k < 10; ++k) {
    feedConstWindow(m, 0.0);
  }
  CHECK(m.valid());

  // Genuine step to +40 ppm. First 4 outlier medians are skipped, the 5th
  // triggers a reseed that adopts the new frequency.
  mmpr::LtSampleResult r;
  for (int k = 0; k < 5; ++k) {
    r = feedConstWindow(m, 40.0);
  }
  CHECK(r.reseeded);
  CHECK(r.accepted);
  CHECK_NEAR(m.ltPpm(), 40.0, 1e-9);
  CHECK(!m.valid());  // re-characterizing after reseed
  CHECK(m.updateCount() == 1);
}

// --- TempFM: recovers a synthetic slope over a real temperature excursion -----
void test_tempfm_slope_recovery() {
  TempFrequencyModel t;
  const double trueSlope = -0.4;   // ppm/C
  const double intercept = 1.5;    // ppm at 25 C
  // Sweep temperature 15..35 C repeatedly; ppm = intercept + slope*(T-25).
  for (int pass = 0; pass < 60; ++pass) {
    for (int tc = 15; tc <= 35; ++tc) {
      const double ppm = intercept + trueSlope * (tc - 25.0);
      t.addSample(static_cast<double>(tc), ppm);
    }
  }
  CHECK(t.valid());
  CHECK_NEAR(t.slopePpmPerC(), trueSlope, 0.02);
  CHECK_NEAR(t.predictRelativePpm(10.0), trueSlope * 10.0, 0.2);
  CHECK(t.residRmsPpm() < 0.05);
}

// --- TempFM: insufficient temperature range -> not valid ----------------------
void test_tempfm_insufficient_range() {
  TempFrequencyModel t;
  // Temperature barely moves (sigma_T well under 0.5 C): slope not solvable.
  for (int i = 0; i < 5000; ++i) {
    const double tc = 25.0 + ((i % 2) ? 0.05 : -0.05);
    t.addSample(tc, 1.0);
  }
  CHECK(!t.valid());
  CHECK(t.varX() < TempFrequencyModel::kMinVarX);
}

// --- TempFM: physically implausible slope is rejected by valid() --------------
void test_tempfm_slope_clamp() {
  TempFrequencyModel t;
  const double crazySlope = 5.0;  // ppm/C, far above the 1.0 physical bound
  for (int pass = 0; pass < 60; ++pass) {
    for (int tc = 15; tc <= 35; ++tc) {
      const double ppm = crazySlope * (tc - 25.0);
      t.addSample(static_cast<double>(tc), ppm);
    }
  }
  // Enough range/n to solve, but the slope exceeds the physical bound.
  CHECK(t.varX() >= TempFrequencyModel::kMinVarX);
  CHECK(t.nEff() >= TempFrequencyModel::kMinNEff);
  CHECK(std::fabs(t.slopePpmPerC()) > TempFrequencyModel::kMaxSlopePpmPerC);
  CHECK(!t.valid());
}

// --- Budget: defaults reproduce exactly the legacy 60 s window ----------------
void test_budget_matches_legacy_60s() {
  const double driftUncertaintyPpm = 1.0;
  const uint64_t errorBudgetNs = 60000ULL;
  const uint64_t maxAgeUs = 900000000ULL;

  // At the 60 s boundary with a valid model at low sigma, drift floor = 1 ppm.
  const double drift = mmpr::effectiveHoldoverDriftPpm(driftUncertaintyPpm, 0.05);
  CHECK_NEAR(drift, 1.0, 1e-9);

  const uint64_t at60s = mmpr::predictedHoldoverErrorNs(60000000ULL, drift);
  CHECK(at60s == 60000ULL);
  CHECK(at60s <= errorBudgetNs);         // in budget at exactly 60 s
  CHECK(60000000ULL <= maxAgeUs);

  const uint64_t justOver = mmpr::predictedHoldoverErrorNs(60500000ULL, drift);
  CHECK(justOver > errorBudgetNs);       // out of budget past 60 s (60.5 s -> 60500 ns)

  // A TCXO rev (0.1 ppm) stretches the same budget to ~600 s.
  const uint64_t at600s = mmpr::predictedHoldoverErrorNs(600000000ULL, 0.1);
  CHECK(at600s == 60000ULL);
}

// --- Budget: 3-sigma widening dominates when the model is noisy ---------------
void test_budget_sigma_widening() {
  // sigma = 0.5 ppm -> 3 sigma = 1.5 ppm > 1.0 floor.
  const double drift = mmpr::effectiveHoldoverDriftPpm(1.0, 0.5);
  CHECK_NEAR(drift, 1.5, 1e-9);
  // Window shrinks accordingly: 60000 ns budget / 1.5 ppm ~= 40 s.
  const uint64_t at40s = mmpr::predictedHoldoverErrorNs(40000000ULL, drift);
  CHECK(at40s == 60000ULL);
}

}  // namespace

int main() {
  test_ltfm_seed_and_valid();
  test_ltfm_median_outlier_rejection();
  test_ltfm_outlier_gate();
  test_ltfm_survives_reset();
  test_ltfm_reseed_on_shift();
  test_tempfm_slope_recovery();
  test_tempfm_insufficient_range();
  test_tempfm_slope_clamp();
  test_budget_matches_legacy_60s();
  test_budget_sigma_widening();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
