#pragma once

// Pure-logic clock holdover models for GPS-jamming tolerance.
//
// This header is deliberately free of any pico-sdk / printf dependency so it can
// be unit-tested with a plain host toolchain (see tests/host/). Callers own all
// logging. All state is fixed-size scalars plus a 9-slot array — no dynamic
// allocation, no locks. All math runs at <=1 Hz cadences from main-loop context;
// nothing here is called from an IRQ or the per-audio-frame path.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mmpr {

// Result of feeding one fully-locked instantaneous ppm sample into the
// long-term frequency model. A median is only produced once every 9 samples.
struct LtSampleResult {
  bool medianReady = false;  // a full 9-sample window produced a median this call
  double medianPpm = 0.0;    // the median value (valid iff medianReady)
  bool accepted = false;     // the median updated ltPpm (seed, EWMA step, or reseed)
  bool reseeded = false;     // 5 consecutive outliers forced a reseed (freq shift)
};

// Median-of-9 prefilter feeding a slow, outlier-gated EWMA of the crystal
// frequency (ppm) plus a variance EWMA. The learned ltPpm / variance / update
// count survive lock loss, rebase, and re-acquisition — only the partial
// sample window is discarded on those events (resetWindow()). This is the fix
// for the legacy "reset learned frequency on every dropout" defect.
class LongTermFrequencyModel {
 public:
  static constexpr int kWindow = 9;
  static constexpr uint32_t kMinUpdates = 5;       // ~45 s of lock before valid()
  static constexpr uint32_t kReseedOutliers = 5;   // ~45 s of outliers => genuine shift
  static constexpr double kSeedVarPpm2 = 0.25;     // (0.5 ppm)^2 initial variance
  static constexpr double kEwmaShift = 64.0;       // tau ~10 min at 1 median / 9 s

  // Feed one instantaneous ppm reading taken while fully GPS-locked.
  LtSampleResult addLockedSample(double ppmInst) {
    LtSampleResult result;
    if (windowCount_ < kWindow) {
      window_[windowCount_++] = ppmInst;
    }
    if (windowCount_ < kWindow) {
      return result;
    }

    double tmp[kWindow];
    for (int i = 0; i < kWindow; ++i) {
      tmp[i] = window_[i];
    }
    std::nth_element(tmp, tmp + kWindow / 2, tmp + kWindow);
    const double m = tmp[kWindow / 2];
    windowCount_ = 0;

    result.medianReady = true;
    result.medianPpm = m;

    if (!seeded_) {
      seeded_ = true;
      ltPpm_ = m;
      ltVarPpm2_ = kSeedVarPpm2;
      updateCount_ = 1;
      consecutiveOutliers_ = 0;
      result.accepted = true;
      return result;
    }

    const double dev = m - ltPpm_;
    const double gate = std::max(0.5, 5.0 * std::sqrt(ltVarPpm2_));
    if (std::fabs(dev) > gate) {
      ++consecutiveOutliers_;
      if (consecutiveOutliers_ >= kReseedOutliers) {
        // Sustained disagreement is a genuine frequency shift, not a glitch:
        // adopt the new median and restart characterization.
        ltPpm_ = m;
        ltVarPpm2_ = kSeedVarPpm2;
        updateCount_ = 1;
        consecutiveOutliers_ = 0;
        result.reseeded = true;
        result.accepted = true;
      }
      return result;
    }

    consecutiveOutliers_ = 0;
    ltPpm_ += dev / kEwmaShift;
    ltVarPpm2_ += (dev * dev - ltVarPpm2_) / kEwmaShift;
    ++updateCount_;
    result.accepted = true;
    return result;
  }

  // Discard only the in-progress sample window. Learned frequency state is
  // preserved so a dropout/rebase/re-acquisition never throws it away.
  void resetWindow() { windowCount_ = 0; }

  bool valid() const { return seeded_ && updateCount_ >= kMinUpdates; }
  bool seeded() const { return seeded_; }
  double ltPpm() const { return ltPpm_; }
  double varPpm2() const { return ltVarPpm2_; }
  double sigmaPpm() const { return std::sqrt(ltVarPpm2_); }
  uint32_t updateCount() const { return updateCount_; }
  uint32_t consecutiveOutliers() const { return consecutiveOutliers_; }
  int windowFill() const { return windowCount_; }

 private:
  double window_[kWindow] = {};
  int windowCount_ = 0;
  bool seeded_ = false;
  double ltPpm_ = 0.0;
  double ltVarPpm2_ = kSeedVarPpm2;
  uint32_t updateCount_ = 0;
  uint32_t consecutiveOutliers_ = 0;
};

// Shadow-mode exponentially-forgetting linear least squares of ppm on
// (T - 25 C). Learns a temperature->frequency slope while locked. Compensation
// during holdover is RELATIVE (slope x delta-T from holdover-entry temperature)
// so it is immune to intercept error. Applying it in holdover is gated behind a
// config flag (default OFF); this model only ever learns + logs until then.
class TempFrequencyModel {
 public:
  static constexpr double kRefTempC = 25.0;
  static constexpr double kLambda = 0.9995;     // forgetting factor, ~2000-sample memory
  static constexpr double kMinVarX = 0.25;      // require sigma_T >= 0.5 C to solve
  static constexpr double kMaxSlopePpmPerC = 1.0;  // physical bound; larger = confounded
  static constexpr double kMinNEff = 200.0;
  static constexpr double kResidAlpha = 0.05;
  static constexpr double kTempAlpha = 0.05;

  // Feed one (temperature, ppm) pair. ppm should be an accepted LTFM median and
  // the temperature a sample < 10 s old (freshness enforced by the caller).
  void addSample(double tempC, double ppm) {
    const double x = tempC - kRefTempC;

    if (solvable_) {
      const double pred = beta0_ + beta1_ * x;
      const double resid = ppm - pred;
      if (residInit_) {
        residVarEwma_ += kResidAlpha * (resid * resid - residVarEwma_);
      } else {
        residVarEwma_ = resid * resid;
        residInit_ = true;
      }
    }

    n_ = kLambda * n_ + 1.0;
    sx_ = kLambda * sx_ + x;
    sy_ = kLambda * sy_ + ppm;
    sxx_ = kLambda * sxx_ + x * x;
    sxy_ = kLambda * sxy_ + x * ppm;

    if (tempInit_) {
      tempEwmaC_ += kTempAlpha * (tempC - tempEwmaC_);
    } else {
      tempEwmaC_ = tempC;
      tempInit_ = true;
    }

    solve();
  }

  bool valid() const {
    return solvable_ && n_ >= kMinNEff && varX_ >= kMinVarX &&
           std::fabs(beta1_) <= kMaxSlopePpmPerC;
  }

  double slopePpmPerC() const { return solvable_ ? beta1_ : 0.0; }
  // Predicted ppm at an absolute temperature (shadow-validation / logging).
  double predictPpm(double tempC) const {
    return solvable_ ? (beta0_ + beta1_ * (tempC - kRefTempC)) : 0.0;
  }
  // Relative correction for holdover: slope x temperature change since entry.
  double predictRelativePpm(double deltaTempC) const {
    return solvable_ ? (beta1_ * deltaTempC) : 0.0;
  }
  double residRmsPpm() const { return std::sqrt(residVarEwma_); }
  double tempEwmaC() const { return tempEwmaC_; }
  double nEff() const { return n_; }
  double varX() const { return varX_; }

 private:
  void solve() {
    if (n_ <= 0.0) {
      solvable_ = false;
      return;
    }
    const double meanX = sx_ / n_;
    varX_ = sxx_ / n_ - meanX * meanX;
    if (varX_ < kMinVarX) {
      solvable_ = false;
      return;
    }
    const double covXY = sxy_ / n_ - meanX * (sy_ / n_);
    beta1_ = covXY / varX_;
    beta0_ = (sy_ - beta1_ * sx_) / n_;
    solvable_ = true;
  }

  double n_ = 0.0;
  double sx_ = 0.0;
  double sy_ = 0.0;
  double sxx_ = 0.0;
  double sxy_ = 0.0;
  double beta0_ = 0.0;
  double beta1_ = 0.0;
  double varX_ = 0.0;
  bool solvable_ = false;
  double residVarEwma_ = 0.0;
  bool residInit_ = false;
  double tempEwmaC_ = kRefTempC;
  bool tempInit_ = false;
};

// Error-budget helpers for the holdover window. Defaults (1.0 ppm uncertainty,
// 60000 ns budget) reproduce exactly the legacy fixed 60 s window; a TCXO rev
// only lowers driftUncertaintyPpm (0.1 -> 600 s at the same budget).

// Rounded predicted phase error after coasting `ageUs` at `driftPpm`.
// predictedErrorNs = ageUs * driftPpm / 1000 (exact in double at these scales).
inline uint64_t predictedHoldoverErrorNs(uint64_t ageUs, double driftPpm) {
  double d = driftPpm < 0.0 ? -driftPpm : driftPpm;
  const double ns = static_cast<double>(ageUs) * d / 1000.0;
  if (!(ns > 0.0)) {
    return 0;
  }
  if (ns >= 1.8e19) {  // guard against u64 overflow on absurd inputs
    return UINT64_MAX;
  }
  return static_cast<uint64_t>(ns + 0.5);
}

// Drift used for the budget: floor of the configured uncertainty, widened to
// 3 sigma of the learned frequency when that is larger.
inline double effectiveHoldoverDriftPpm(double driftUncertaintyPpm, double ltSigmaPpm) {
  return std::max(driftUncertaintyPpm, 3.0 * ltSigmaPpm);
}

}  // namespace mmpr
