"""Statistical coverage of stated position covariances, measured on held-out seeds.

A localizer that states covariance ``S`` at confidence ``p`` claims the truth
lies inside the ``p``-ellipsoid ``e^T S^-1 e <= chi2_k(p)`` a fraction ``p`` of
the time. Over ``n`` independent held-out samples the covered count is then
``Binomial(n, p)``; a two-sided binomial test at the configured significance
decides whether the empirical coverage is consistent with the claim.

Held-out seeds: any tuning of covariance inflation (clock-class defaults,
hand-measured ``position_std_m``) is fit on development seeds. Coverage is
measured only on samples whose seed is in the held-out set, so the check is
not graded on the data the uncertainty model was tuned to.

A singular or non-finite covariance cannot contain anything: it is counted
as *invalid* and as not covered, so a solver emitting degenerate covariances
fails the gate instead of being silently skipped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest, chi2

from minimappr.sim.replay_gate_config import UncertaintyCoverageConfig

COVERAGE_VERDICT_CONSISTENT = "consistent"
COVERAGE_VERDICT_UNDER_COVERED = "under_covered"
COVERAGE_VERDICT_OVER_COVERED = "over_covered"
COVERAGE_VERDICT_INSUFFICIENT_SAMPLES = "insufficient_samples"


@dataclass(frozen=True)
class CoverageSample:
    """One localized fix: the seed it came from, its error vector and stated covariance."""

    seed: int
    position_error_m: tuple[float, float, float]
    covariance_m2: tuple[tuple[float, float, float], ...]


@dataclass
class UncertaintyCoverageReport:
    stated_confidence_level: float
    held_out_seeds: tuple[int, ...]
    sample_count: int = 0
    covered_count: int = 0
    invalid_covariance_count: int = 0
    empirical_coverage: float | None = None
    binomial_p_value: float | None = None
    acceptance_interval_covered_count: tuple[int, int] | None = None
    coverage_by_seed: dict[int, float] = field(default_factory=dict)
    verdict: str = COVERAGE_VERDICT_INSUFFICIENT_SAMPLES
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def coverage_sample_from_detection(seed: int, detection: dict, truth_position_m: Iterable[float]) -> CoverageSample | None:
    """Build a sample from a stored detection row; ``None`` when it states no covariance."""
    covariance = detection.get("position_covariance_m2")
    position = detection.get("position_m")
    if covariance is None or position is None:
        return None
    error = np.asarray(position, dtype=np.float64)[:3] - np.asarray(list(truth_position_m), dtype=np.float64)[:3]
    return CoverageSample(
        seed=int(seed),
        position_error_m=tuple(float(axis) for axis in error),
        covariance_m2=tuple(tuple(float(value) for value in row[:3]) for row in covariance[:3]),
    )


def partition_held_out_seeds(seeds: Iterable[int], held_out_every_nth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic (development, held_out) split: every ``held_out_every_nth`` sorted seed is held out.

    Rank-based rather than random so the split is reproducible across runs
    and machines without carrying an RNG seed of its own.
    """
    if held_out_every_nth < 2:
        raise ValueError("held_out_every_nth must be >= 2 (otherwise nothing is left for development)")
    ordered_seeds = sorted(set(int(seed) for seed in seeds))
    held_out = tuple(seed for rank, seed in enumerate(ordered_seeds) if rank % held_out_every_nth == held_out_every_nth - 1)
    development = tuple(seed for seed in ordered_seeds if seed not in held_out)
    return development, held_out


def squared_mahalanobis_distance(error_m: np.ndarray, covariance_m2: np.ndarray) -> float | None:
    """``e^T S^-1 e``; ``None`` for a singular, non-finite or indefinite covariance."""
    if not (np.all(np.isfinite(error_m)) and np.all(np.isfinite(covariance_m2))):
        return None
    try:
        # Cholesky both inverts and rejects non-positive-definite matrices,
        # which a sign error in covariance propagation would produce.
        cholesky_factor = np.linalg.cholesky(covariance_m2)
        whitened = np.linalg.solve(cholesky_factor, error_m)
    except np.linalg.LinAlgError:
        return None
    return float(whitened @ whitened)


def binomial_acceptance_interval(sample_count: int, probability: float, significance: float) -> tuple[int, int]:
    """Covered counts the two-sided binomial test at ``significance`` does not reject."""
    accepted = [
        covered_count
        for covered_count in range(sample_count + 1)
        if binomtest(covered_count, sample_count, probability, alternative="two-sided").pvalue >= significance
    ]
    return (accepted[0], accepted[-1]) if accepted else (sample_count + 1, -1)


def measure_uncertainty_coverage(
    samples: Iterable[CoverageSample],
    held_out_seeds: Iterable[int],
    config: UncertaintyCoverageConfig,
) -> UncertaintyCoverageReport:
    held_out = tuple(sorted(set(int(seed) for seed in held_out_seeds)))
    report = UncertaintyCoverageReport(stated_confidence_level=config.stated_confidence_level, held_out_seeds=held_out)
    if not held_out:
        report.errors.append("no held-out seeds: coverage must be measured on seeds not used for tuning")
        return report
    dimension_count = config.position_dimension_count
    containment_threshold = float(chi2.ppf(config.stated_confidence_level, df=dimension_count))
    held_out_set = set(held_out)
    covered_by_seed: dict[int, list[bool]] = {seed: [] for seed in held_out}

    for sample in samples:
        if sample.seed not in held_out_set:
            continue
        error = np.asarray(sample.position_error_m, dtype=np.float64)[:dimension_count]
        covariance = np.asarray(sample.covariance_m2, dtype=np.float64)[:dimension_count, :dimension_count]
        distance_squared = squared_mahalanobis_distance(error, covariance)
        report.sample_count += 1
        if distance_squared is None:
            report.invalid_covariance_count += 1
            covered_by_seed[sample.seed].append(False)
            continue
        covered = distance_squared <= containment_threshold
        report.covered_count += int(covered)
        covered_by_seed[sample.seed].append(covered)

    report.coverage_by_seed = {
        seed: sum(flags) / len(flags) for seed, flags in covered_by_seed.items() if flags
    }
    missing_seeds = [seed for seed, flags in covered_by_seed.items() if not flags]
    if missing_seeds:
        report.errors.append(f"held-out seeds produced no samples: {missing_seeds}")
    if report.sample_count < config.minimum_held_out_sample_count:
        report.errors.append(
            f"{report.sample_count} held-out samples below the minimum {config.minimum_held_out_sample_count}"
        )
        if report.sample_count == 0:
            return report
    report.empirical_coverage = report.covered_count / report.sample_count
    report.binomial_p_value = float(
        binomtest(report.covered_count, report.sample_count, config.stated_confidence_level, alternative="two-sided").pvalue
    )
    report.acceptance_interval_covered_count = binomial_acceptance_interval(
        report.sample_count, config.stated_confidence_level, config.binomial_test_significance
    )
    if report.sample_count < config.minimum_held_out_sample_count:
        return report
    lower_bound, upper_bound = report.acceptance_interval_covered_count
    if report.covered_count < lower_bound:
        report.verdict = COVERAGE_VERDICT_UNDER_COVERED
        report.errors.append(
            f"covariance under-covers: {report.covered_count}/{report.sample_count} inside the "
            f"{config.stated_confidence_level:.0%} ellipsoid (accepted {lower_bound}..{upper_bound}, "
            f"p={report.binomial_p_value:.2g})"
        )
    elif report.covered_count > upper_bound:
        report.verdict = COVERAGE_VERDICT_OVER_COVERED
        if config.fail_on_over_coverage:
            report.errors.append(
                f"covariance over-covers: {report.covered_count}/{report.sample_count} inside the "
                f"{config.stated_confidence_level:.0%} ellipsoid (accepted {lower_bound}..{upper_bound})"
            )
    else:
        report.verdict = COVERAGE_VERDICT_CONSISTENT
    return report
