//! Which sensor pairs one solve may correlate. Milestone 2 (R7) computes it
//! from rate, baseline, coverage and clock evidence and threads it through
//! pair observations, covariance, render selection and the deferred path; the
//! milestone 0 seam admits every pair, so behaviour is unchanged.

use minimappr_core::localization_sensor_entry::ExcludedSensorPair;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SolvePairAdmission {
    excluded_pairs: Vec<ExcludedSensorPair>,
}

#[cfg_attr(not(test), expect(dead_code, reason = "consumed by milestone 2 (R7)"))]
impl SolvePairAdmission {
    pub(crate) fn admit_all() -> Self {
        Self::default()
    }

    pub(crate) fn admits_pair(&self, sensor_index_a: usize, sensor_index_b: usize) -> bool {
        !self.excluded_pairs.iter().any(|pair| {
            (pair.sensor_index_a, pair.sensor_index_b) == (sensor_index_a, sensor_index_b)
                || (pair.sensor_index_a, pair.sensor_index_b) == (sensor_index_b, sensor_index_a)
        })
    }

    pub(crate) fn excluded_pairs(&self) -> &[ExcludedSensorPair] {
        &self.excluded_pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_all_admits_every_pair() {
        let admission = SolvePairAdmission::admit_all();
        assert!(admission.admits_pair(0, 5));
        assert!(admission.excluded_pairs().is_empty());
    }
}
