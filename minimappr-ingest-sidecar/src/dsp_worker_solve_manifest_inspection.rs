//! Readers over published solve manifests for worker-level localization
//! tests: provenance lookup, lags keyed by sensor-id pair, cross-node counts
//! and geometry checks against fixture truth.

use super::dsp_worker_test_node_fixtures::{
    CoherentPointSource, FixtureNode, FIXTURE_SAMPLE_RATE_HZ,
};
use super::*;
use minimappr_core::localization_sensor_entry::SolveProvenance;

/// GCC-PHAT peaks on an integer-sample synthetic delay to well inside half a sample.
const LAG_TOLERANCE_SECONDS: f64 = 0.5 / FIXTURE_SAMPLE_RATE_HZ as f64;
const POSITION_TOLERANCE_M: f64 = 0.01;

pub(super) type SensorPairKey = (String, String);

pub(super) fn solve_provenance(manifest: &DspManifest) -> &SolveProvenance {
    manifest
        .localization
        .as_ref()
        .and_then(|localization| localization.solve_provenance.as_ref())
        .unwrap_or_else(|| {
            panic!(
                "{} has no localization.solve_provenance (emitted from milestone 1, R6)",
                manifest.manifest_id
            )
        })
}

pub(super) fn attempted_localization_results(published: &[DspManifest]) -> Vec<&DspManifest> {
    published
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .filter(|manifest| {
            manifest.localization.as_ref().is_some_and(|localization| {
                localization.resolved_algorithm != "localization_cadence_skipped"
            })
        })
        .collect()
}

/// Lags keyed by sensor-id pair, oriented so the key is sorted (sign flipped
/// when the solve listed the pair the other way round).
pub(super) fn lags_by_sensor_pair(manifest: &DspManifest) -> HashMap<SensorPairKey, f64> {
    let provenance = solve_provenance(manifest);
    manifest
        .localization
        .as_ref()
        .expect("localization payload")
        .pair_tdoas
        .iter()
        .map(|pair| {
            let id_a = provenance.sensors[pair.ch_a].sensor_id.clone();
            let id_b = provenance.sensors[pair.ch_b].sensor_id.clone();
            let lag_seconds = f64::from(pair.lag_seconds);
            if id_a <= id_b {
                ((id_a, id_b), lag_seconds)
            } else {
                ((id_b, id_a), -lag_seconds)
            }
        })
        .collect()
}

pub(super) fn cross_node_pair_count(manifest: &DspManifest) -> usize {
    let provenance = solve_provenance(manifest);
    let localization = manifest
        .localization
        .as_ref()
        .expect("localization payload");
    for pair in &localization.pair_tdoas {
        let is_cross_node =
            provenance.sensors[pair.ch_a].node_id != provenance.sensors[pair.ch_b].node_id;
        assert_eq!(pair.is_cross_node, Some(is_cross_node), "{pair:?}");
    }
    localization
        .pair_tdoas
        .iter()
        .filter(|pair| pair.is_cross_node == Some(true))
        .count()
}

pub(super) fn world_positions_by_sensor_id(nodes: &[&FixtureNode]) -> HashMap<String, [f64; 3]> {
    nodes
        .iter()
        .flat_map(|node| node.sensor_world_positions_m())
        .collect()
}

pub(super) fn assert_lags_match_geometry(
    lags: &HashMap<SensorPairKey, f64>,
    source: &CoherentPointSource,
    nodes: &[&FixtureNode],
) {
    let positions = world_positions_by_sensor_id(nodes);
    for ((id_a, id_b), lag_seconds) in lags {
        let expected = source.expected_lag_seconds(positions[id_a], positions[id_b]);
        assert!(
            (lag_seconds - expected).abs() <= LAG_TOLERANCE_SECONDS,
            "{id_a}/{id_b}: lag {lag_seconds} s, geometry says {expected} s"
        );
    }
}

pub(super) fn sensor_position_m(provenance: &SolveProvenance, sensor_id: &str) -> [f64; 3] {
    provenance
        .sensors
        .iter()
        .find(|sensor| sensor.sensor_id == sensor_id)
        .unwrap_or_else(|| panic!("{sensor_id} missing from the solve layout"))
        .position_m
}

pub(super) fn assert_position_near(actual: [f64; 3], expected: [f64; 3], context: &str) {
    let error_m = (0..3)
        .map(|axis| (actual[axis] - expected[axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        error_m <= POSITION_TOLERANCE_M,
        "{context}: {actual:?} vs expected {expected:?} ({error_m} m)"
    );
}
