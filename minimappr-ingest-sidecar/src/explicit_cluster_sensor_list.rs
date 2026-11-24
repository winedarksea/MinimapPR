//! The operator-declared cluster a manifest carries (`cluster_id` +
//! `cluster_sensor_positions`), normalised to sensor order so the buffer
//! layout does not depend on the order Python listed the sensors in.

use super::*;

/// Clusters smaller than this never shared a buffer (historical minimum:
/// fewer sensors than SRP needs gain nothing from a shared buffer).
pub(crate) const EXPLICIT_CLUSTER_MIN_SENSOR_COUNT: usize = MIN_SRP_CHANNEL_COUNT;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitClusterSensorList {
    pub(crate) cluster_id: String,
    /// `(sensor_id, world position)`, sorted by [`sensor_order_key`], one
    /// entry per sensor id (a repeated id keeps its last position).
    pub(crate) sensors: Vec<(String, [f64; 3])>,
}

impl ExplicitClusterSensorList {
    pub(crate) fn declares(&self, sensor_id: &str) -> bool {
        self.sensors
            .iter()
            .any(|(declared, _)| declared == sensor_id)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExplicitClusterDeclaration {
    /// No cluster fields: the node is a candidate for the automatic cluster.
    Absent,
    /// Only one of `cluster_id` / non-empty `cluster_sensor_positions`, or a
    /// non-finite position: an operator boundary that cannot be honoured.
    Incomplete,
    Declared(ExplicitClusterSensorList),
}

pub(crate) fn explicit_cluster_declaration(manifest: &DspManifest) -> ExplicitClusterDeclaration {
    let positions = manifest
        .cluster_sensor_positions
        .as_ref()
        .filter(|positions| !positions.is_empty());
    match (manifest.cluster_id.as_deref(), positions) {
        (None, None) => ExplicitClusterDeclaration::Absent,
        (Some(cluster_id), Some(positions)) => {
            let all_finite = positions
                .iter()
                .all(|(_, position)| position.iter().all(|axis| axis.is_finite()));
            if !all_finite {
                return ExplicitClusterDeclaration::Incomplete;
            }
            let mut sensors: Vec<(String, [f64; 3])> = Vec::with_capacity(positions.len());
            for (sensor_id, position) in positions {
                let position_m = position.map(f64::from);
                match sensors
                    .iter_mut()
                    .find(|(declared, _)| declared == sensor_id)
                {
                    Some(existing) => existing.1 = position_m,
                    None => sensors.push((sensor_id.clone(), position_m)),
                }
            }
            sensors.sort_by_cached_key(|(sensor_id, _)| sensor_order_key(sensor_id));
            ExplicitClusterDeclaration::Declared(ExplicitClusterSensorList {
                cluster_id: cluster_id.to_string(),
                sensors,
            })
        }
        _ => ExplicitClusterDeclaration::Incomplete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with_cluster(
        cluster_id: Option<&str>,
        positions: Option<Vec<(String, [f32; 3])>>,
    ) -> DspManifest {
        let mut manifest: DspManifest = serde_json::from_value(serde_json::json!({
            "manifest_id": "m", "manifest_type": "raw_journal_append", "created_ns": 0,
            "source_handles": [], "promotion_ready": false
        }))
        .unwrap();
        manifest.cluster_id = cluster_id.map(str::to_string);
        manifest.cluster_sensor_positions = positions;
        manifest
    }

    #[test]
    fn declared_sensors_are_sorted_by_node_then_channel_whatever_the_listed_order() {
        let listed = |ids: &[&str]| -> Vec<(String, [f32; 3])> {
            ids.iter()
                .enumerate()
                .map(|(index, id)| (id.to_string(), [index as f32, 0.0, 0.0]))
                .collect()
        };
        let forward = ["tetra:ch0", "tetra:ch10", "tetra:ch2", "point:ch0"];
        let reversed = ["point:ch0", "tetra:ch2", "tetra:ch10", "tetra:ch0"];
        let ids = |declaration: ExplicitClusterDeclaration| match declaration {
            ExplicitClusterDeclaration::Declared(list) => list
                .sensors
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            other => panic!("expected a declared cluster, got {other:?}"),
        };
        let expected = ["point:ch0", "tetra:ch0", "tetra:ch2", "tetra:ch10"];
        for order in [forward, reversed] {
            let manifest = manifest_with_cluster(Some("c"), Some(listed(&order)));
            assert_eq!(ids(explicit_cluster_declaration(&manifest)), expected);
        }
    }

    #[test]
    fn half_declared_cluster_is_incomplete_and_no_fields_is_absent() {
        assert_eq!(
            explicit_cluster_declaration(&manifest_with_cluster(None, None)),
            ExplicitClusterDeclaration::Absent
        );
        assert_eq!(
            explicit_cluster_declaration(&manifest_with_cluster(Some("c"), None)),
            ExplicitClusterDeclaration::Incomplete
        );
        assert_eq!(
            explicit_cluster_declaration(&manifest_with_cluster(
                None,
                Some(vec![("a:ch0".to_string(), [0.0; 3])])
            )),
            ExplicitClusterDeclaration::Incomplete
        );
    }
}
