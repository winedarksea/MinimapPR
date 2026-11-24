//! Solve-sensor registry: which sensors back each solve buffer, in which
//! order, at which positions and sample rates.
//!
//! Routing: a node is first validated against its OWN array offsets
//! (`dsp_node_array_geometry_gate`), then placed into its explicit cluster or
//! the automatic site cluster. A node that cannot be placed solves node-local
//! with its own offsets and a stated reason.
//!
//! Buffer channel order is sensor order (node id, then channel), so every
//! arrival order produces the same layout. A join or eviction that would move
//! an existing channel is reported to the worker through
//! [`plan_buffer_channel_reconcile`], which then moves the audio with it.
//!
//! Bounded: sensors not refreshed within
//! `localization_sensor_registry_stale_seconds` are evicted, and empty
//! buffer sets are dropped.

use super::*;
use crate::srp_phat::HalfSpace;
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, SensorPositionSource, SolveBufferScope, SolveCoordinateFrame,
    SolveExclusionReason,
};

const DEFAULT_STREAM_ID: &str = "audio_main";
const NANOSECONDS_PER_SECOND_F64: f64 = 1_000_000_000.0;

/// One node's ingest, as the registry needs it.
pub(crate) struct NodeRoutingRequest<'a> {
    pub(crate) stream_key: &'a str,
    pub(crate) manifest: &'a DspManifest,
    pub(crate) channel_count: usize,
    pub(crate) sample_rate_hz: u32,
    pub(crate) now_ns: u128,
}

#[derive(Clone, Debug)]
pub(crate) struct SensorRoute {
    pub(crate) buffer_key: String,
    /// Sensor id behind every buffer channel, in buffer order.
    pub(crate) buffer_sensor_ids: Arc<Vec<String>>,
    /// Buffer channel of each of the node's decoded channels.
    pub(crate) buffer_channel_indices: Vec<usize>,
    pub(crate) layout: Arc<SolveSensorLayout>,
}

pub(crate) struct NodeRoutingOutcome {
    pub(crate) route: SensorRoute,
    /// Set when the node's own array cannot run the spatial pipeline at all.
    pub(crate) large_array_omni_fallback_reason: Option<&'static str>,
}

/// Placement of one node's sensors in the solve frame.
enum NodeWorldSensors {
    Placed {
        sensor_positions_m: Vec<[f64; 3]>,
        position_source: SensorPositionSource,
        position_std_m: Option<f64>,
    },
    #[expect(dead_code, reason = "the R2 seam returns the exclusion reason")]
    Untrusted { reason: SolveExclusionReason },
}

/// R2 seam: the single call that places a node's sensors in the solve frame.
/// The integration replaces the body with
/// `SiteGeometryState::resolve_node_placement` (pushed placement > trusted GPS
/// > untrusted) followed by `sensor_world_position_m` per offset; until then
/// it is the legacy automatic-GPS projection with identity orientation.
fn resolve_node_world_sensors(
    site_frame: &mut AutomaticSiteFrame,
    reported_fix: Option<ReportedGpsPosition>,
    body_offsets_m: &[[f32; 3]],
) -> NodeWorldSensors {
    let Some(fix) = reported_fix else {
        return NodeWorldSensors::Untrusted {
            reason: SolveExclusionReason::NodePositionUntrustedNoOverride,
        };
    };
    let node_position_m = site_frame.local_position_m(fix);
    NodeWorldSensors::Placed {
        sensor_positions_m: body_offsets_m
            .iter()
            .map(|offset| {
                let offset = offset.map(f64::from);
                [
                    node_position_m[0] + offset[0],
                    node_position_m[1] + offset[1],
                    node_position_m[2] + offset[2],
                ]
            })
            .collect(),
        position_source: SensorPositionSource::ReportedGps,
        position_std_m: None,
    }
}

pub(crate) struct ClusterSensorRegistry {
    sets_by_buffer_key: HashMap<String, BufferSensorSet>,
    automatic_site_frame: AutomaticSiteFrame,
    max_solve_sensors: usize,
    stale_after_ns: u128,
    /// Revisions are issued from one counter so a buffer set that is evicted
    /// and rebuilt never reports a revision it already used.
    last_issued_layout_revision: u64,
}

fn issue_layout_revision(last_issued_layout_revision: &mut u64) -> u64 {
    *last_issued_layout_revision += 1;
    *last_issued_layout_revision
}

impl ClusterSensorRegistry {
    pub(crate) fn new(config: &crate::localization_solve_config::LocalizationSolveConfig) -> Self {
        Self {
            sets_by_buffer_key: HashMap::new(),
            automatic_site_frame: AutomaticSiteFrame::default(),
            max_solve_sensors: config.localization_max_solve_sensors,
            stale_after_ns: (config.localization_sensor_registry_stale_seconds.max(0.0)
                * NANOSECONDS_PER_SECOND_F64)
                .round() as u128,
            last_issued_layout_revision: 0,
        }
    }

    pub(crate) fn route_node(&mut self, request: &NodeRoutingRequest) -> NodeRoutingOutcome {
        let node_id = stream_key_node_id(request.stream_key);
        let node = request
            .manifest
            .node_context
            .as_ref()
            .and_then(|context| context.get("node"));
        let array = match validate_node_array_geometry(node, request.channel_count) {
            NodeArrayGeometryVerdict::LargeArrayOmniFallback { reason } => {
                return NodeRoutingOutcome {
                    route: self.unbuffered_large_array_route(request, node_id),
                    large_array_omni_fallback_reason: Some(reason),
                };
            }
            NodeArrayGeometryVerdict::Usable(array) => Some(array),
            NodeArrayGeometryVerdict::UnusableSmallArray => None,
        };
        let routed = match explicit_cluster_declaration(request.manifest) {
            ExplicitClusterDeclaration::Declared(cluster) => {
                self.route_into_explicit_cluster(request, node_id, array.as_ref(), cluster)
            }
            ExplicitClusterDeclaration::Incomplete => {
                Err(NodeLocalRoutingReason::ExplicitClusterIncomplete)
            }
            ExplicitClusterDeclaration::Absent => match &array {
                Some(array) => self.route_into_automatic_cluster(request, node_id, node, array),
                None => Err(NodeLocalRoutingReason::NodeArrayGeometryUnusable),
            },
        };
        let route = routed.unwrap_or_else(|reason| {
            debug!(
                stream_key = request.stream_key,
                reason = reason.as_str(),
                "node solves node-local"
            );
            self.route_node_local(request, node_id, array.as_ref(), reason)
        });
        NodeRoutingOutcome {
            route,
            large_array_omni_fallback_reason: None,
        }
    }

    /// Stale-stream purge: the buffer is gone, so is its sensor set.
    pub(crate) fn forget_buffer(&mut self, buffer_key: &str) {
        self.sets_by_buffer_key.remove(buffer_key);
    }

    /// Periodic sweep so sets whose buffers are never routed again still shrink.
    pub(crate) fn evict_stale_sensors(&mut self, now_ns: u128) {
        let stale_after_ns = self.stale_after_ns;
        let last_issued = &mut self.last_issued_layout_revision;
        self.sets_by_buffer_key.retain(|_, set| {
            if set.evict_stale(now_ns, stale_after_ns) {
                set.layout_revision = issue_layout_revision(last_issued);
            }
            !set.sensors.is_empty()
        });
    }

    #[cfg(test)]
    pub(crate) fn tracked_sensor_count(&self) -> usize {
        self.sets_by_buffer_key
            .values()
            .map(|set| set.sensors.len())
            .sum()
    }

    fn route_into_explicit_cluster(
        &mut self,
        request: &NodeRoutingRequest,
        node_id: &str,
        array: Option<&ValidatedNodeArray>,
        cluster: ExplicitClusterSensorList,
    ) -> Result<SensorRoute, NodeLocalRoutingReason> {
        if cluster.sensors.len() < EXPLICIT_CLUSTER_MIN_SENSOR_COUNT {
            return Err(NodeLocalRoutingReason::ExplicitClusterBelowMinimumSensors);
        }
        let every_channel_declared = (0..request.channel_count)
            .all(|channel| cluster.declares(&sensor_id_for_node_channel(node_id, channel)));
        if !every_channel_declared {
            return Err(NodeLocalRoutingReason::NodeChannelsMissingFromExplicitCluster);
        }
        let buffer_key = cluster_buffer_key(&cluster.cluster_id, request.stream_key);
        let set = self
            .sets_by_buffer_key
            .entry(buffer_key.clone())
            .or_insert_with(|| {
                BufferSensorSet::new(
                    SolveBufferScope::ExplicitCluster,
                    Some(cluster.cluster_id.clone()),
                )
            });
        let mut changed = set.evict_stale(request.now_ns, self.stale_after_ns);
        let reporting_sensor_ids: Vec<String> = (0..request.channel_count)
            .map(|channel| sensor_id_for_node_channel(node_id, channel))
            .collect();
        for (sensor_id, position_m) in &cluster.sensors {
            let is_reporting_sensor = reporting_sensor_ids.contains(sensor_id);
            changed |= set.observe(
                sensor_id,
                SensorObservation {
                    position_m: *position_m,
                    position_source: SensorPositionSource::ExplicitClusterManifest,
                    position_std_m: None,
                    sample_rate_hz: is_reporting_sensor.then_some(request.sample_rate_hz),
                },
                request.now_ns,
            );
        }
        set.set_node_half_space(
            node_id,
            array.map_or(HalfSpace::None, |array| array.half_space),
        );
        if changed {
            set.layout_revision = issue_layout_revision(&mut self.last_issued_layout_revision);
        }
        let frame = SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::ExplicitClusterPositions,
            origin: None,
            projection: None,
        };
        Ok(snapshot_route(
            set,
            buffer_key,
            request,
            node_id,
            RouteSnapshotContext {
                coordinate_frame: frame,
                geometry_push_revision: None,
                node_local_routing_reason: None,
                geometry_unusable_node_id: None,
                max_solve_sensors: self.max_solve_sensors,
            },
        ))
    }

    fn route_into_automatic_cluster(
        &mut self,
        request: &NodeRoutingRequest,
        node_id: &str,
        node: Option<&serde_json::Value>,
        array: &ValidatedNodeArray,
    ) -> Result<SensorRoute, NodeLocalRoutingReason> {
        let reported_fix = node.and_then(reported_gps_position);
        let NodeWorldSensors::Placed {
            sensor_positions_m,
            position_source,
            position_std_m,
        } = resolve_node_world_sensors(
            &mut self.automatic_site_frame,
            reported_fix,
            &array.body_offsets_m,
        )
        else {
            return Err(NodeLocalRoutingReason::NodePositionUntrusted);
        };
        let buffer_key = cluster_buffer_key(AUTOMATIC_GPS_CLUSTER_ID, request.stream_key);
        let set = self
            .sets_by_buffer_key
            .entry(buffer_key.clone())
            .or_insert_with(|| {
                BufferSensorSet::new(
                    SolveBufferScope::AutomaticSiteCluster,
                    Some(AUTOMATIC_GPS_CLUSTER_ID.to_string()),
                )
            });
        let mut changed = set.evict_stale(request.now_ns, self.stale_after_ns);
        changed |= set.retain_node_channels(node_id, request.channel_count);
        for (channel_index, position_m) in sensor_positions_m.into_iter().enumerate() {
            changed |= set.observe(
                &sensor_id_for_node_channel(node_id, channel_index),
                SensorObservation {
                    position_m,
                    position_source,
                    position_std_m,
                    sample_rate_hz: Some(request.sample_rate_hz),
                },
                request.now_ns,
            );
        }
        set.set_node_half_space(node_id, array.half_space);
        if changed {
            set.layout_revision = issue_layout_revision(&mut self.last_issued_layout_revision);
        }
        let context = RouteSnapshotContext {
            coordinate_frame: self.automatic_site_frame.coordinate_frame(),
            geometry_push_revision: self.automatic_site_frame.geometry_push_revision(),
            node_local_routing_reason: None,
            geometry_unusable_node_id: None,
            max_solve_sensors: self.max_solve_sensors,
        };
        Ok(snapshot_route(set, buffer_key, request, node_id, context))
    }

    fn route_node_local(
        &mut self,
        request: &NodeRoutingRequest,
        node_id: &str,
        array: Option<&ValidatedNodeArray>,
        reason: NodeLocalRoutingReason,
    ) -> SensorRoute {
        let buffer_key = request.stream_key.to_string();
        let set = self
            .sets_by_buffer_key
            .entry(buffer_key.clone())
            .or_insert_with(|| BufferSensorSet::new(SolveBufferScope::NodeLocal, None));
        let changed = observe_node_body_sensors(set, request, node_id, array);
        if changed {
            set.layout_revision = issue_layout_revision(&mut self.last_issued_layout_revision);
        }
        let context = node_body_snapshot_context(Some(reason), None, self.max_solve_sensors);
        snapshot_route(set, buffer_key, request, node_id, context)
    }

    /// Large-array omni renders bypass the buffers, so nothing is retained.
    fn unbuffered_large_array_route(
        &self,
        request: &NodeRoutingRequest,
        node_id: &str,
    ) -> SensorRoute {
        let mut set = BufferSensorSet::new(SolveBufferScope::NodeLocal, None);
        observe_node_body_sensors(&mut set, request, node_id, None);
        set.layout_revision = self.last_issued_layout_revision;
        let context = node_body_snapshot_context(
            Some(NodeLocalRoutingReason::NodeArrayGeometryUnusable),
            Some(node_id.to_string()),
            self.max_solve_sensors,
        );
        snapshot_route(
            &set,
            request.stream_key.to_string(),
            request,
            node_id,
            context,
        )
    }
}

/// The node's own channels at their body offsets (zeros when it has none:
/// such a node never solves, but every channel still needs a layout entry).
fn observe_node_body_sensors(
    set: &mut BufferSensorSet,
    request: &NodeRoutingRequest,
    node_id: &str,
    array: Option<&ValidatedNodeArray>,
) -> bool {
    let mut changed = set.retain_node_channels(node_id, request.channel_count);
    for channel_index in 0..request.channel_count {
        let offset = array
            .and_then(|array| array.body_offsets_m.get(channel_index))
            .copied()
            .unwrap_or([0.0; 3]);
        changed |= set.observe(
            &sensor_id_for_node_channel(node_id, channel_index),
            SensorObservation {
                position_m: offset.map(f64::from),
                position_source: SensorPositionSource::NodeBodyOffset,
                position_std_m: None,
                sample_rate_hz: Some(request.sample_rate_hz),
            },
            request.now_ns,
        );
    }
    set.set_node_half_space(
        node_id,
        array.map_or(HalfSpace::None, |array| array.half_space),
    );
    changed
}

/// Shared multi-node buffers are keyed `cluster::<cluster_id>::<stream_id>`.
fn cluster_buffer_key(cluster_id: &str, stream_key: &str) -> String {
    let stream_id = stream_key_stream_id(stream_key).unwrap_or(DEFAULT_STREAM_ID);
    format!("{CLUSTER_BUFFER_KEY_PREFIX}{cluster_id}::{stream_id}")
}

#[cfg(test)]
#[path = "dsp_cluster_sensor_registry_tests.rs"]
mod dsp_cluster_sensor_registry_tests;
