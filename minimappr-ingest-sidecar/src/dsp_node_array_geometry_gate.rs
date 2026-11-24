//! Per-node array validation, run before any cluster routing: a node is
//! judged on its OWN body offsets and channel count, never on the size of the
//! cluster it is about to join (the old gate compared a 5-mic planar node with
//! the whole cluster's sensor list and dropped it from every mixed cluster).

use super::*;
use crate::srp_phat::HalfSpace;

/// Largest mic array the spatial pipeline (SRP-PHAT / GCC-PHAT / render_mvdr_n)
/// runs on. Sirith Planar (5 mics) fits; anything larger takes the omni fallback.
pub(crate) const MAX_SPATIAL_CHANNELS: usize = 8;
/// Minimum out-of-collinearity span (metres) a >4-mic array needs before the
/// spatial pipeline can use it.
const SPATIAL_GEOMETRY_MIN_SPAN_M: f32 = 0.005;
/// Channel count of the legacy tetra path, whose geometry has firmware defaults.
const TETRA_CHANNEL_COUNT: usize = 4;
pub(crate) const LARGE_ARRAY_REASON_EXCEEDS_MAX_CHANNELS: &str =
    "array_exceeds_max_spatial_channels";
pub(crate) const LARGE_ARRAY_REASON_GEOMETRY_UNUSABLE: &str =
    "non_tetrahedral_array_geometry_unusable";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ValidatedNodeArray {
    /// One body-frame offset per decoded channel, in channel order.
    pub(crate) body_offsets_m: Vec<[f32; 3]>,
    /// Coplanar up/down declaration; only applied to single-node solves.
    pub(crate) half_space: HalfSpace,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NodeArrayGeometryVerdict {
    Usable(ValidatedNodeArray),
    /// A 2-3 channel node without offsets. It cannot run SRP on its own and
    /// cannot be placed in the automatic cluster; an explicit cluster that
    /// lists its channels still positions it.
    UnusableSmallArray,
    /// >4 channels with too many mics or missing / collinear offsets. Skips
    /// the rolling buffers entirely (omni render of the newest segment).
    LargeArrayOmniFallback {
        reason: &'static str,
    },
}

pub(crate) fn validate_node_array_geometry(
    node: Option<&serde_json::Value>,
    channel_count: usize,
) -> NodeArrayGeometryVerdict {
    if channel_count > MAX_SPATIAL_CHANNELS {
        return NodeArrayGeometryVerdict::LargeArrayOmniFallback {
            reason: LARGE_ARRAY_REASON_EXCEEDS_MAX_CHANNELS,
        };
    }
    let body_offsets_m = node_body_offsets_for_channel_count(node, channel_count);
    let usable = body_offsets_m.len() == channel_count
        && (channel_count <= TETRA_CHANNEL_COUNT
            || crate::dsp_math::array_spans_at_least_2d(
                &body_offsets_m,
                SPATIAL_GEOMETRY_MIN_SPAN_M,
            ));
    match (usable, channel_count > TETRA_CHANNEL_COUNT) {
        (true, _) => NodeArrayGeometryVerdict::Usable(ValidatedNodeArray {
            body_offsets_m,
            half_space: half_space_from_node(node),
        }),
        (false, true) => NodeArrayGeometryVerdict::LargeArrayOmniFallback {
            reason: LARGE_ARRAY_REASON_GEOMETRY_UNUSABLE,
        },
        (false, false) => NodeArrayGeometryVerdict::UnusableSmallArray,
    }
}

/// `node.sensor_offsets_m` (first `channel_count` entries) when it covers every
/// channel; otherwise the firmware defaults for tetra (4) and point (1) nodes;
/// otherwise empty (unusable).
fn node_body_offsets_for_channel_count(
    node: Option<&serde_json::Value>,
    channel_count: usize,
) -> Vec<[f32; 3]> {
    let declared = node
        .and_then(|node| node.get("sensor_offsets_m"))
        .and_then(serde_json::Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .take(channel_count)
                .map(|value| {
                    let axes = value.as_array()?;
                    if axes.len() < 3 {
                        return None;
                    }
                    Some([
                        axes[0].as_f64()? as f32,
                        axes[1].as_f64()? as f32,
                        axes[2].as_f64()? as f32,
                    ])
                })
                .collect::<Option<Vec<_>>>()
        });
    match declared {
        Some(offsets) if offsets.len() == channel_count => offsets,
        _ if channel_count == SIRITH_MIC_POSITIONS_M.len() => SIRITH_MIC_POSITIONS_M.to_vec(),
        _ if channel_count == 1 => vec![[0.0, 0.0, 0.0]],
        _ => Vec::new(),
    }
}

/// `node.half_space` ("upper" | "lower" | "none"); absent means no constraint.
fn half_space_from_node(node: Option<&serde_json::Value>) -> HalfSpace {
    node.and_then(|node| node.get("half_space"))
        .and_then(serde_json::Value::as_str)
        .map(HalfSpace::from_wire_str)
        .unwrap_or(HalfSpace::None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minimappr_core::sirith_array_geometry::SIRITH_PLANAR_MIC_OFFSETS_M;

    #[test]
    fn planar_node_is_validated_against_its_own_five_offsets() {
        let node = serde_json::json!({
            "sensor_offsets_m": SIRITH_PLANAR_MIC_OFFSETS_M,
            "half_space": "upper"
        });
        let NodeArrayGeometryVerdict::Usable(array) = validate_node_array_geometry(Some(&node), 5)
        else {
            panic!("planar offsets are a valid 2-D array");
        };
        assert_eq!(array.body_offsets_m, SIRITH_PLANAR_MIC_OFFSETS_M.to_vec());
        assert_eq!(array.half_space, HalfSpace::Upper);
    }

    #[test]
    fn tetra_and_point_nodes_fall_back_to_firmware_geometry() {
        assert_eq!(
            validate_node_array_geometry(None, 4),
            NodeArrayGeometryVerdict::Usable(ValidatedNodeArray {
                body_offsets_m: SIRITH_MIC_POSITIONS_M.to_vec(),
                half_space: HalfSpace::None,
            })
        );
        assert_eq!(
            validate_node_array_geometry(None, 1),
            NodeArrayGeometryVerdict::Usable(ValidatedNodeArray {
                body_offsets_m: vec![[0.0, 0.0, 0.0]],
                half_space: HalfSpace::None,
            })
        );
        assert_eq!(
            validate_node_array_geometry(None, 2),
            NodeArrayGeometryVerdict::UnusableSmallArray
        );
    }

    #[test]
    fn large_arrays_without_usable_offsets_take_the_omni_fallback() {
        assert_eq!(
            validate_node_array_geometry(None, 5),
            NodeArrayGeometryVerdict::LargeArrayOmniFallback {
                reason: LARGE_ARRAY_REASON_GEOMETRY_UNUSABLE
            }
        );
        let collinear = serde_json::json!({
            "sensor_offsets_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0],
                                 [0.03, 0.0, 0.0], [0.04, 0.0, 0.0]]
        });
        assert_eq!(
            validate_node_array_geometry(Some(&collinear), 5),
            NodeArrayGeometryVerdict::LargeArrayOmniFallback {
                reason: LARGE_ARRAY_REASON_GEOMETRY_UNUSABLE
            }
        );
        assert_eq!(
            validate_node_array_geometry(None, MAX_SPATIAL_CHANNELS + 1),
            NodeArrayGeometryVerdict::LargeArrayOmniFallback {
                reason: LARGE_ARRAY_REASON_EXCEEDS_MAX_CHANNELS
            }
        );
    }
}
