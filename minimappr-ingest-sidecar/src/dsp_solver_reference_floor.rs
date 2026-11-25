//! The SRP solver's reference-channel floor for one solve: the absolute
//! dead-channel level, lowered to `localization_solver_floor_relative_db`
//! over the solving nodes' tracked noise floor (see `SolverReferenceChannelFloor`).
//!
//! The floor only answers "is there any energy to localize"; the trigger
//! decides whether there is an event. So it is measured against the quietest
//! active node: if the loudest channel is not even `relative_db` over that
//! floor, no member holds a usable signal. A noisy member cannot raise the
//! bar for a quiet member that heard a faint far call.

use std::collections::BTreeMap;

use super::dsp_localization_trigger::window_rms;
use super::*;
use crate::srp_phat::{NoiseFloorRelativeReference, SolverReferenceChannelFloor};
use minimappr_core::stage1::noise_floor_relative_trigger::RelativeTriggerConfig;

/// No floor estimate yet: the node gets the trigger's own bootstrap, which
/// seeds a floor from the first level it hears (clamped at the digital-silence
/// minimum). The reporting node's frame is always observed before its solve
/// is built, so this only covers a member whose frames never reached the
/// trigger (stale-skipped, or LRU-evicted). Seeded from this window, such a
/// node can never lower the bar below its own current level, exactly as the
/// trigger never fires on a node's first frame.
pub(crate) fn solver_reference_channel_floor(
    layout: &SolveSensorLayout,
    active_channels: &[usize],
    channel_states: &[LocalizationChannelState],
    tracked_floor_rms_by_node: &BTreeMap<String, f32>,
    relative_db: f32,
) -> SolverReferenceChannelFloor {
    let silence_clamp_rms = RelativeTriggerConfig::default().minimum_rms as f32;
    let mut bootstrap_level_by_node: BTreeMap<&str, f32> = BTreeMap::new();
    for &channel in active_channels {
        let (Some(sensor), Some(state)) =
            (layout.sensors.get(channel), channel_states.get(channel))
        else {
            continue;
        };
        let level = bootstrap_level_by_node
            .entry(sensor.node_id.as_str())
            .or_insert(silence_clamp_rms);
        *level = level.max(window_rms(&state.window));
    }
    let quietest_node_floor_rms = bootstrap_level_by_node
        .into_iter()
        .map(|(node_id, bootstrap_level)| {
            tracked_floor_rms_by_node
                .get(node_id)
                .copied()
                .unwrap_or(bootstrap_level)
        })
        .reduce(f32::min);
    SolverReferenceChannelFloor {
        // No active channel leaves no floor to be relative to; SRP reports
        // the partial array before it reaches this check anyway.
        noise_floor_relative: quietest_node_floor_rms.map(|noise_floor_rms| {
            NoiseFloorRelativeReference {
                noise_floor_rms,
                relative_db,
            }
        }),
        ..SolverReferenceChannelFloor::default()
    }
}

#[cfg(test)]
#[path = "dsp_solver_reference_floor_tests.rs"]
mod dsp_solver_reference_floor_tests;
