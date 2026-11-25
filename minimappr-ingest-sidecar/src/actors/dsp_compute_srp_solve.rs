//! The SRP-PHAT half of `run_math`: one payload's solve over its admitted
//! pairs, configured from the worker config and the payload's snapshots.

use crate::{
    dsp_worker::{solver_reference_channel_floor, ComputePayload},
    srp_phat::{
        estimate_tetrahedral_steering_with_solve_constraints, SolveChannelConstraints,
        SrpPhatConfig, SrpPhatEvaluation,
    },
};

#[cfg(test)]
thread_local! {
    /// `solve_id` of every SRP evaluation computed on this thread, so a test
    /// can show a repeated solve reused its evaluation instead of re-solving.
    pub(crate) static COMPUTED_SRP_SOLVE_IDS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The SRP-PHAT solve of `payload`'s localization window over its admitted pairs.
pub(super) fn solve_srp_phat(payload: &ComputePayload) -> SrpPhatEvaluation {
    #[cfg(test)]
    COMPUTED_SRP_SOLVE_IDS.with(|ids| ids.borrow_mut().push(payload.solve_id.clone()));
    let windows = payload
        .channel_states
        .iter()
        .map(|state| state.window.clone())
        .collect::<Vec<_>>();
    // Amplitude/SNR range prior (Phase 1c): when enabled, derive the projection
    // distance for unobservable-range solves from the reference-channel received
    // level instead of the fixed default. Ships off; both `None` reproduces the
    // fixed-prior behaviour. The Python re-solve path applies its own prior from
    // the manifest `received_level_dbfs`; this governs the Rust-native SRP solve.
    let (range_prior_m, range_prior_std_m) = if payload
        .config
        .localization_amplitude_range_prior_enabled
    {
        match max_window_rms(&payload.channel_states) {
            Some(rms) => {
                let received_level_db = (20.0 * (rms.max(1.0e-9)).log10()) as f32;
                let (prior_range_m, _clamped) = crate::range_projection::amplitude_range_prior_m(
                    received_level_db,
                    payload.config.localization_amplitude_reference_level_db,
                    payload.config.localization_amplitude_prior_min_range_m,
                    payload.config.localization_amplitude_prior_max_range_m,
                );
                (
                    Some(prior_range_m),
                    Some(payload.config.localization_amplitude_prior_std_factor * prior_range_m),
                )
            }
            None => (None, None),
        }
    } else {
        (None, None)
    };
    // Only the reporter's admitted, connected sensors: a timeline-rejected
    // node's channels would otherwise stay in the centroid, aperture and
    // grid although none of its pairs with the reporter is measured.
    let solve_channels = payload.pair_admission.solve_sensor_indices();
    let srp_config = SrpPhatConfig {
        band_policy: payload.config.localization_band_policy,
        grid_resolution_m: payload.config.localization_srp_grid_resolution_m,
        search_padding_m: payload.config.localization_search_padding_m,
        interp_factor: payload.config.gcc_phat_interp_factor,
        far_field_default_range_m: payload.config.localization_far_field_default_range_m,
        far_field_max_range_m: payload.config.localization_far_field_max_range_m,
        range_prior_m,
        range_prior_std_m,
        half_space: payload.sensor_layout.solve_half_space(solve_channels),
        reference_channel_floor: solver_reference_channel_floor(
            &payload.sensor_layout,
            solve_channels,
            &payload.channel_states,
            &payload.node_noise_floor_rms,
            payload
                .config
                .localization_solve
                .localization_solver_floor_relative_db,
        ),
        ..SrpPhatConfig::default()
    };
    let admits_pair = |ch_a: usize, ch_b: usize| payload.pair_admission.admits_pair(ch_a, ch_b);
    estimate_tetrahedral_steering_with_solve_constraints(
        windows.as_slice(),
        solve_channels,
        payload.sensor_layout.positions_m.as_slice(),
        payload.sr,
        payload.effective_sound_speed_mps,
        srp_config,
        SolveChannelConstraints {
            admitted_pair_filter: Some(&admits_pair),
            channel_clock_offsets: Some(payload.pair_admission.channel_clock_offsets()),
            channel_node_position_std: Some(payload.pair_admission.channel_node_position_std()),
            array_index_by_channel: Some(payload.pair_admission.sensor_node_indices()),
        },
    )
}

pub(super) fn max_window_rms(
    channel_states: &[crate::dsp_worker::LocalizationChannelState],
) -> Option<f64> {
    channel_states
        .iter()
        .filter_map(|state| rms_for_window(&state.window))
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn rms_for_window(window: &[f32]) -> Option<f64> {
    if window.is_empty() {
        return None;
    }
    let energy = window
        .iter()
        .map(|sample| {
            let sample64 = f64::from(*sample);
            sample64 * sample64
        })
        .sum::<f64>();
    Some((energy / window.len() as f64).sqrt())
}
