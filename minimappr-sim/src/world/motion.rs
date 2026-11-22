//! Device motion generators: walking legs, micro-motion shifts and the
//! correlated (Ornstein–Uhlenbeck) GPS error track.

use std::f64::consts::TAU;

use rand_chacha::ChaCha8Rng;

use crate::physics::geometry::{add, Trajectory, Vec3, Waypoint};
use crate::scenario::chaos::{CanopyGpsKnob, MicroMotionKnob, WalkingKnob};
use crate::signals::noise::{standard_normal_approx, uniform_in_range};

fn random_horizontal_unit(rng: &mut ChaCha8Rng) -> Vec3 {
    let heading = uniform_in_range(rng, 0.0, TAU);
    [heading.cos(), heading.sin(), 0.0]
}

/// Walk/stop/walk… from `start_position_m`, beginning at `knob.start_s`.
pub fn walking_trajectory(
    start_position_m: Vec3,
    knob: &WalkingKnob,
    session_duration_s: f64,
    rng: &mut ChaCha8Rng,
) -> Trajectory {
    let mut waypoints = vec![Waypoint {
        time_s: 0.0,
        position_m: start_position_m,
    }];
    let mut time_s = knob.start_s.max(0.0);
    let mut position = start_position_m;
    waypoints.push(Waypoint {
        time_s,
        position_m: position,
    });
    while time_s < session_duration_s {
        let leg_s = uniform_in_range(
            rng,
            knob.leg_duration_range_s[0],
            knob.leg_duration_range_s[1],
        );
        let direction = random_horizontal_unit(rng);
        position = add(
            position,
            [
                direction[0] * knob.speed_mps * leg_s,
                direction[1] * knob.speed_mps * leg_s,
                0.0,
            ],
        );
        time_s += leg_s;
        waypoints.push(Waypoint {
            time_s,
            position_m: position,
        });
        time_s += knob.stop_duration_s;
        waypoints.push(Waypoint {
            time_s,
            position_m: position,
        });
    }
    Trajectory { waypoints }
}

/// Cumulative offset waypoints for micro-motion: each shift moves the
/// device 10–30 cm horizontally over `shift_duration_s`, no steps.
pub fn micro_motion_offsets(knob: &MicroMotionKnob, rng: &mut ChaCha8Rng) -> Vec<Waypoint> {
    let mut shift_times: Vec<f64> = (0..knob.shift_count)
        .map(|_| {
            uniform_in_range(
                rng,
                knob.start_s,
                (knob.end_s - knob.shift_duration_s).max(knob.start_s),
            )
        })
        .collect();
    shift_times.sort_by(f64::total_cmp);
    let mut offset = [0.0; 3];
    let mut waypoints = vec![Waypoint {
        time_s: 0.0,
        position_m: offset,
    }];
    let mut previous_end = f64::NEG_INFINITY;
    for start in shift_times {
        // Serialise overlapping shifts so the offset path stays monotone in time.
        let start = start.max(previous_end);
        waypoints.push(Waypoint {
            time_s: start,
            position_m: offset,
        });
        let magnitude = uniform_in_range(
            rng,
            knob.displacement_range_m[0],
            knob.displacement_range_m[1],
        );
        let direction = random_horizontal_unit(rng);
        offset = add(
            offset,
            [direction[0] * magnitude, direction[1] * magnitude, 0.0],
        );
        previous_end = start + knob.shift_duration_s;
        waypoints.push(Waypoint {
            time_s: previous_end,
            position_m: offset,
        });
    }
    waypoints
}

/// base(t) + offset(t), sampled at the union of both vertex sets (exact
/// for piecewise-linear inputs).
pub fn compose_with_offsets(base: &Trajectory, offsets: &[Waypoint]) -> Trajectory {
    let offset_track = Trajectory {
        waypoints: offsets.to_vec(),
    };
    let mut times: Vec<f64> = base
        .waypoints
        .iter()
        .chain(offsets)
        .map(|w| w.time_s)
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    Trajectory {
        waypoints: times
            .into_iter()
            .map(|time_s| Waypoint {
                time_s,
                position_m: add(base.position_at(time_s), offset_track.position_at(time_s)),
            })
            .collect(),
    }
}

/// Exact discretisation of a horizontal OU process with stationary σ and
/// correlation time τ: x_{k+1} = a x_k + σ√(1−a²) ε, a = exp(−Δt/τ).
/// Starts from the stationary distribution. Rows are `[t, dx, dy, 0]`.
pub fn gps_error_track(
    knob: &CanopyGpsKnob,
    session_duration_s: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<[f64; 4]> {
    let step = knob.gps_track_step_s.max(f64::EPSILON);
    let decay = (-step / knob.gps_error_correlation_time_s.max(f64::EPSILON)).exp();
    let innovation = knob.gps_error_sigma_m * (1.0 - decay * decay).sqrt();
    let mut error = [
        knob.gps_error_sigma_m * standard_normal_approx(rng) as f64,
        knob.gps_error_sigma_m * standard_normal_approx(rng) as f64,
    ];
    let steps = (session_duration_s / step).ceil() as usize;
    (0..=steps)
        .map(|k| {
            let row = [k as f64 * step, error[0], error[1], 0.0];
            error = [
                decay * error[0] + innovation * standard_normal_approx(rng) as f64,
                decay * error[1] + innovation * standard_normal_approx(rng) as f64,
            ];
            row
        })
        .collect()
}
