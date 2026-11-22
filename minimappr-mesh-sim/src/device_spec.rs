//! What a scenario says about each simulated device: platform, abilities,
//! where it walks, and which chaos knobs apply to it.

use minimappr_core::protocol::session_messages::NodeDescriptor;
use minimappr_core::protocol::wire_codes::NodeKind;

const PHONE_CAPTURE_SAMPLE_RATE_HZ: u32 = 48_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePlatform {
    Android,
    IosNative,
    IosPwa,
}

/// Piecewise-linear path; holds the last waypoint afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    pub waypoints: Vec<(f64, [f64; 3])>,
}

impl Trajectory {
    pub fn stationary(position_m: [f64; 3]) -> Self {
        Self { waypoints: vec![(0.0, position_m)] }
    }

    /// Append a leg: hold until `start_s`, then move to `to` by `end_s`.
    pub fn then_move(mut self, start_s: f64, end_s: f64, to: [f64; 3]) -> Self {
        let held = self.position_at(start_s);
        self.waypoints.push((start_s, held));
        self.waypoints.push((end_s, to));
        self
    }

    pub fn position_at(&self, time_s: f64) -> [f64; 3] {
        let first = self.waypoints[0];
        if time_s <= first.0 {
            return first.1;
        }
        for pair in self.waypoints.windows(2) {
            let ((t0, p0), (t1, p1)) = (pair[0], pair[1]);
            if time_s <= t1 {
                let span = (t1 - t0).max(f64::MIN_POSITIVE);
                let fraction = ((time_s - t0) / span).clamp(0.0, 1.0);
                return std::array::from_fn(|axis| p0[axis] + fraction * (p1[axis] - p0[axis]));
            }
        }
        self.waypoints.last().expect("non-empty").1
    }

    pub fn speed_at(&self, time_s: f64) -> f64 {
        const SPEED_PROBE_S: f64 = 0.05;
        let (a, b) = (self.position_at(time_s - SPEED_PROBE_S), self.position_at(time_s + SPEED_PROBE_S));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() / (2.0 * SPEED_PROBE_S)
    }
}

/// Thermal headroom: 1 (cool) ramps linearly to 0 (throttling) over
/// `[start_s, end_s]`, then recovers linearly by `recovered_s` (a phone
/// that shed its classifier cools down again).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalRamp {
    pub start_s: f64,
    pub end_s: f64,
    pub recovered_s: f64,
}

impl ThermalRamp {
    pub fn headroom_at(&self, time_s: f64) -> f64 {
        if time_s <= self.end_s {
            return (1.0 - (time_s - self.start_s) / (self.end_s - self.start_s)).clamp(0.0, 1.0);
        }
        ((time_s - self.end_s) / (self.recovered_s - self.end_s).max(f64::MIN_POSITIVE)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSpec {
    pub platform: DevicePlatform,
    pub can_classify: bool,
    pub can_host: bool,
    /// Stereo phone or array node: reports a world bearing of chirps it hears.
    pub reports_chirp_direction: bool,
    pub trajectory: Trajectory,
    /// Speaker→ranging-mic path d_ii (per-model table value).
    pub self_path_length_m: f64,
    pub muted_speaker: bool,
    pub gps_error_sigma_m: f64,
    pub thermal_ramp: Option<ThermalRamp>,
    /// Extra in-band attenuation (pocket), dB.
    pub placement_attenuation_db: f64,
}

impl DeviceSpec {
    pub fn android_at(position_m: [f64; 3]) -> Self {
        Self {
            platform: DevicePlatform::Android,
            can_classify: true,
            can_host: true,
            reports_chirp_direction: false,
            trajectory: Trajectory::stationary(position_m),
            self_path_length_m: 0.02,
            muted_speaker: false,
            gps_error_sigma_m: 3.0,
            thermal_ramp: None,
            placement_attenuation_db: 0.0,
        }
    }

    pub fn ios_native_at(position_m: [f64; 3]) -> Self {
        Self {
            platform: DevicePlatform::IosNative,
            can_host: false,
            ..Self::android_at(position_m)
        }
    }

    pub fn platform_name(&self) -> String {
        let platform = match self.platform {
            DevicePlatform::Android => "sim-android",
            DevicePlatform::IosNative => "sim-ios",
            DevicePlatform::IosPwa => "sim-ios-pwa",
        };
        format!("{platform}-dii-{:.3}", self.self_path_length_m)
    }

    pub fn node_descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            node_kind: match self.platform {
                DevicePlatform::IosPwa => NodeKind::BrowserPwa,
                _ => NodeKind::Phone,
            },
            platform_name: self.platform_name(),
            app_version: "mesh-sim".into(),
            capture_sample_rate_hz: PHONE_CAPTURE_SAMPLE_RATE_HZ,
            microphone_count: 1,
            microphone_positions_body_frame_m: Vec::new(),
            can_classify: self.can_classify,
            can_host: self.can_host,
            can_emit_chirps: true,
            has_pps_clock: false,
        }
    }
}
