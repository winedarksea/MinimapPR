//! `manifest.json` in the shape `minimappr/calibration/pipeline.py`
//! writes and `bundle.load_bundle` / `replay.py` read.

use serde_json::{json, Value};

use super::geo::{FlatLocalFrame, COORDINATE_MODE_FLAT};
use super::{BUNDLE_KIND, BUNDLE_SCHEMA_VERSION, SIMULATOR_NAME};
use crate::physics::constants::NANOSECONDS_PER_SECOND;
use crate::scenario::devices::DeviceKind;
use crate::scenario::Scenario;
use crate::world::{PlannedDevice, WorldPlan};

const SECONDS_PER_DAY: i64 = 86_400;
const SECONDS_PER_HOUR: i64 = 3_600;
const SECONDS_PER_MINUTE: i64 = 60;

/// Proleptic-Gregorian civil date from days since 1970-01-01 (H. Hinnant's algorithm).
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

pub fn iso8601_utc_from_epoch_ns(epoch_ns: i64) -> String {
    let seconds = epoch_ns.div_euclid(NANOSECONDS_PER_SECOND as i64);
    let (year, month, day) = civil_from_days(seconds.div_euclid(SECONDS_PER_DAY));
    let second_of_day = seconds.rem_euclid(SECONDS_PER_DAY);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / SECONDS_PER_HOUR,
        (second_of_day % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE,
        second_of_day % SECONDS_PER_MINUTE
    )
}

pub fn audio_file_name(device_id: &str) -> String {
    format!("audio/{device_id}.wav")
}

pub fn channel_sensor_ids(device: &PlannedDevice) -> Vec<String> {
    (0..device.mic_offsets_m.len())
        .map(|i| format!("{}:ch{i}", device.device_id))
        .collect()
}

pub fn session_end_ns(scenario: &Scenario) -> i64 {
    scenario.site.start_time_ns + (scenario.duration_s * NANOSECONDS_PER_SECOND).round() as i64
}

fn node_manifest(device: &PlannedDevice, scenario: &Scenario, frame: &FlatLocalFrame) -> Value {
    let reported = device.reported_position_m;
    let geo = frame.local_to_geo(reported);
    json!({
        "node_id": device.device_id,
        "audio_file": audio_file_name(&device.device_id),
        "sample_rate_hz": scenario.sample_rate_hz,
        // The device believes its first sample is at session start; the true
        // offset/drift live in ground_truth.json → simulator.devices.
        "audio_start_time_ns": scenario.site.start_time_ns,
        "channel_sensor_ids": channel_sensor_ids(device),
        "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
        "position_m": reported,
        "sensor_offsets_m": device.mic_offsets_m,
        "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        "sync_diag": {"source": SIMULATOR_NAME},
        "device_kind": match device.kind {
            DeviceKind::SirithTetra => "sirith_tetra",
            DeviceKind::Phone => "phone",
        },
    })
}

pub fn build_manifest(plan: &WorldPlan, scenario: &Scenario, frame: &FlatLocalFrame) -> Value {
    let environment = &scenario.environment;
    json!({
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "session_id": format!("{}-seed{}", scenario.name, scenario.seed),
        // Deterministic (session start), so identical seeds give identical bundles.
        "created_utc": iso8601_utc_from_epoch_ns(scenario.site.start_time_ns),
        "time_window": {"start_ns": scenario.site.start_time_ns, "end_ns": session_end_ns(scenario)},
        "site": {
            "origin": {"lat": frame.origin.lat, "lon": frame.origin.lon, "alt_m": frame.origin.alt_m},
            "coordinate_mode": COORDINATE_MODE_FLAT,
        },
        "environment": {
            "temperature_c": environment.temperature_c,
            "humidity_fraction": environment.humidity_fraction,
            "pressure_kpa": environment.pressure_kpa,
            "speed_of_sound_mps": plan.speed_of_sound_mps,
            "source": "simulated",
        },
        "nodes": plan.devices.iter().map(|d| node_manifest(d, scenario, frame)).collect::<Vec<_>>(),
        "reference_audio": Value::Null,
        "simulator": {
            "name": SIMULATOR_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "scenario_name": scenario.name,
            "seed": scenario.seed,
        },
    })
}
