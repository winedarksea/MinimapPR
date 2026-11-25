//! Session tuning the shell sets at start: chirp schedule, band, waveform,
//! lead time and air temperature, mapped onto the core `MeshConfig`. Both
//! phones must use the same tuning (the schedule carries only the band).

use minimappr_core::mesh::config::{ChirpScheduleMode, MeshConfig};
use minimappr_core::sync::chirp_waveform::ChirpWaveformKind;

use crate::chirp_waveforms::{validate_waveform_spec, FieldChirpWaveformKind, FieldChirpWaveformSpec, FieldStereoChannel};
use crate::error::FieldFfiError;

/// Speed of sound in dry air at 0 °C and the ideal-gas temperature law
/// c = c0 · sqrt(1 + T/273.15); humidity adds < 0.4 % and is ignored.
const SOUND_SPEED_AT_ZERO_CELSIUS_MPS: f64 = 331.3;
const ZERO_CELSIUS_IN_KELVIN: f64 = 273.15;
const MINIMUM_AIR_TEMPERATURE_C: f64 = -40.0;
const MAXIMUM_AIR_TEMPERATURE_C: f64 = 60.0;
/// Below this a round cannot finish before the next schedule is due.
const MINIMUM_PERIODIC_INTERVAL_MS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldChirpScheduleMode {
    SilentUntilNeeded,
    Periodic,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldMeshTuning {
    pub schedule_mode: FieldChirpScheduleMode,
    pub periodic_round_interval_ms: u64,
    /// Schedule → first slot. Wi-Fi power save needs about 1 s.
    pub schedule_lead_time_ms: u64,
    pub slot_duration_ms: u64,
    pub coarse_schedule_uncertainty_ms: u64,
    pub waveform: FieldChirpWaveformSpec,
    pub air_temperature_c: f64,
    /// Playback channel; `None` takes the per-model table.
    pub chirp_output_channel: Option<FieldStereoChannel>,
    pub chirp_peak_dbfs: f64,
}

pub fn speed_of_sound_mps(air_temperature_c: f64) -> f64 {
    SOUND_SPEED_AT_ZERO_CELSIUS_MPS * (1.0 + air_temperature_c / ZERO_CELSIUS_IN_KELVIN).sqrt()
}

fn core_waveform_kind(kind: FieldChirpWaveformKind) -> ChirpWaveformKind {
    match kind {
        FieldChirpWaveformKind::UpDownHyperbolic => ChirpWaveformKind::UpDownHyperbolic,
        FieldChirpWaveformKind::HyperbolicFm => ChirpWaveformKind::HyperbolicFm,
        FieldChirpWaveformKind::LinearFm => ChirpWaveformKind::LinearFm,
    }
}

impl FieldMeshTuning {
    pub fn validate(&self) -> Result<(), FieldFfiError> {
        validate_waveform_spec(&self.waveform)?;
        let invalid = |detail: String| Err(FieldFfiError::InvalidArgument { detail });
        if !(MINIMUM_AIR_TEMPERATURE_C..=MAXIMUM_AIR_TEMPERATURE_C).contains(&self.air_temperature_c) {
            return invalid(format!("air temperature {} °C out of range", self.air_temperature_c));
        }
        let chirp_ms = (self.waveform.duration_s * 1_000.0).ceil() as u64;
        if self.slot_duration_ms <= chirp_ms + 2 * self.coarse_schedule_uncertainty_ms {
            return invalid(format!(
                "slot {} ms leaves no flight time after a {chirp_ms} ms chirp and ±{} ms uncertainty",
                self.slot_duration_ms, self.coarse_schedule_uncertainty_ms
            ));
        }
        if self.schedule_mode == FieldChirpScheduleMode::Periodic && self.periodic_round_interval_ms < MINIMUM_PERIODIC_INTERVAL_MS {
            return invalid(format!("periodic interval below {MINIMUM_PERIODIC_INTERVAL_MS} ms"));
        }
        if !self.chirp_peak_dbfs.is_finite() || self.chirp_peak_dbfs > 0.0 {
            return invalid(format!("chirp peak {} dBFS must be <= 0", self.chirp_peak_dbfs));
        }
        Ok(())
    }

    pub fn speed_of_sound_mps(&self) -> f64 {
        speed_of_sound_mps(self.air_temperature_c)
    }

    pub fn apply_to(&self, config: &mut MeshConfig) {
        let scheduler = &mut config.scheduler;
        scheduler.mode = match self.schedule_mode {
            FieldChirpScheduleMode::SilentUntilNeeded => ChirpScheduleMode::SilentUntilNeeded,
            FieldChirpScheduleMode::Periodic => ChirpScheduleMode::Periodic,
        };
        scheduler.periodic_round_interval_ms = self.periodic_round_interval_ms;
        scheduler.schedule_lead_time_ms = self.schedule_lead_time_ms;
        scheduler.slot_duration_ms = self.slot_duration_ms;
        scheduler.coarse_schedule_uncertainty_ms = self.coarse_schedule_uncertainty_ms;
        scheduler.chirp_band_hz = [self.waveform.band_low_hz as f32, self.waveform.band_high_hz as f32];
        scheduler.chirp_waveform_kind = core_waveform_kind(self.waveform.kind);
        scheduler.chirp_duration_s = self.waveform.duration_s;
        config.island_sync.sound_speed_mps = self.speed_of_sound_mps();
    }

    pub fn mesh_config(&self) -> Result<MeshConfig, FieldFfiError> {
        self.validate()?;
        let mut config = MeshConfig::default();
        self.apply_to(&mut config);
        Ok(config)
    }
}

/// Defaults for a live LAN session: periodic rounds and a Wi-Fi lead time.
#[uniffi::export]
pub fn default_field_mesh_tuning() -> FieldMeshTuning {
    let scheduler = MeshConfig::default().scheduler;
    FieldMeshTuning {
        schedule_mode: FieldChirpScheduleMode::Periodic,
        periodic_round_interval_ms: 5_000,
        schedule_lead_time_ms: 1_000,
        slot_duration_ms: scheduler.slot_duration_ms,
        coarse_schedule_uncertainty_ms: scheduler.coarse_schedule_uncertainty_ms,
        waveform: FieldChirpWaveformSpec {
            kind: FieldChirpWaveformKind::UpDownHyperbolic,
            band_low_hz: f64::from(scheduler.chirp_band_hz[0]),
            band_high_hz: f64::from(scheduler.chirp_band_hz[1]),
            duration_s: scheduler.chirp_duration_s,
        },
        air_temperature_c: 20.0,
        chirp_output_channel: None,
        chirp_peak_dbfs: -3.0,
    }
}

#[uniffi::export]
pub fn speed_of_sound_for_temperature(air_temperature_c: f64) -> f64 {
    speed_of_sound_mps(air_temperature_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sound_speed_follows_temperature() {
        assert!((speed_of_sound_mps(20.0) - 343.2).abs() < 0.2);
        assert!((speed_of_sound_mps(0.0) - SOUND_SPEED_AT_ZERO_CELSIUS_MPS).abs() < 1e-9);
    }

    #[test]
    fn defaults_validate_and_map_to_periodic() {
        let config = default_field_mesh_tuning().mesh_config().unwrap();
        assert_eq!(config.scheduler.mode, ChirpScheduleMode::Periodic);
        assert_eq!(config.scheduler.schedule_lead_time_ms, 1_000);
        assert!((config.island_sync.sound_speed_mps - 343.2).abs() < 0.2);
    }

    #[test]
    fn slot_too_short_for_the_chirp_is_rejected() {
        let tuning = FieldMeshTuning { slot_duration_ms: 100, ..default_field_mesh_tuning() };
        assert!(tuning.validate().is_err());
    }
}
