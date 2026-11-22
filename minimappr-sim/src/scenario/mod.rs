//! Scenario description (serde / TOML). Everything that varies between
//! test cases is a field here; the renderer has no hidden tunables.

pub mod chaos;
pub mod devices;
pub mod sources;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::physics::atmosphere::AtmosphericConditions;
use crate::physics::constants::STANDARD_ATMOSPHERE_PRESSURE_KPA;
use crate::physics::fractional_delay::FractionalDelayKernelConfig;
use crate::physics::geometry::Vec3;
pub use chaos::ChaosConfig;
pub use devices::{DeviceConfig, DeviceKind};
pub use sources::{CallConfig, ChirpScheduleConfig, SourceConfig};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SiteConfig {
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub origin_alt_m: f64,
    /// Wall-clock epoch (ns) of true session time 0.
    pub start_time_ns: i64,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            origin_lat: 45.0,
            origin_lon: -93.0,
            origin_alt_m: 250.0,
            start_time_ns: 1_763_700_000_000_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentConfig {
    pub temperature_c: f64,
    pub humidity_fraction: f64,
    pub pressure_kpa: f64,
    /// Disable to test pure geometry (delay/Doppler) without air colouring.
    pub absorption_enabled: bool,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            humidity_fraction: 0.5,
            pressure_kpa: STANDARD_ATMOSPHERE_PRESSURE_KPA,
            absorption_enabled: true,
        }
    }
}

impl EnvironmentConfig {
    pub fn atmospheric_conditions(&self) -> AtmosphericConditions {
        AtmosphericConditions {
            temperature_c: self.temperature_c,
            relative_humidity_fraction: self.humidity_fraction,
            pressure_kpa: self.pressure_kpa,
        }
    }
}

/// Flat ground plane producing one image-source reflection per path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundConfig {
    pub enabled: bool,
    pub height_m: f64,
    /// Broadband pressure reflection coefficient (soft forest floor ≈ 0.5–0.7).
    pub reflection_coefficient: f64,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            height_m: 0.0,
            reflection_coefficient: 0.6,
        }
    }
}

/// Point reflector (tree trunk): path = source → reflector → receiver with
/// 1/(r1 + r2) spreading times `reflection_coefficient`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReflectorConfig {
    #[serde(default)]
    pub reflector_id: Option<String>,
    pub position_m: Vec3,
    pub reflection_coefficient: f64,
}

/// Random trunks around the device centroid; coefficient drawn in dB.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RandomTrunksConfig {
    pub count_range: [usize; 2],
    /// Trunks placed at this distance band from the device centroid, which
    /// lands their extra path in the 3–25 ms-late range for typical scenes.
    pub distance_range_m: [f64; 2],
    pub height_m: f64,
    pub coefficient_db_range: [f64; 2],
}

impl Default for RandomTrunksConfig {
    fn default() -> Self {
        Self {
            count_range: [3, 6],
            distance_range_m: [2.0, 8.0],
            height_m: 1.5,
            coefficient_db_range: [-18.0, -6.0],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReflectorsConfig {
    pub explicit: Vec<ReflectorConfig>,
    pub random_trunks: Option<RandomTrunksConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AmbientNoiseConfig {
    pub enabled: bool,
    /// Broadband ambient level, dB SPL (independent per mic).
    pub level_db_spl: f64,
    /// One-pole low-pass corner giving the forest/wind spectral tilt.
    pub spectral_tilt_corner_hz: f64,
}

impl Default for AmbientNoiseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level_db_spl: 35.0,
            spectral_tilt_corner_hz: 800.0,
        }
    }
}

/// Renderer accuracy/speed trade-offs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    /// Retarded-time solves happen every N output samples and are linearly
    /// interpolated between (exact for static geometry + drift).
    pub control_point_spacing_samples: usize,
    /// 1/r is clamped below this distance.
    pub minimum_path_length_m: f64,
    /// Zero padding around each source buffer for the absorption FFT filter.
    pub absorption_guard_samples: usize,
    pub kernel_half_width_taps: usize,
    pub kernel_phase_count: usize,
    pub kernel_kaiser_beta: f64,
    pub kernel_cutoff_fraction_of_nyquist: f64,
    /// Crossfade at time-gated receiver filters (pocket, route change).
    pub gate_crossfade_s: f64,
    /// Receive-chain segments start this early so IIR states converge.
    pub chain_segment_warmup_s: f64,
    /// Paths of one emission share an absorption filter when their lengths
    /// differ by less than this much attenuation at the emission's band top
    /// (99.9 % energy frequency). 0 = exact filter per path.
    pub absorption_sharing_tolerance_db: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        let kernel = FractionalDelayKernelConfig::default();
        Self {
            control_point_spacing_samples: 32,
            minimum_path_length_m: 0.01,
            absorption_guard_samples: 512,
            kernel_half_width_taps: kernel.half_width_taps,
            kernel_phase_count: kernel.phase_count,
            kernel_kaiser_beta: kernel.kaiser_beta,
            kernel_cutoff_fraction_of_nyquist: kernel.cutoff_fraction_of_nyquist,
            gate_crossfade_s: 0.005,
            chain_segment_warmup_s: 0.1,
            absorption_sharing_tolerance_db: 0.1,
        }
    }
}

impl RenderConfig {
    pub fn kernel_config(&self) -> FractionalDelayKernelConfig {
        FractionalDelayKernelConfig {
            half_width_taps: self.kernel_half_width_taps,
            phase_count: self.kernel_phase_count,
            kaiser_beta: self.kernel_kaiser_beta,
            cutoff_fraction_of_nyquist: self.kernel_cutoff_fraction_of_nyquist,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    pub seed: u64,
    pub duration_s: f64,
    pub sample_rate_hz: u32,
    pub site: SiteConfig,
    pub environment: EnvironmentConfig,
    pub ground: GroundConfig,
    pub reflectors: ReflectorsConfig,
    pub ambient_noise: AmbientNoiseConfig,
    pub devices: Vec<DeviceConfig>,
    pub sources: Vec<SourceConfig>,
    pub chirps: Option<ChirpScheduleConfig>,
    pub chaos: ChaosConfig,
    pub render: RenderConfig,
    /// Passed through verbatim as the bundle's expectations.json.
    pub expectations: Option<toml::Value>,
    /// Directory relative WAV paths resolve against (set by `from_path`).
    #[serde(skip)]
    pub base_directory: PathBuf,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            name: "sim".into(),
            seed: 0,
            duration_s: 10.0,
            sample_rate_hz: 48_000,
            site: SiteConfig::default(),
            environment: EnvironmentConfig::default(),
            ground: GroundConfig::default(),
            reflectors: ReflectorsConfig::default(),
            ambient_noise: AmbientNoiseConfig::default(),
            devices: vec![],
            sources: vec![],
            chirps: None,
            chaos: ChaosConfig::default(),
            render: RenderConfig::default(),
            expectations: None,
            base_directory: PathBuf::from("."),
        }
    }
}

impl Scenario {
    pub fn from_toml_str(text: &str) -> Result<Self, String> {
        let scenario: Scenario = toml::from_str(text).map_err(|e| format!("scenario TOML: {e}"))?;
        scenario.validate()?;
        Ok(scenario)
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut scenario = Self::from_toml_str(&text)?;
        scenario.base_directory = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(scenario)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.duration_s.is_finite() && self.duration_s > 0.0) {
            return Err("duration_s must be positive and finite".into());
        }
        if self.sample_rate_hz == 0 {
            return Err("sample_rate_hz must be positive".into());
        }
        if self.devices.is_empty() {
            return Err("scenario needs at least one device".into());
        }
        let mut seen = std::collections::HashSet::new();
        for device in &self.devices {
            if !seen.insert(device.device_id.as_str()) {
                return Err(format!("duplicate device_id {}", device.device_id));
            }
            if device.second_mic.is_some() && device.kind != DeviceKind::Phone {
                return Err(format!(
                    "{}: second_mic is only valid on phones",
                    device.device_id
                ));
            }
        }
        if self.render.control_point_spacing_samples == 0
            || self.render.kernel_phase_count == 0
            || self.render.kernel_half_width_taps == 0
        {
            return Err(
                "render spacing, kernel phase count and kernel width must be positive".into(),
            );
        }
        self.chaos.validate()?;
        let cutoff = self.render.kernel_cutoff_fraction_of_nyquist;
        if !(cutoff > 0.0 && cutoff <= 1.0) {
            return Err("render.kernel_cutoff_fraction_of_nyquist must be in (0, 1]".into());
        }
        Ok(())
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate_hz as f64
    }
}
