//! Stage-1 configuration per detection mode.
//!
//! Every threshold is *relative to the running per-band noise floor*: there
//! is deliberately no absolute level gate, so a faint distant call passes as
//! long as it stands out of its own background (AGENTS.md: never block faint
//! long-range detections with RMS gates).

/// Which band layout and trigger band Stage 1 uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage1DetectionMode {
    /// History 0.5-12 kHz, trigger band 1.5-10 kHz.
    Birds,
    /// History 60 Hz-6 kHz (rotor fundamentals and harmonics), trigger band
    /// 80 Hz-4 kHz. Pair with the flat-to-DC decimation filter.
    Drone,
}

/// Tunables for `Stage1Detector`. Built with `for_mode`, then adjusted.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage1Configuration {
    pub detection_mode: Stage1DetectionMode,
    pub input_sample_rate_hz: u32,
    /// 20 ms at 48 kHz: one history frame per hop.
    pub hop_length_samples: usize,
    /// FFT/analysis window length (Hann). Longer in drone mode so the
    /// narrow low bands each get at least one FFT bin.
    pub analysis_window_samples: usize,
    pub band_count: usize,
    pub history_lowest_band_edge_hz: f64,
    pub history_highest_band_edge_hz: f64,
    /// Low bands are widened to at least this many FFT bins (mel-like), so
    /// no band is a 1-2 bin chi-square that flickers in stationary noise.
    pub minimum_band_width_bins: usize,
    /// Bands whose centre lies in this range feed the trigger.
    pub trigger_band_low_hz: f64,
    pub trigger_band_high_hz: f64,
    /// Ring-buffer length of the u8 band history (30 s * 50 fps * 40 B = 60 KB).
    pub history_duration_seconds: f64,
    /// History quantization: byte value v means `v * step + floor` dBFS.
    pub history_quantization_step_db: f64,
    pub history_quantization_floor_dbfs: f64,
    /// Trigger contrast uses band energies averaged over this many frames;
    /// averaging lifts the effective degrees of freedom of narrow low bands
    /// so stationary noise does not flicker past the threshold.
    pub trigger_smoothing_frames: usize,
    /// Noise floor: first frames converge fast from the first observation.
    pub noise_floor_warmup_frames: u64,
    pub noise_floor_warmup_smoothing: f64,
    /// Symmetric EMA while a band sits within +/- the update gate of its floor.
    pub noise_floor_tracking_smoothing: f64,
    /// Very slow rise while a band stands above the gate (a call must not lift
    /// its own floor, but a new stationary source is absorbed in ~20 s).
    pub noise_floor_rise_while_active_smoothing: f64,
    /// Faster fall when a band drops more than the gate below its floor.
    pub noise_floor_fall_smoothing: f64,
    pub noise_floor_update_gate_db: f64,
    /// Contrast = mean of the K largest per-band excesses (dB over floor) in
    /// the trigger band: tonal calls light only a few bands.
    pub contrast_top_band_count: usize,
    pub contrast_open_threshold_db: f64,
    /// Close below `open - hysteresis` (stops chatter at the threshold).
    pub contrast_close_hysteresis_db: f64,
    pub contrast_open_minimum_frames: usize,
    pub trigger_release_frames: usize,
    /// Flux is computed on band energies averaged over this many frames: 2
    /// halves per-frame chi-square flicker but keeps onsets within a hop.
    pub flux_smoothing_frames: usize,
    /// Per-band excess below this counts as zero in spectral flux, so
    /// stationary noise contributes ~0 flux.
    pub flux_excess_noise_allowance_db: f64,
    /// Positive-flux peak needed to mark a note onset.
    pub onset_flux_threshold_db: f64,
    /// A sharp onset at or above this opens the trigger without persistence.
    pub onset_opens_trigger_flux_db: f64,
    pub onset_minimum_spacing_frames: u64,
    /// Bounded onset list (oldest dropped and counted).
    pub onset_list_capacity: usize,
    /// Floor for the broadband noise RMS used by noise fill (1 PCM16 LSB in
    /// float units): digital silence still gets non-zero-variance fill.
    pub minimum_noise_fill_rms: f32,
}

const DEVICE_SAMPLE_RATE_HZ: u32 = 48_000;
const HOP_LENGTH_SAMPLES_20_MS_AT_48_KHZ: usize = 960;
const PCM16_ONE_LSB_FULL_SCALE_FRACTION: f32 = 1.0 / 32_768.0;

impl Stage1Configuration {
    pub fn for_mode(detection_mode: Stage1DetectionMode) -> Self {
        let common = Self {
            detection_mode,
            input_sample_rate_hz: DEVICE_SAMPLE_RATE_HZ,
            hop_length_samples: HOP_LENGTH_SAMPLES_20_MS_AT_48_KHZ,
            analysis_window_samples: 2048,
            band_count: 40,
            history_lowest_band_edge_hz: 500.0,
            history_highest_band_edge_hz: 12_000.0,
            minimum_band_width_bins: 4,
            trigger_band_low_hz: 1_500.0,
            trigger_band_high_hz: 10_000.0,
            history_duration_seconds: 30.0,
            history_quantization_step_db: 0.5,
            history_quantization_floor_dbfs: -127.5,
            trigger_smoothing_frames: 3,
            noise_floor_warmup_frames: 25,
            noise_floor_warmup_smoothing: 0.2,
            noise_floor_tracking_smoothing: 0.02,
            noise_floor_rise_while_active_smoothing: 0.001,
            noise_floor_fall_smoothing: 0.1,
            noise_floor_update_gate_db: 3.0,
            contrast_top_band_count: 1,
            contrast_open_threshold_db: 7.0,
            contrast_close_hysteresis_db: 2.0,
            contrast_open_minimum_frames: 2,
            trigger_release_frames: 10,
            flux_smoothing_frames: 2,
            flux_excess_noise_allowance_db: 3.0,
            onset_flux_threshold_db: 6.0,
            onset_opens_trigger_flux_db: 12.0,
            onset_minimum_spacing_frames: 2,
            onset_list_capacity: 512,
            minimum_noise_fill_rms: PCM16_ONE_LSB_FULL_SCALE_FRACTION,
        };
        match detection_mode {
            Stage1DetectionMode::Birds => common,
            Stage1DetectionMode::Drone => Self {
                analysis_window_samples: 8192,
                band_count: 32,
                history_lowest_band_edge_hz: 60.0,
                history_highest_band_edge_hz: 6_000.0,
                minimum_band_width_bins: 8,
                trigger_band_low_hz: 80.0,
                trigger_band_high_hz: 4_000.0,
                // 8192-sample windows overlap ~88% at a 20 ms hop, so frames
                // are nearly redundant: smooth over 160 ms to gain real
                // degrees of freedom. Rotor tones are steady, so the extra
                // latency and persistence cost nothing.
                trigger_smoothing_frames: 8,
                contrast_open_minimum_frames: 5,
                ..common
            },
        }
    }

    pub fn history_capacity_frames(&self) -> usize {
        let frames_per_second =
            f64::from(self.input_sample_rate_hz) / self.hop_length_samples as f64;
        (self.history_duration_seconds * frames_per_second).ceil() as usize
    }
}
