//! Streaming Stage-1 detector: 20 ms sub-band history, relative energy
//! contrast + positive spectral flux trigger, and a note-onset list.
//!
//! Sans-IO: callers push contiguous device-clock audio blocks and get events
//! back; nothing here reads a clock. Frame `f` analyses the window ending at
//! device sample `stream_origin + (f + 1) * hop`; onsets are stamped at the
//! start of their newest hop (+/- one hop, 20 ms: the host refines timing by
//! correlation, the onset only anchors the frame).

use std::collections::VecDeque;

use super::band_analyzer::{SubBandEnergyAnalyzer, SubBandLayout};
use super::band_history::QuantizedBandHistory;
use super::configuration::Stage1Configuration;
use super::noise_floor::{energy_ratio_db, GatedNoiseFloorTracker, NoiseFloorSmoothing};

/// A positive-flux peak: a note onset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoteOnset {
    pub frame_index: u64,
    pub device_sample_index: u64,
    pub positive_flux_db: f64,
}

/// Trigger transitions and onsets produced by one `process_block` call.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage1Event {
    TriggerOpened {
        frame_index: u64,
        device_sample_index: u64,
        contrast_db: f64,
        active_band_low_hz: f64,
        active_band_high_hz: f64,
    },
    TriggerClosed {
        frame_index: u64,
        device_sample_index: u64,
    },
    Onset(NoteOnset),
}

/// Counters for the Stage-1 miss-rate audit (plan: rejects are counted).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stage1Counters {
    pub frames_evaluated: u64,
    pub frames_while_triggered: u64,
    pub frames_rejected: u64,
    pub triggers_opened: u64,
    pub onsets_detected: u64,
    pub onsets_dropped_from_full_list: u64,
    pub stream_discontinuities: u64,
}

/// Last evaluated frame's trigger statistics (field tuning / logging).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stage1FrameDiagnostics {
    pub contrast_db: f64,
    pub positive_flux_db: f64,
}

pub struct Stage1Detector {
    configuration: Stage1Configuration,
    analyzer: SubBandEnergyAnalyzer,
    history: QuantizedBandHistory,
    noise_floor: GatedNoiseFloorTracker,
    trigger_band_indices: Vec<usize>,
    analysis_buffer: VecDeque<f32>,
    stream_origin_sample_index: Option<u64>,
    next_expected_sample_index: u64,
    samples_since_last_frame: usize,
    recent_band_energies: VecDeque<Vec<f64>>,
    /// Excess rows of the last `flux_smoothing_frames` frames (oldest first):
    /// flux compares against the row one smoothing span back, so the two
    /// averaging windows do not overlap and an onset shows its full rise.
    lagged_band_excess_db: VecDeque<Vec<f64>>,
    flux_two_frames_ago: f64,
    flux_one_frame_ago: f64,
    last_onset_frame_index: Option<u64>,
    onsets: VecDeque<NoteOnset>,
    trigger_is_open: bool,
    frames_above_open_threshold: usize,
    frames_below_close_threshold: usize,
    counters: Stage1Counters,
    last_frame_diagnostics: Stage1FrameDiagnostics,
    scratch_band_energies: Vec<f64>,
}

impl Stage1Detector {
    pub fn new(configuration: Stage1Configuration) -> Self {
        let analyzer = SubBandEnergyAnalyzer::new(&configuration);
        let trigger_band_indices = analyzer
            .band_layouts()
            .iter()
            .enumerate()
            .filter(|(_, layout)| {
                let centre_hz = layout.centre_frequency_hz();
                centre_hz >= configuration.trigger_band_low_hz
                    && centre_hz <= configuration.trigger_band_high_hz
            })
            .map(|(band_index, _)| band_index)
            .collect();
        Self {
            history: QuantizedBandHistory::new(
                configuration.band_count,
                configuration.history_capacity_frames(),
                configuration.history_quantization_step_db,
                configuration.history_quantization_floor_dbfs,
            ),
            noise_floor: GatedNoiseFloorTracker::new(
                configuration.band_count,
                NoiseFloorSmoothing::from_configuration(&configuration),
            ),
            trigger_band_indices,
            analysis_buffer: VecDeque::with_capacity(configuration.analysis_window_samples),
            stream_origin_sample_index: None,
            next_expected_sample_index: 0,
            samples_since_last_frame: 0,
            recent_band_energies: VecDeque::with_capacity(configuration.trigger_smoothing_frames),
            lagged_band_excess_db: VecDeque::with_capacity(configuration.flux_smoothing_frames),
            flux_two_frames_ago: 0.0,
            flux_one_frame_ago: 0.0,
            last_onset_frame_index: None,
            onsets: VecDeque::new(),
            trigger_is_open: false,
            frames_above_open_threshold: 0,
            frames_below_close_threshold: 0,
            counters: Stage1Counters::default(),
            last_frame_diagnostics: Stage1FrameDiagnostics::default(),
            scratch_band_energies: Vec::with_capacity(configuration.band_count),
            analyzer,
            configuration,
        }
    }

    pub fn configuration(&self) -> &Stage1Configuration {
        &self.configuration
    }

    pub fn band_layouts(&self) -> &[SubBandLayout] {
        self.analyzer.band_layouts()
    }

    pub fn history(&self) -> &QuantizedBandHistory {
        &self.history
    }

    pub fn counters(&self) -> Stage1Counters {
        self.counters
    }

    pub fn last_frame_diagnostics(&self) -> Stage1FrameDiagnostics {
        self.last_frame_diagnostics
    }

    pub fn trigger_is_open(&self) -> bool {
        self.trigger_is_open
    }

    /// Bounded list of recent onsets (oldest first).
    pub fn onsets(&self) -> impl Iterator<Item = &NoteOnset> {
        self.onsets.iter()
    }

    /// Onsets whose device sample index falls in `[start, end)`: the list a
    /// clip payload carries so the host need not re-detect them.
    pub fn onsets_in_device_range(&self, start: u64, end: u64) -> Vec<NoteOnset> {
        self.onsets
            .iter()
            .filter(|onset| onset.device_sample_index >= start && onset.device_sample_index < end)
            .copied()
            .collect()
    }

    /// Broadband noise-floor RMS (full scale = 1.0) for noise fill, never
    /// below the configured one-LSB minimum.
    pub fn noise_fill_rms(&self) -> f32 {
        (self.noise_floor.broadband_floor_rms() as f32)
            .max(self.configuration.minimum_noise_fill_rms)
    }

    /// Current per-band floor in dBFS (readiness answers, LOCO fill level).
    pub fn band_floor_dbfs(&self) -> Vec<f64> {
        self.noise_floor
            .band_floor_energies()
            .iter()
            .map(|&energy| 10.0 * energy.max(f64::MIN_POSITIVE).log10())
            .collect()
    }

    /// Feed float samples (full scale 1.0) starting at device-clock index
    /// `first_sample_index`. A gap or overlap restarts the analysis window
    /// (the noise floor and history persist) and is counted.
    pub fn process_block(&mut self, first_sample_index: u64, samples: &[f32]) -> Vec<Stage1Event> {
        let mut events = Vec::new();
        match self.stream_origin_sample_index {
            None => {
                self.stream_origin_sample_index = Some(first_sample_index);
                self.next_expected_sample_index = first_sample_index;
            }
            Some(_) if first_sample_index != self.next_expected_sample_index => {
                self.counters.stream_discontinuities += 1;
                self.analysis_buffer.clear();
                self.samples_since_last_frame = 0;
                self.next_expected_sample_index = first_sample_index;
            }
            Some(_) => {}
        }
        for &sample in samples {
            if self.analysis_buffer.len() == self.configuration.analysis_window_samples {
                self.analysis_buffer.pop_front();
            }
            self.analysis_buffer.push_back(sample);
            self.next_expected_sample_index += 1;
            self.samples_since_last_frame += 1;
            if self.samples_since_last_frame >= self.configuration.hop_length_samples
                && self.analysis_buffer.len() == self.configuration.analysis_window_samples
            {
                self.samples_since_last_frame = 0;
                self.analyze_frame(&mut events);
            }
        }
        events
    }

    /// Convenience for PCM16 capture.
    pub fn process_pcm16_block(
        &mut self,
        first_sample_index: u64,
        samples: &[i16],
    ) -> Vec<Stage1Event> {
        const PCM16_FULL_SCALE: f32 = 32_768.0;
        let float_samples: Vec<f32> = samples
            .iter()
            .map(|&sample| f32::from(sample) / PCM16_FULL_SCALE)
            .collect();
        self.process_block(first_sample_index, &float_samples)
    }

    fn analyze_frame(&mut self, events: &mut Vec<Stage1Event>) {
        let frame: Vec<f32> = self.analysis_buffer.iter().copied().collect();
        let mut band_energies = std::mem::take(&mut self.scratch_band_energies);
        self.analyzer.band_energies(&frame, &mut band_energies);
        let hop = self.configuration.hop_length_samples;
        let hop_mean_square = frame[frame.len() - hop..]
            .iter()
            .map(|&sample| f64::from(sample) * f64::from(sample))
            .sum::<f64>()
            / hop as f64;
        let frame_index = self.history.next_frame_index();
        let newest_hop_start_sample_index = self.next_expected_sample_index - hop as u64;
        self.history.push_frame_energies(&band_energies);

        let retained_frames = self
            .configuration
            .trigger_smoothing_frames
            .max(self.configuration.flux_smoothing_frames);
        if self.recent_band_energies.len() == retained_frames {
            self.recent_band_energies.pop_front();
        }
        self.recent_band_energies.push_back(band_energies.clone());
        let smoothed_band_energies =
            self.mean_of_newest_frames(self.configuration.trigger_smoothing_frames);
        let flux_band_energies =
            self.mean_of_newest_frames(self.configuration.flux_smoothing_frames);

        let warmed_up =
            self.noise_floor.frames_observed() >= self.configuration.noise_floor_warmup_frames;
        if warmed_up {
            self.evaluate_trigger_and_onsets(
                frame_index,
                newest_hop_start_sample_index,
                &flux_band_energies,
                &smoothed_band_energies,
                events,
            );
        }
        self.noise_floor
            .update(&smoothed_band_energies, hop_mean_square);
        self.scratch_band_energies = band_energies;
    }

    /// Per-band mean energy over the newest `frame_count` retained frames.
    fn mean_of_newest_frames(&self, frame_count: usize) -> Vec<f64> {
        let used_frames = frame_count.clamp(1, self.recent_band_energies.len());
        let first_used = self.recent_band_energies.len() - used_frames;
        (0..self.configuration.band_count)
            .map(|band_index| {
                self.recent_band_energies
                    .iter()
                    .skip(first_used)
                    .map(|row| row[band_index])
                    .sum::<f64>()
                    / used_frames as f64
            })
            .collect()
    }

    fn evaluate_trigger_and_onsets(
        &mut self,
        frame_index: u64,
        newest_hop_start_sample_index: u64,
        flux_band_energies: &[f64],
        smoothed_band_energies: &[f64],
        events: &mut Vec<Stage1Event>,
    ) {
        let floors = self.noise_floor.band_floor_energies();
        let mut trigger_band_excess: Vec<(usize, f64)> = self
            .trigger_band_indices
            .iter()
            .map(|&band_index| {
                (
                    band_index,
                    energy_ratio_db(smoothed_band_energies[band_index], floors[band_index]),
                )
            })
            .collect();
        trigger_band_excess.sort_by(|left, right| right.1.total_cmp(&left.1));
        let top_count = self
            .configuration
            .contrast_top_band_count
            .min(trigger_band_excess.len())
            .max(1);
        let contrast_db = trigger_band_excess
            .iter()
            .take(top_count)
            .map(|(_, excess)| excess)
            .sum::<f64>()
            / top_count as f64;

        // Positive flux on excess over floor, with the noise allowance removed
        // so stationary background contributes ~0.
        let allowance = self.configuration.flux_excess_noise_allowance_db;
        let current_excess: Vec<f64> = (0..floors.len())
            .map(|band_index| {
                (energy_ratio_db(flux_band_energies[band_index], floors[band_index]) - allowance)
                    .max(0.0)
            })
            .collect();
        let lag_frames = self.configuration.flux_smoothing_frames.max(1);
        let positive_flux_db = match self.lagged_band_excess_db.front() {
            Some(lagged_excess) if self.lagged_band_excess_db.len() == lag_frames => self
                .trigger_band_indices
                .iter()
                .map(|&band_index| {
                    (current_excess[band_index] - lagged_excess[band_index]).max(0.0)
                })
                .sum(),
            _ => 0.0,
        };
        if self.lagged_band_excess_db.len() == lag_frames {
            self.lagged_band_excess_db.pop_front();
        }
        self.lagged_band_excess_db.push_back(current_excess);
        let onset_opened_trigger = self.detect_onset_at_previous_frame(
            frame_index,
            newest_hop_start_sample_index,
            positive_flux_db,
            events,
        );

        self.counters.frames_evaluated += 1;
        self.last_frame_diagnostics = Stage1FrameDiagnostics {
            contrast_db,
            positive_flux_db,
        };
        let open_threshold = self.configuration.contrast_open_threshold_db;
        let close_threshold = open_threshold - self.configuration.contrast_close_hysteresis_db;
        if contrast_db >= open_threshold {
            self.frames_above_open_threshold += 1;
        } else {
            self.frames_above_open_threshold = 0;
        }
        if !self.trigger_is_open
            && (self.frames_above_open_threshold >= self.configuration.contrast_open_minimum_frames
                || onset_opened_trigger)
        {
            self.trigger_is_open = true;
            self.frames_below_close_threshold = 0;
            self.counters.triggers_opened += 1;
            let (active_band_low_hz, active_band_high_hz) =
                self.active_band_range_hz(&trigger_band_excess, open_threshold);
            events.push(Stage1Event::TriggerOpened {
                frame_index,
                device_sample_index: newest_hop_start_sample_index,
                contrast_db,
                active_band_low_hz,
                active_band_high_hz,
            });
        } else if self.trigger_is_open {
            if contrast_db < close_threshold {
                self.frames_below_close_threshold += 1;
            } else {
                self.frames_below_close_threshold = 0;
            }
            if self.frames_below_close_threshold >= self.configuration.trigger_release_frames {
                self.trigger_is_open = false;
                events.push(Stage1Event::TriggerClosed {
                    frame_index,
                    device_sample_index: newest_hop_start_sample_index,
                });
            }
        }
        if self.trigger_is_open {
            self.counters.frames_while_triggered += 1;
        } else {
            self.counters.frames_rejected += 1;
        }
    }

    /// Peak-picks the previous frame's flux (one frame of look-ahead).
    /// Returns true when that onset is sharp enough to open the trigger.
    fn detect_onset_at_previous_frame(
        &mut self,
        frame_index: u64,
        newest_hop_start_sample_index: u64,
        current_flux_db: f64,
        events: &mut Vec<Stage1Event>,
    ) -> bool {
        let candidate_flux = self.flux_one_frame_ago;
        let is_peak = candidate_flux >= self.configuration.onset_flux_threshold_db
            && candidate_flux >= self.flux_two_frames_ago
            && candidate_flux > current_flux_db;
        self.flux_two_frames_ago = self.flux_one_frame_ago;
        self.flux_one_frame_ago = current_flux_db;
        let Some(candidate_frame_index) = frame_index.checked_sub(1) else {
            return false;
        };
        let spaced = self.last_onset_frame_index.is_none_or(|last| {
            candidate_frame_index >= last + self.configuration.onset_minimum_spacing_frames
        });
        if !is_peak || !spaced {
            return false;
        }
        let onset = NoteOnset {
            frame_index: candidate_frame_index,
            device_sample_index: newest_hop_start_sample_index
                .saturating_sub(self.configuration.hop_length_samples as u64),
            positive_flux_db: candidate_flux,
        };
        self.last_onset_frame_index = Some(candidate_frame_index);
        self.counters.onsets_detected += 1;
        if self.onsets.len() == self.configuration.onset_list_capacity {
            self.onsets.pop_front();
            self.counters.onsets_dropped_from_full_list += 1;
        }
        self.onsets.push_back(onset);
        events.push(Stage1Event::Onset(onset));
        candidate_flux >= self.configuration.onset_opens_trigger_flux_db
    }

    /// Lowest and highest edge of trigger bands at or above the threshold
    /// (falls back to the single strongest band).
    fn active_band_range_hz(
        &self,
        sorted_excess: &[(usize, f64)],
        threshold_db: f64,
    ) -> (f64, f64) {
        let layouts = self.analyzer.band_layouts();
        let mut active = sorted_excess
            .iter()
            .filter(|(_, excess)| *excess >= threshold_db)
            .peekable();
        let first = match active.peek() {
            Some(entry) => entry.0,
            None => sorted_excess.first().map_or(0, |entry| entry.0),
        };
        let (low_band, high_band) = active.fold((first, first), |(low, high), (band_index, _)| {
            (low.min(*band_index), high.max(*band_index))
        });
        (
            layouts[low_band].low_edge_hz,
            layouts[high_band].high_edge_hz,
        )
    }
}
