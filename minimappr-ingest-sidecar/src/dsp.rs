use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioCoverageStats {
    pub sample_count: usize,
    pub covered_samples: usize,
    pub missing_samples: usize,
    pub coverage_ratio: f64,
    pub missing_ratio: f64,
    pub max_gap_samples: usize,
    pub max_gap_seconds: f64,
    pub warning: bool,
    pub degraded: bool,
}

#[derive(Clone, Debug)]
pub struct SensorStreamBuffer {
    sample_rate_hz: u32,
    max_samples: i64,
    timeline_origin_ns: Option<i128>,
    timeline_origin_sample_index: i64,
    buffer_start_sample_index: i64,
    samples: VecDeque<f32>,
    coverage: VecDeque<bool>,
}

impl SensorStreamBuffer {
    pub fn new(sample_rate_hz: u32, max_duration_seconds: f64) -> Self {
        let max_samples = (max_duration_seconds * f64::from(sample_rate_hz))
            .round()
            .max(1.0) as i64;
        Self {
            sample_rate_hz,
            max_samples,
            timeline_origin_ns: None,
            timeline_origin_sample_index: 0,
            buffer_start_sample_index: 0,
            samples: VecDeque::new(),
            coverage: VecDeque::new(),
        }
    }

    fn sample_len(&self) -> usize {
        self.samples.len()
    }

    fn collect_samples_range(&self, start_offset: usize, end_offset: usize) -> Vec<f32> {
        self.samples
            .iter()
            .skip(start_offset)
            .take(end_offset.saturating_sub(start_offset))
            .copied()
            .collect()
    }

    fn collect_coverage_range(&self, start_offset: usize, end_offset: usize) -> Vec<bool> {
        self.coverage
            .iter()
            .skip(start_offset)
            .take(end_offset.saturating_sub(start_offset))
            .copied()
            .collect()
    }

    pub fn append(
        &mut self,
        start_time_ns: i128,
        samples: &[f32],
        start_sample_index: Option<i64>,
        end_sample_index: Option<i64>,
    ) -> Result<(), String> {
        let explicit_sample_coverage = start_sample_index.is_some();
        let incoming_start = start_sample_index.unwrap_or_else(|| {
            if self.timeline_origin_ns.is_none() {
                0
            } else {
                self.time_to_sample_index(start_time_ns).unwrap_or(0)
            }
        });
        let incoming_end = end_sample_index.unwrap_or(incoming_start + samples.len() as i64);
        if incoming_end < incoming_start {
            return Err("end_sample_index must be >= start_sample_index".to_string());
        }
        if (incoming_end - incoming_start) as usize != samples.len() {
            return Err("explicit sample coverage must match appended sample count".to_string());
        }

        if self.timeline_origin_ns.is_none() {
            self.reset(start_time_ns, incoming_start, samples);
            return Ok(());
        }
        if samples.is_empty() {
            return Ok(());
        }

        let current_start = self.buffer_start_sample_index;
        let current_end = current_start + self.sample_len() as i64;
        let mut incoming_start = incoming_start;

        if explicit_sample_coverage {
            let expected_start_time_ns = self.sample_index_to_time_ns(incoming_start)?;
            let timestamp_correction_ns = (start_time_ns - expected_start_time_ns).abs();
            let reset_threshold_ns =
                self.max_samples as i128 * 1_000_000_000_i128 / i128::from(self.sample_rate_hz);
            if timestamp_correction_ns > reset_threshold_ns {
                self.reset(start_time_ns, incoming_start, samples);
                return Ok(());
            }
        }

        if incoming_start > current_end + self.max_samples
            || incoming_start < current_start - self.max_samples
        {
            self.reset(start_time_ns, incoming_start, samples);
            return Ok(());
        }

        if !explicit_sample_coverage {
            let drift = incoming_start - current_end;
            let snap_tolerance = (samples.len() as i64 / 2).max(1);
            if -snap_tolerance < drift && drift < snap_tolerance {
                incoming_start = current_end;
            }
        }

        let incoming_end = incoming_start + samples.len() as i64;

        // Fast path: append after current tail, optionally zero-filling a gap.
        if incoming_start >= current_end {
            let gap = (incoming_start - current_end) as usize;
            if gap > 0 {
                self.samples.resize(self.sample_len() + gap, 0.0_f32);
                self.coverage.resize(self.coverage.len() + gap, false);
            }
            self.samples.extend(samples.iter().copied());
            self.coverage
                .resize(self.coverage.len() + samples.len(), true);
            self.prune();
            return Ok(());
        }

        // Fast path: incoming frame is fully inside current buffer bounds.
        if incoming_start >= current_start && incoming_end <= current_end {
            let offset = (incoming_start - current_start) as usize;
            for (index, sample) in samples.iter().copied().enumerate() {
                self.samples[offset + index] = sample;
                self.coverage[offset + index] = true;
            }
            return Ok(());
        }

        // Fast path: overlap tail then extend past current end.
        if incoming_start >= current_start
            && incoming_start < current_end
            && incoming_end > current_end
        {
            let offset = (incoming_start - current_start) as usize;
            let overlap_len = (current_end - incoming_start) as usize;
            for (index, sample) in samples.iter().take(overlap_len).copied().enumerate() {
                self.samples[offset + index] = sample;
                self.coverage[offset + index] = true;
            }
            self.samples.extend(samples[overlap_len..].iter().copied());
            self.coverage.resize(
                self.coverage.len() + samples.len().saturating_sub(overlap_len),
                true,
            );
            self.prune();
            return Ok(());
        }

        // Fallback path for prepend and complex overlap patterns.
        let merged_start = current_start.min(incoming_start);
        let merged_end = current_end.max(incoming_end);
        let mut merged = vec![0.0_f32; (merged_end - merged_start) as usize];
        let mut merged_coverage = vec![false; (merged_end - merged_start) as usize];

        let existing_offset = (current_start - merged_start) as usize;
        for (index, sample) in self.samples.iter().copied().enumerate() {
            merged[existing_offset + index] = sample;
        }
        for (index, covered) in self.coverage.iter().copied().enumerate() {
            merged_coverage[existing_offset + index] = covered;
        }

        let incoming_offset = (incoming_start - merged_start) as usize;
        merged[incoming_offset..incoming_offset + samples.len()].copy_from_slice(samples);
        merged_coverage[incoming_offset..incoming_offset + samples.len()].fill(true);

        self.samples = VecDeque::from(merged);
        self.coverage = VecDeque::from(merged_coverage);
        self.buffer_start_sample_index = merged_start;
        self.prune();
        Ok(())
    }

    pub fn window_ending_at(&self, end_time_ns: i128, window_seconds: f64) -> Option<Vec<f32>> {
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as i64;
        let end_sample = self.time_to_sample_index(end_time_ns).ok()?;
        let start_sample = end_sample - window_samples;
        let end_offset = end_sample - self.buffer_start_sample_index;
        // Clamp the start to the beginning of the buffer so partial windows (e.g.
        // when the buffer has less than classification_window_seconds of history)
        // return whatever audio IS available rather than None.  Returning None here
        // caused every sensor to fall back to the 80 ms localization window, which
        // is far too short for BirdNET and produced only "unknown" (0.0) classifications.
        // We still return None when the end is beyond what has been buffered because
        // that would require future samples.
        if end_offset <= 0 || end_offset > self.sample_len() as i64 {
            return None;
        }
        let start_offset = (start_sample - self.buffer_start_sample_index).max(0) as usize;
        if start_offset >= end_offset as usize {
            return None;
        }
        Some(self.collect_samples_range(start_offset, end_offset as usize))
    }

    /// Extract a window centered on `center_time_ns`, matching Python's `get_window`.
    /// Used for localization where TDOA is computed relative to the window center.
    pub fn window_centered_at(
        &self,
        center_time_ns: i128,
        window_seconds: f64,
    ) -> Option<Vec<f32>> {
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as i64;
        let center_sample = self.time_to_sample_index(center_time_ns).ok()?;
        let start_sample = center_sample - (window_samples / 2);
        let end_sample = start_sample + window_samples;
        let start_offset = start_sample - self.buffer_start_sample_index;
        let end_offset = end_sample - self.buffer_start_sample_index;
        if start_offset < 0 || end_offset > self.sample_len() as i64 {
            return None;
        }
        Some(self.collect_samples_range(start_offset as usize, end_offset as usize))
    }

    /// Coverage stats for a window centered on `center_time_ns`.
    pub fn coverage_centered_at(
        &self,
        center_time_ns: i128,
        window_seconds: f64,
    ) -> Option<AudioCoverageStats> {
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as i64;
        let center_sample = self.time_to_sample_index(center_time_ns).ok()?;
        let start_sample = center_sample - (window_samples / 2);
        let end_sample = start_sample + window_samples;
        let start_offset = start_sample - self.buffer_start_sample_index;
        let end_offset = end_sample - self.buffer_start_sample_index;
        if start_offset < 0 || end_offset > self.coverage.len() as i64 {
            return None;
        }
        let window_coverage =
            self.collect_coverage_range(start_offset as usize, end_offset as usize);
        Some(coverage_stats(&window_coverage, self.sample_rate_hz))
    }

    pub fn latest_window(&self, window_seconds: f64) -> Vec<f32> {
        if self.samples.is_empty() {
            return Vec::new();
        }
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as usize;
        let start_offset = self.sample_len().saturating_sub(window_samples);
        self.collect_samples_range(start_offset, self.sample_len())
    }

    pub fn latest_coverage_stats(&self, window_seconds: f64) -> Option<AudioCoverageStats> {
        if self.coverage.is_empty() {
            return None;
        }
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as usize;
        let start_offset = self.coverage.len().saturating_sub(window_samples);
        let window_coverage = self.collect_coverage_range(start_offset, self.coverage.len());
        Some(coverage_stats(&window_coverage, self.sample_rate_hz))
    }

    pub fn coverage_ending_at(
        &self,
        end_time_ns: i128,
        window_seconds: f64,
    ) -> Option<AudioCoverageStats> {
        let window_samples = (window_seconds * f64::from(self.sample_rate_hz))
            .round()
            .max(1.0) as i64;
        let end_sample = self.time_to_sample_index(end_time_ns).ok()?;
        let start_sample = end_sample - window_samples;
        let end_offset = end_sample - self.buffer_start_sample_index;
        if end_offset <= 0 || end_offset > self.coverage.len() as i64 {
            return None;
        }
        let start_offset = (start_sample - self.buffer_start_sample_index).max(0) as usize;
        if start_offset >= end_offset as usize {
            return None;
        }
        let window_coverage = self.collect_coverage_range(start_offset, end_offset as usize);
        Some(coverage_stats(&window_coverage, self.sample_rate_hz))
    }

    fn reset(&mut self, start_time_ns: i128, start_sample_index: i64, samples: &[f32]) {
        self.timeline_origin_ns = Some(start_time_ns);
        self.timeline_origin_sample_index = start_sample_index;
        self.buffer_start_sample_index = start_sample_index;
        self.samples = VecDeque::from(samples.to_vec());
        self.coverage = VecDeque::from(vec![true; samples.len()]);
        self.prune();
    }

    fn prune(&mut self) {
        if self.sample_len() as i64 <= self.max_samples {
            return;
        }
        // Drop exactly the minimum number of samples needed to fit within max_samples.
        // VecDeque front pops are O(1), so there is no memmove cost — aggressive
        // chunked dropping is not needed and would create coverage gaps in the
        // 30-second BirdNET classification window.
        let drop = (self.sample_len() as i64 - self.max_samples).max(0) as usize;
        for _ in 0..drop {
            let _ = self.samples.pop_front();
            let _ = self.coverage.pop_front();
        }
        self.buffer_start_sample_index += drop as i64;
    }

    fn time_to_sample_index(&self, time_ns: i128) -> Result<i64, String> {
        let origin = self
            .timeline_origin_ns
            .ok_or_else(|| "timeline origin is not initialized".to_string())?;
        let delta_ns = time_ns - origin;
        Ok(self.timeline_origin_sample_index
            + round_divide(delta_ns * i128::from(self.sample_rate_hz), 1_000_000_000) as i64)
    }

    fn sample_index_to_time_ns(&self, sample_index: i64) -> Result<i128, String> {
        let origin = self
            .timeline_origin_ns
            .ok_or_else(|| "timeline origin is not initialized".to_string())?;
        Ok(origin
            + round_divide(
                i128::from(sample_index - self.timeline_origin_sample_index) * 1_000_000_000,
                i128::from(self.sample_rate_hz),
            ))
    }

    pub fn time_for_sample_index(&self, sample_index: i64) -> Option<i128> {
        self.sample_index_to_time_ns(sample_index).ok()
    }
}

pub fn coverage_stats(coverage: &[bool], sample_rate_hz: u32) -> AudioCoverageStats {
    let sample_count = coverage.len();
    if sample_count == 0 {
        return AudioCoverageStats {
            sample_count: 0,
            covered_samples: 0,
            missing_samples: 0,
            coverage_ratio: 0.0,
            missing_ratio: 1.0,
            max_gap_samples: 0,
            max_gap_seconds: 0.0,
            warning: true,
            degraded: true,
        };
    }
    let mut missing_samples = 0;
    let mut max_gap_samples = 0;
    let mut current_gap = 0;
    for covered in coverage {
        if *covered {
            current_gap = 0;
        } else {
            missing_samples += 1;
            current_gap += 1;
            max_gap_samples = max_gap_samples.max(current_gap);
        }
    }
    let covered_samples = sample_count - missing_samples;
    let max_gap_seconds = max_gap_samples as f64 / f64::from(sample_rate_hz);
    let missing_ratio = missing_samples as f64 / sample_count as f64;
    AudioCoverageStats {
        sample_count,
        covered_samples,
        missing_samples,
        coverage_ratio: covered_samples as f64 / sample_count as f64,
        missing_ratio,
        max_gap_samples,
        max_gap_seconds,
        warning: missing_ratio > 0.01 || max_gap_seconds > 0.100,
        degraded: missing_ratio > 0.05 || max_gap_seconds > 0.250,
    }
}

fn round_divide(numerator: i128, denominator: i128) -> i128 {
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        -((-numerator + denominator / 2) / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_merge_overwrites_existing_samples() {
        let sr = 16_000;
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);
        buffer
            .append(t0, &[1.0, 1.0, 1.0, 1.0], Some(0), Some(4))
            .unwrap();
        buffer
            .append(
                t0 + (2 * 1_000_000_000_i128 / i128::from(sr)),
                &[9.0, 9.0, 9.0, 9.0],
                Some(2),
                Some(6),
            )
            .unwrap();

        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 9.0, 9.0, 9.0, 9.0]
        );
        assert_eq!(
            buffer.coverage.iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, true, true, true]
        );
    }

    #[test]
    fn late_frame_replaces_zero_gap() {
        let sr = 16_000;
        let chunk_ns = 4 * 1_000_000_000_i128 / i128::from(sr);
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);
        buffer.append(t0, &[1.0; 4], None, None).unwrap();
        buffer
            .append(t0 + (2 * chunk_ns), &[3.0; 4], None, None)
            .unwrap();
        buffer.append(t0 + chunk_ns, &[2.0; 4], None, None).unwrap();
        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        assert_eq!(
            buffer
                .coverage_ending_at(
                    t0 + (12 * 1_000_000_000_i128 / i128::from(sr)),
                    12.0 / f64::from(sr)
                )
                .unwrap()
                .coverage_ratio,
            1.0
        );
    }

    #[test]
    fn explicit_sample_index_gaps_mark_missing_coverage() {
        let sr = 16_000;
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);
        buffer
            .append(t0, &[1.0, 1.0, 1.0, 1.0], Some(0), Some(4))
            .unwrap();
        buffer
            .append(
                t0 + (8 * 1_000_000_000_i128 / i128::from(sr)),
                &[2.0, 2.0, 2.0, 2.0],
                Some(8),
                Some(12),
            )
            .unwrap();

        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        );
        let stats = buffer
            .coverage_ending_at(
                t0 + (12 * 1_000_000_000_i128 / i128::from(sr)),
                12.0 / f64::from(sr),
            )
            .unwrap();
        assert_eq!(stats.covered_samples, 8);
        assert_eq!(stats.missing_samples, 4);
        assert_eq!(stats.max_gap_samples, 4);
        assert!(stats.warning);
        assert!(stats.degraded);
    }

    #[test]
    fn explicit_contiguous_samples_reanchor_after_large_timestamp_correction() {
        let sr = 16_000;
        let frame_samples = 1_024;
        let stale_start_ns = 1_000_000_000_000_000_000_i128;
        let corrected_start_ns = stale_start_ns + 600_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 30.0);
        buffer
            .append(
                stale_start_ns,
                &vec![1.0; frame_samples],
                Some(0),
                Some(frame_samples as i64),
            )
            .unwrap();
        buffer
            .append(
                corrected_start_ns,
                &vec![2.0; frame_samples],
                Some(frame_samples as i64),
                Some((frame_samples * 2) as i64),
            )
            .unwrap();

        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![2.0; frame_samples]
        );
        assert_eq!(buffer.timeline_origin_ns, Some(corrected_start_ns));
        assert_eq!(buffer.timeline_origin_sample_index, frame_samples as i64);
        assert_eq!(buffer.buffer_start_sample_index, frame_samples as i64);
        assert_eq!(
            buffer.coverage.iter().copied().collect::<Vec<_>>(),
            vec![true; frame_samples]
        );
    }

    #[test]
    fn trailing_window_returns_partial_audio() {
        let sr = 16_000;
        let mut buffer = SensorStreamBuffer::new(sr, 40.0);
        let samples = vec![1.0_f32; 3 * sr as usize];
        let t0 = 1_000_000_000_000_000_000_i128;
        buffer.append(t0, &samples, None, None).unwrap();
        let window = buffer.window_ending_at(t0 + 3_000_000_000, 30.0).unwrap();
        assert_eq!(window.len(), samples.len());
    }

    // --- Phase 1a: cross-language buffer parity ---

    /// Two non-overlapping frames with an explicit gap produce the correct merged
    /// sample values and leave the gap zero-filled with false coverage bits.
    #[test]
    fn multi_frame_merge_with_gap_zero_fills_and_preserves_coverage() {
        let sr = 16_000;
        let t0 = 1_000_000_000_000_000_000_i128;
        let ns_per_sample = 1_000_000_000_i128 / i128::from(sr);
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);

        // Frame A: samples 0–3 with value 1.0
        buffer.append(t0, &[1.0; 4], Some(0), Some(4)).unwrap();
        // Frame B: samples 8–11 with value 2.0 (gap of 4 samples)
        buffer
            .append(t0 + 8 * ns_per_sample, &[2.0; 4], Some(8), Some(12))
            .unwrap();

        // Expected: [1,1,1,1, 0,0,0,0, 2,2,2,2]
        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        );
        // Gap in the middle must have false coverage.
        let expected_coverage = vec![
            true, true, true, true, false, false, false, false, true, true, true, true,
        ];
        assert_eq!(
            buffer.coverage.iter().copied().collect::<Vec<_>>(),
            expected_coverage
        );
    }

    /// A single continuous frame produces full coverage with no gaps.
    #[test]
    fn zero_fill_semantics_continuous_frame_gives_full_coverage() {
        let sr = 16_000;
        let t0 = 2_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 2.0);

        buffer
            .append(t0, &[0.5_f32; 512], Some(0), Some(512))
            .unwrap();

        let stats = buffer
            .coverage_ending_at(
                t0 + 512 * 1_000_000_000_i128 / i128::from(sr),
                512.0 / f64::from(sr),
            )
            .unwrap();

        assert_eq!(stats.missing_samples, 0);
        assert_eq!(stats.coverage_ratio, 1.0);
        assert!(!stats.warning);
        assert!(!stats.degraded);
    }

    /// Coverage stats at the warning/degraded thresholds match the documented
    /// boundaries (>1% missing → warning, >5% missing → degraded).
    #[test]
    fn coverage_stats_warning_and_degraded_thresholds() {
        let sr = 16_000_u32;
        // 1.5% missing → warning only (not degraded)
        let mut cov = vec![true; 1000];
        cov[100..115].fill(false); // 15 / 1000 = 1.5%
        let stats = coverage_stats(&cov, sr);
        assert!(stats.warning, "1.5% missing should be warning");
        assert!(!stats.degraded, "1.5% missing should not be degraded");

        // 6% missing → both warning and degraded
        let mut cov2 = vec![true; 1000];
        cov2[0..60].fill(false); // 60 / 1000 = 6%
        let stats2 = coverage_stats(&cov2, sr);
        assert!(stats2.warning);
        assert!(stats2.degraded, "6% missing should be degraded");

        // max_gap_seconds > 0.100 also triggers warning (120ms = 1920 samples)
        let gap_samples = (0.120 * f64::from(sr)) as usize; // 1920
        let mut cov3 = vec![true; gap_samples + 500]; // ensure slice is large enough
        cov3[0..gap_samples].fill(false);
        let stats3 = coverage_stats(&cov3, sr);
        assert!(stats3.warning, "120ms gap should trigger warning");
        assert_eq!(stats3.max_gap_samples, gap_samples);
    }

    /// Appending a frame whose timestamp disagrees with its explicit sample index
    /// by more than max_duration causes a buffer reset, discarding stale data.
    #[test]
    fn sample_rate_reset_discards_stale_data_on_clock_correction() {
        let sr = 16_000_u32;
        let t0 = 1_000_000_000_000_000_000_i128;
        // Buffer capacity = 1 second worth.
        let mut buffer = SensorStreamBuffer::new(sr, 1.0);

        // First frame: sample index 0, timestamp t0.
        buffer
            .append(t0, &[1.0_f32; 128], Some(0), Some(128))
            .unwrap();
        assert_eq!(buffer.samples.len(), 128);

        // Second frame: same sample indices but timestamp shifted by >1 s.
        // This exceeds the reset threshold, so the buffer should reset.
        let corrected_t0 = t0 + 2_000_000_000_i128; // +2 s
        buffer
            .append(corrected_t0, &[2.0_f32; 128], Some(0), Some(128))
            .unwrap();

        // After reset, only the new frame should be in the buffer.
        assert_eq!(
            buffer.samples.iter().copied().collect::<Vec<_>>(),
            vec![2.0_f32; 128]
        );
        assert_eq!(buffer.timeline_origin_ns, Some(corrected_t0));
    }

    /// The implicit-index snap tolerance correctly joins back-to-back frames
    /// without creating false gaps.
    #[test]
    fn snap_tolerance_joins_consecutive_frames_without_gap() {
        let sr = 16_000_u32;
        let ns_per_chunk = 512 * 1_000_000_000_i128 / i128::from(sr);
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);

        // Send 4 consecutive 512-sample frames without explicit sample indices.
        for i in 0..4_i128 {
            buffer
                .append(
                    t0 + i * ns_per_chunk,
                    &[f32::from(i as i16 + 1); 512],
                    None,
                    None,
                )
                .unwrap();
        }

        // Should have exactly 4 × 512 samples with 100% coverage.
        assert_eq!(buffer.samples.len(), 4 * 512);
        let end_ns = t0 + 4 * ns_per_chunk;
        let stats = buffer
            .coverage_ending_at(end_ns, 4.0 * 512.0 / f64::from(sr))
            .unwrap();
        assert_eq!(stats.missing_samples, 0);
        assert_eq!(stats.coverage_ratio, 1.0);
    }

    #[test]
    fn coverage_window_before_buffer_start_returns_none() {
        let sr = 16_000_u32;
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);
        buffer
            .append(t0, &[1.0_f32; 2_048], Some(0), Some(2_048))
            .unwrap();

        let early_end_time_ns = t0 - 1_000_000_000_i128;
        assert!(
            buffer.coverage_ending_at(early_end_time_ns, 0.5).is_none(),
            "coverage window ending before buffer start should return None"
        );
    }

    #[test]
    fn audio_window_before_buffer_start_returns_none() {
        let sr = 16_000_u32;
        let t0 = 1_000_000_000_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 5.0);
        buffer
            .append(t0, &[1.0_f32; 2_048], Some(0), Some(2_048))
            .unwrap();

        let early_end_time_ns = t0 - 1_000_000_000_i128;
        assert!(
            buffer.window_ending_at(early_end_time_ns, 0.5).is_none(),
            "audio window ending before buffer start should return None"
        );
    }

    #[test]
    fn prune_drops_exact_overflow_to_preserve_coverage_continuity() {
        let sr = 16_u32;
        let t0 = 1_000_000_000_i128;
        let mut buffer = SensorStreamBuffer::new(sr, 1.0);

        buffer
            .append(t0, &[1.0_f32; 16], Some(0), Some(16))
            .unwrap();
        buffer
            .append(
                t0 + (16 * 1_000_000_000_i128 / i128::from(sr)),
                &[2.0_f32; 1],
                Some(16),
                Some(17),
            )
            .unwrap();

        // max_samples=16, overflow=1: exactly 1 sample should be dropped.
        // Aggressive chunked dropping would create false coverage gaps in the
        // BirdNET classification window, so prune must be exact.
        assert_eq!(buffer.buffer_start_sample_index, 1);
        assert_eq!(buffer.samples.len(), 16);
        assert_eq!(buffer.samples[0], 1.0);
        assert_eq!(buffer.samples.back().copied(), Some(2.0));
    }
}
