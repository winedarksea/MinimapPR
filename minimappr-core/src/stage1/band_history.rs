//! Rolling u8-quantized sub-band history (one row of `band_count` bytes per
//! 20 ms frame; ~2 KB/s at 40 bands). Feeds readiness answers, the active
//! band reported with a trigger, and later LOCO masks.

/// Ring of quantized band levels addressed by absolute frame index.
#[derive(Debug, Clone)]
pub struct QuantizedBandHistory {
    band_count: usize,
    capacity_frames: usize,
    quantization_step_db: f64,
    quantization_floor_dbfs: f64,
    levels: Vec<u8>,
    /// Absolute index of the next frame to be written.
    next_frame_index: u64,
}

impl QuantizedBandHistory {
    pub fn new(
        band_count: usize,
        capacity_frames: usize,
        quantization_step_db: f64,
        quantization_floor_dbfs: f64,
    ) -> Self {
        Self {
            band_count,
            capacity_frames,
            quantization_step_db,
            quantization_floor_dbfs,
            levels: vec![0; band_count * capacity_frames],
            next_frame_index: 0,
        }
    }

    pub fn band_count(&self) -> usize {
        self.band_count
    }

    pub fn next_frame_index(&self) -> u64 {
        self.next_frame_index
    }

    /// Oldest frame index still held.
    pub fn oldest_frame_index(&self) -> u64 {
        self.next_frame_index
            .saturating_sub(self.capacity_frames as u64)
    }

    pub fn quantize_dbfs(&self, level_dbfs: f64) -> u8 {
        let steps =
            ((level_dbfs - self.quantization_floor_dbfs) / self.quantization_step_db).round();
        steps.clamp(0.0, f64::from(u8::MAX)) as u8
    }

    pub fn dequantize_dbfs(&self, quantized_level: u8) -> f64 {
        self.quantization_floor_dbfs + f64::from(quantized_level) * self.quantization_step_db
    }

    pub fn push_frame_energies(&mut self, band_energies: &[f64]) {
        debug_assert_eq!(band_energies.len(), self.band_count);
        if self.capacity_frames == 0 {
            self.next_frame_index += 1;
            return;
        }
        let row_start = self.row_slot(self.next_frame_index) * self.band_count;
        for (band_index, &energy) in band_energies.iter().enumerate() {
            let level_dbfs = 10.0 * energy.max(f64::MIN_POSITIVE).log10();
            self.levels[row_start + band_index] = self.quantize_dbfs(level_dbfs);
        }
        self.next_frame_index += 1;
    }

    /// Ring slot of an absolute frame index. Reduce in u64 first: on wasm32
    /// `usize` is 32 bits and a truncating cast would jump slots at 2^32.
    fn row_slot(&self, frame_index: u64) -> usize {
        (frame_index % self.capacity_frames as u64) as usize
    }

    /// Quantized row for an absolute frame index, if still held.
    pub fn frame_levels(&self, frame_index: u64) -> Option<&[u8]> {
        if frame_index >= self.next_frame_index || frame_index < self.oldest_frame_index() {
            return None;
        }
        let row_start = self.row_slot(frame_index) * self.band_count;
        Some(&self.levels[row_start..row_start + self.band_count])
    }
}
