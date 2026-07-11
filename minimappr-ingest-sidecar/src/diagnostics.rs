//! Lightweight, lock-free counters for the `/api/v1/diagnostics/summary`
//! endpoint. Mirrors the equivalent Python `FusionMetrics` latency fields
//! (`minimappr/core/fusion_node.py`) so an operator can compare the Rust and
//! Python ingest backends side by side using the same field names.
//!
//! All counters are plain `AtomicU64` updated with `Ordering::Relaxed` — no
//! locks, no allocation, safe to call unconditionally on every frame from the
//! hot path. This is deliberately separate from `DspWorkerState` (behind a
//! `RwLock`) because that state is only written in small, infrequent batches
//! (`count > 0` guards); latency here is recorded on *every* frame, so it
//! needs its own non-blocking path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

#[derive(Debug, Default)]
pub struct IngestDiagnostics {
    overload_rejections: AtomicU64,
    queue_wait_sum_ms: AtomicU64,
    queue_wait_count: AtomicU64,
    queue_wait_max_ms: AtomicU64,
    processing_sum_ms: AtomicU64,
    processing_count: AtomicU64,
    processing_max_ms: AtomicU64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct IngestDiagnosticsSnapshot {
    pub overload_rejections: u64,
    /// Frames the DSP worker dequeued for processing — the Rust analog of
    /// Python's `FusionMetrics.frames_accepted`.
    pub frames_received: u64,
    /// Frames for which the DSP compute closure ran to completion.
    pub frames_processed: u64,
    pub queue_wait_avg_ms: f64,
    pub queue_wait_max_ms: u64,
    pub processing_avg_ms: f64,
    pub processing_max_ms: u64,
}

fn record_max(counter: &AtomicU64, candidate_ms: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    while candidate_ms > current {
        match counter.compare_exchange_weak(
            current,
            candidate_ms,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn avg_ms(sum_ms: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum_ms as f64 / count as f64
    }
}

impl IngestDiagnostics {
    pub fn record_overload_rejection(&self) {
        self.overload_rejections.fetch_add(1, Ordering::Relaxed);
    }

    /// Time from HTTP arrival (`received_ns` on the journal handle) to the DSP
    /// worker dequeuing the manifest for processing.
    pub fn record_queue_wait_ms(&self, wait_ms: u64) {
        self.queue_wait_sum_ms.fetch_add(wait_ms, Ordering::Relaxed);
        self.queue_wait_count.fetch_add(1, Ordering::Relaxed);
        record_max(&self.queue_wait_max_ms, wait_ms);
    }

    /// Time spent inside the DSP compute closure (localization + classifier
    /// render), i.e. pure compute time excluding queue wait.
    pub fn record_processing_ms(&self, processing_ms: u64) {
        self.processing_sum_ms
            .fetch_add(processing_ms, Ordering::Relaxed);
        self.processing_count.fetch_add(1, Ordering::Relaxed);
        record_max(&self.processing_max_ms, processing_ms);
    }

    pub fn snapshot(&self) -> IngestDiagnosticsSnapshot {
        let queue_wait_sum = self.queue_wait_sum_ms.load(Ordering::Relaxed);
        let queue_wait_count = self.queue_wait_count.load(Ordering::Relaxed);
        let processing_sum = self.processing_sum_ms.load(Ordering::Relaxed);
        let processing_count = self.processing_count.load(Ordering::Relaxed);
        IngestDiagnosticsSnapshot {
            overload_rejections: self.overload_rejections.load(Ordering::Relaxed),
            frames_received: queue_wait_count,
            frames_processed: processing_count,
            queue_wait_avg_ms: avg_ms(queue_wait_sum, queue_wait_count),
            queue_wait_max_ms: self.queue_wait_max_ms.load(Ordering::Relaxed),
            processing_avg_ms: avg_ms(processing_sum, processing_count),
            processing_max_ms: self.processing_max_ms.load(Ordering::Relaxed),
        }
    }
}

/// Records elapsed wall-clock time into `record_processing_ms` when dropped,
/// regardless of which branch/early-return the caller takes. Used to time the
/// full DSP compute closure (`actors::dsp_compute::run_compute`) without
/// having to instrument every exit path individually.
pub struct ScopedProcessingTimer<'a> {
    diagnostics: &'a IngestDiagnostics,
    start: Instant,
}

impl<'a> ScopedProcessingTimer<'a> {
    pub fn start(diagnostics: &'a IngestDiagnostics) -> Self {
        Self {
            diagnostics,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedProcessingTimer<'_> {
    fn drop(&mut self) {
        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        self.diagnostics.record_processing_ms(elapsed_ms);
    }
}
