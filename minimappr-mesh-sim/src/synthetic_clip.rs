//! Arrival-time clips and the simulator's `ClipSetLocalizer`.
//!
//! A simulated clip carries, instead of FLAC audio, the arrivals inside the
//! requested window (call id, arrival on the device's own sample clock with
//! SNR-dependent timing noise, SNR, and the call's species/band so the
//! classifier stand-in can score a window), padded to the size a real 24 kHz
//! FLAC clip of that window would have, so traffic accounting stays
//! realistic. The localizer stands in for clip SRP + onset clustering: every
//! call several clips share is a source cluster (echoes rejected by the S2
//! independent-onset rule), each solved by the core `ArrivalTimeLocalizer`
//! (see [`crate::synthetic_clusters`]).

use std::cell::RefCell;

use minimappr_core::mesh::arrival_time_localizer::ArrivalTimeLocalizer;
use minimappr_core::mesh::call_pipeline::{CallFix, CallLocalizationRequest, ClipSetLocalizer, LocalizedCluster};
use minimappr_core::onset_clustering::attribution_decision::{ClassifierAnswer, InterfererOnset};
use minimappr_core::protocol::NodeId;

use crate::synthetic_clusters::{classify_window, decode_request_clips, localize_window_clusters};

const SYNTHETIC_CLIP_MAGIC: &[u8; 4] = b"MSIM";
const ENTRY_BYTES: usize = 4 + 8 + 4 + 4 + 4 + 4 + 4;
const HEADER_BYTES: usize = 4 + 2;
/// The classifier stand-in's detection threshold when the runner sets none.
const DEFAULT_CLASSIFIER_DETECTION_SNR_DB: f64 = 15.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticArrival {
    pub call_index: u32,
    pub arrival_device_sample: f64,
    pub snr_db: f32,
    pub timing_std_s: f32,
    pub species_label: u32,
    pub classifier_score: f32,
    pub centre_frequency_hz: f32,
}

pub fn encode_synthetic_clip(arrivals: &[SyntheticArrival], padded_size_bytes: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(padded_size_bytes.max(HEADER_BYTES + arrivals.len() * ENTRY_BYTES));
    bytes.extend_from_slice(SYNTHETIC_CLIP_MAGIC);
    bytes.extend_from_slice(&(arrivals.len() as u16).to_le_bytes());
    for arrival in arrivals {
        bytes.extend_from_slice(&arrival.call_index.to_le_bytes());
        bytes.extend_from_slice(&arrival.arrival_device_sample.to_le_bytes());
        bytes.extend_from_slice(&arrival.snr_db.to_le_bytes());
        bytes.extend_from_slice(&arrival.timing_std_s.to_le_bytes());
        bytes.extend_from_slice(&arrival.species_label.to_le_bytes());
        bytes.extend_from_slice(&arrival.classifier_score.to_le_bytes());
        bytes.extend_from_slice(&arrival.centre_frequency_hz.to_le_bytes());
    }
    if bytes.len() < padded_size_bytes {
        bytes.resize(padded_size_bytes, 0);
    }
    bytes
}

pub fn decode_synthetic_clip(bytes: &[u8]) -> Option<Vec<SyntheticArrival>> {
    if bytes.len() < HEADER_BYTES || &bytes[..4] != SYNTHETIC_CLIP_MAGIC {
        return None;
    }
    let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let mut arrivals = Vec::with_capacity(count);
    let f32_at = |entry: &[u8], offset: usize| entry[offset..offset + 4].try_into().ok().map(f32::from_le_bytes);
    for index in 0..count {
        let start = HEADER_BYTES + index * ENTRY_BYTES;
        let entry = bytes.get(start..start + ENTRY_BYTES)?;
        arrivals.push(SyntheticArrival {
            call_index: u32::from_le_bytes(entry[0..4].try_into().ok()?),
            arrival_device_sample: f64::from_le_bytes(entry[4..12].try_into().ok()?),
            snr_db: f32_at(entry, 12)?,
            timing_std_s: f32_at(entry, 16)?,
            species_label: u32::from_le_bytes(entry[20..24].try_into().ok()?),
            classifier_score: f32_at(entry, 24)?,
            centre_frequency_hz: f32_at(entry, 28)?,
        });
    }
    Some(arrivals)
}

/// What the localizer saw for one source cluster (scored against truth later).
#[derive(Debug, Clone, PartialEq)]
pub struct SolveTrace {
    pub host_call_id: u64,
    pub true_call_index: u32,
    pub node_ids: Vec<NodeId>,
    pub estimated_node_positions_m: Vec<[f64; 3]>,
    pub fix: Option<CallFix>,
    pub chirality_confidence: Option<f64>,
    pub chirality_locked: bool,
}

#[derive(Debug)]
pub struct ArrivalClipLocalizer {
    pub localizer: ArrivalTimeLocalizer,
    pub traces: RefCell<Vec<SolveTrace>>,
    /// The classifier stand-in reports calls above this in-band SNR.
    pub classifier_detection_snr_db: f64,
}

impl Default for ArrivalClipLocalizer {
    fn default() -> Self {
        Self {
            localizer: ArrivalTimeLocalizer::default(),
            traces: RefCell::new(Vec::new()),
            classifier_detection_snr_db: DEFAULT_CLASSIFIER_DETECTION_SNR_DB,
        }
    }
}

impl ClipSetLocalizer for ArrivalClipLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        self.localize_clusters(request).and_then(|clusters| {
            clusters.into_iter().next().map(|cluster| cluster.fix).ok_or_else(|| "no source cluster".to_string())
        })
    }

    fn localize_clusters(&self, request: &CallLocalizationRequest) -> Result<Vec<LocalizedCluster>, String> {
        let decoded = decode_request_clips(request);
        let (clusters, traces) = localize_window_clusters(&self.localizer, request, &decoded);
        self.traces.borrow_mut().extend(traces);
        clusters
    }

    fn classify_cluster_window(
        &self,
        request: &CallLocalizationRequest,
        cluster: &LocalizedCluster,
        masked: &[InterfererOnset],
    ) -> Option<(ClassifierAnswer, f64)> {
        classify_window(request, &decode_request_clips(request), cluster.classifying_node, masked, self.classifier_detection_snr_db)
    }
}
