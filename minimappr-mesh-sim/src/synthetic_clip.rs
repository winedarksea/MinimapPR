//! Arrival-time clips and the simulator's `ClipSetLocalizer`.
//!
//! A simulated clip carries, instead of FLAC audio, the arrivals inside the
//! requested window (call id, arrival on the device's own sample clock with
//! SNR-dependent timing noise, SNR), padded to the size a real 24 kHz FLAC
//! clip of that window would have, so traffic accounting stays realistic.
//! The localizer stands in for clip SRP: it picks the call most clips share,
//! maps each arrival through the host's clock model, and runs the core
//! `ArrivalTimeLocalizer` (grid seed + `TdoaSeedRefiner` LM).

use std::cell::RefCell;

use minimappr_core::mesh::arrival_time_localizer::{ArrivalTimeLocalizer, NodeArrival};
use minimappr_core::mesh::call_pipeline::{CallFix, CallLocalizationRequest, ClipSetLocalizer};
use minimappr_core::protocol::NodeId;

const SYNTHETIC_CLIP_MAGIC: &[u8; 4] = b"MSIM";
const ENTRY_BYTES: usize = 4 + 8 + 4 + 4;
const HEADER_BYTES: usize = 4 + 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticArrival {
    pub call_index: u32,
    pub arrival_device_sample: f64,
    pub snr_db: f32,
    pub timing_std_s: f32,
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
    for index in 0..count {
        let start = HEADER_BYTES + index * ENTRY_BYTES;
        let entry = bytes.get(start..start + ENTRY_BYTES)?;
        arrivals.push(SyntheticArrival {
            call_index: u32::from_le_bytes(entry[0..4].try_into().ok()?),
            arrival_device_sample: f64::from_le_bytes(entry[4..12].try_into().ok()?),
            snr_db: f32::from_le_bytes(entry[12..16].try_into().ok()?),
            timing_std_s: f32::from_le_bytes(entry[16..20].try_into().ok()?),
        });
    }
    Some(arrivals)
}

/// What the localizer saw for one call (scored against truth later).
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

#[derive(Debug, Default)]
pub struct ArrivalClipLocalizer {
    pub localizer: ArrivalTimeLocalizer,
    pub traces: RefCell<Vec<SolveTrace>>,
}

impl ClipSetLocalizer for ArrivalClipLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        let decoded: Vec<(usize, Vec<SyntheticArrival>)> = request
            .clips
            .iter()
            .enumerate()
            .filter_map(|(index, clip)| Some((index, decode_synthetic_clip(&clip.payload.clip_container)?)))
            .collect();
        let mut votes: std::collections::BTreeMap<u32, (usize, f32)> = std::collections::BTreeMap::new();
        for (_, arrivals) in &decoded {
            for arrival in arrivals {
                let vote = votes.entry(arrival.call_index).or_insert((0, 0.0));
                vote.0 += 1;
                vote.1 += arrival.snr_db;
            }
        }
        let call_index = votes
            .iter()
            .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(a.1 .1.total_cmp(&b.1 .1)))
            .map(|(call, _)| *call)
            .ok_or_else(|| "no correlatable arrival in any clip".to_string())?;
        let mut node_ids = Vec::new();
        let mut arrivals = Vec::new();
        for (index, clip_arrivals) in &decoded {
            let clip = &request.clips[*index];
            let Some(arrival) = clip_arrivals.iter().find(|a| a.call_index == call_index) else { continue };
            node_ids.push(clip.node_id);
            arrivals.push(NodeArrival {
                position_m: clip.position.position_m,
                position_std_m: clip.position.std_m,
                reference_arrival_s: clip.clock.reference_seconds_from_node_sample(arrival.arrival_device_sample),
                timing_std_s: (arrival.timing_std_s as f64).hypot(clip.clock.offset_sigma_s),
            });
        }
        let mut trace = SolveTrace {
            host_call_id: request.call_id,
            true_call_index: call_index,
            node_ids: node_ids.clone(),
            estimated_node_positions_m: arrivals.iter().map(|a| a.position_m).collect(),
            fix: None,
            chirality_confidence: request.chirality_confidence,
            chirality_locked: request.chirality_locked,
        };
        let result = self.localizer.localize(&arrivals).map_err(|error| format!("{error:?}"));
        let outcome = result.map(|solved| {
            let used_nodes: Vec<NodeId> = solved.used_arrival_indices.iter().map(|&index| node_ids[index]).collect();
            CallFix {
                horizontal_major_sigma_m: solved.fix.horizontal_major_sigma_m(),
                position_m: solved.fix.position_m,
                covariance_m2: solved.fix.covariance_m2,
                rejected_nodes: node_ids.iter().copied().filter(|n| !used_nodes.contains(n)).collect(),
                used_nodes,
            }
        });
        trace.fix = outcome.as_ref().ok().cloned();
        self.traces.borrow_mut().push(trace);
        outcome
    }
}
