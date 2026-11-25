//! The host-side localizer of a simulated run: the arrival-time stand-in,
//! or (rendered-audio mode) the real `OnsetClipSetLocalizer` wrapped so the
//! simulator can score it.
//!
//! The real localizer never sees truth. Scoring needs to know which scripted
//! call each cluster came from, so after the fact each cluster is traced to
//! a call: candidates are calls still sounding at a template node when one
//! of the cluster's led onsets fired (a sub-band onset of a sweep fires
//! when the sweep reaches that band, anywhere in the call), with
//! `onset_match_tolerance_s` slack either side; of those, the one nearest the fix (after moving the host frame onto the
//! phones' true positions) wins. An echo cluster sitting on its image
//! source therefore scores as an echo, and a merged cluster scores as the
//! bird it landed on (the other bird's call simply has no cluster).

use std::cell::RefCell;
use std::sync::Arc;

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::mesh::call_pipeline::{CallFix, CallLocalizationRequest, ClipSetLocalizer, LocalizedCluster};
use minimappr_core::mesh::onset_clip_set_localizer::{ClusterClassificationInput, OnsetClipLocalizerConfig, OnsetClipSetLocalizer, OnsetLocalizedCluster};
use minimappr_core::onset_clustering::attribution_decision::{ClassifierAnswer, InterfererOnset};

use crate::acoustic_world::{AcousticWorld, DEVICE_SAMPLE_RATE_HZ};
use crate::audio_world::AudioModeConfig;
use crate::synthetic_clip::{ArrivalClipLocalizer, SolveTrace, SyntheticArrival};
use crate::synthetic_clusters::classify_arrivals;

/// Calls a node's window held, as the classifier stand-in's arrivals.
fn truth_arrivals(world: &AcousticWorld, device: usize, start_s: f64, end_s: f64) -> Vec<SyntheticArrival> {
    let truth = &world.devices[device];
    world
        .calls_arriving_between(device, start_s, end_s)
        .into_iter()
        .filter(|(_, _, snr)| *snr >= world.config.clip_correlation_minimum_snr_db)
        .map(|(call, arrival_s, snr)| SyntheticArrival {
            call_index: call as u32,
            arrival_device_sample: truth.sample_index_at(arrival_s),
            snr_db: snr as f32,
            timing_std_s: 0.0,
            species_label: world.calls[call].species_label,
            classifier_score: world.calls[call].classifier_score,
            centre_frequency_hz: world.calls[call].centre_frequency_hz as f32,
        })
        .collect()
}

/// True session time at which `device` recorded reference time `reference_s`.
fn true_time_at(world: &AcousticWorld, request: &CallLocalizationRequest, device: usize, reference_s: f64) -> Option<f64> {
    let clip = request.clips.iter().find(|clip| clip.node_id.0 as usize == device)?;
    let node_sample = clip.clock.node_seconds_from_reference_seconds(reference_s) * DEVICE_SAMPLE_RATE_HZ;
    Some(world.devices[device].true_time_of_sample(node_sample))
}

/// The SNR-graded classifier stand-in, fed from world truth for the
/// classifying node's window (the audio itself is what the localizer tests).
fn truth_classifier(world: Arc<AcousticWorld>, detection_snr_db: f64, mask_tolerance_s: f64) -> Box<dyn Fn(&ClusterClassificationInput<'_>) -> Option<(ClassifierAnswer, f64)> + Send> {
    Box::new(move |input: &ClusterClassificationInput<'_>| {
        input.classifying_node_audio?;
        let device = input.cluster.classifying_node.0 as usize;
        let clip = input.request.clips.iter().find(|clip| clip.node_id == input.cluster.classifying_node)?;
        let window_end_s = input.window_start_reference_s + 2.0 * input.request.half_window_s;
        let start_s = true_time_at(&world, input.request, device, input.window_start_reference_s)?;
        let end_s = true_time_at(&world, input.request, device, window_end_s)?;
        let arrivals = truth_arrivals(&world, device, start_s, end_s);
        classify_arrivals(&arrivals, &clip.clock, input.window_start_reference_s, input.masked, detection_snr_db, mask_tolerance_s)
    })
}

pub struct AudioClipLocalizer {
    pub localizer: OnsetClipSetLocalizer,
    world: Arc<AcousticWorld>,
    onset_match_tolerance_s: f64,
    call_duration_s: f64,
    pub traces: RefCell<Vec<SolveTrace>>,
}

impl AudioClipLocalizer {
    pub fn new(world: Arc<AcousticWorld>, config: OnsetClipLocalizerConfig, audio: &AudioModeConfig) -> Self {
        let classifier = truth_classifier(world.clone(), world.config.classifier_detection_snr_db, config.onset.min_separation_seconds);
        Self {
            localizer: OnsetClipSetLocalizer::new(config).with_classifier(classifier),
            world,
            onset_match_tolerance_s: audio.onset_match_tolerance_s,
            call_duration_s: audio.call_duration_s,
            traces: RefCell::new(Vec::new()),
        }
    }

    fn scripted_call_of(&self, request: &CallLocalizationRequest, cluster: &OnsetLocalizedCluster, node_positions: &[[f64; 3]]) -> u32 {
        let world = &self.world;
        let mut best_by_time: Option<(usize, f64)> = None;
        let mut candidates = Vec::new();
        for (node, onset_reference_s) in &cluster.led_onsets_reference_s {
            let device = node.0 as usize;
            let Some(onset_true_s) = true_time_at(world, request, device, *onset_reference_s) else { continue };
            for (index, call) in world.calls.iter().enumerate() {
                let arrival_s = world.arrival_time(call.time_s, call.position_m, device);
                // Distance in time from the onset to the call's sounding interval at this node.
                let gap = (arrival_s - onset_true_s).max(onset_true_s - arrival_s - self.call_duration_s).max(0.0);
                if best_by_time.is_none_or(|(_, best)| gap < best) {
                    best_by_time = Some((index, gap));
                }
                if gap <= self.onset_match_tolerance_s && !candidates.contains(&index) {
                    candidates.push(index);
                }
            }
        }
        let fix = &cluster.cluster.fix;
        let call_time_s = request.centre_reference_s;
        let estimated: Vec<[f64; 2]> = node_positions.iter().map(|p| [p[0], p[1]]).collect();
        let truth: Vec<[f64; 2]> = fix.used_nodes.iter().map(|node| world.position(node.0 as usize, call_time_s)).map(|p| [p[0], p[1]]).collect();
        let transform = weighted_kabsch_2d(&estimated, &truth, &vec![1.0; estimated.len()], false);
        let aligned = transform.apply([fix.position_m[0], fix.position_m[1]]);
        let horizontal = |index: &usize| {
            let p = world.calls[*index].position_m;
            (aligned[0] - p[0]).hypot(aligned[1] - p[1])
        };
        candidates
            .iter()
            .min_by(|a, b| horizontal(a).total_cmp(&horizontal(b)))
            .copied()
            .or(best_by_time.map(|(index, _)| index))
            .unwrap_or(0) as u32
    }
}

impl ClipSetLocalizer for AudioClipLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        self.localize_clusters(request)?.into_iter().next().map(|cluster| cluster.fix).ok_or_else(|| "no source cluster".to_string())
    }

    fn localize_clusters(&self, request: &CallLocalizationRequest) -> Result<Vec<LocalizedCluster>, String> {
        let clusters = self.localizer.localize_window(request)?;
        for cluster in &clusters {
            let fix = &cluster.cluster.fix;
            let used_clips: Vec<_> = fix
                .used_nodes
                .iter()
                .filter_map(|node| request.clips.iter().find(|clip| clip.node_id == *node))
                .collect();
            let node_positions: Vec<[f64; 3]> = used_clips.iter().map(|clip| clip.position.position_m).collect();
            let true_call_index = self.scripted_call_of(request, cluster, &node_positions);
            self.traces.borrow_mut().push(SolveTrace {
                host_call_id: request.call_id,
                true_call_index,
                node_ids: fix.used_nodes.clone(),
                estimated_node_positions_m: node_positions,
                estimated_node_std_m: used_clips.iter().map(|clip| clip.position.std_m).collect(),
                localizer_inputs: Vec::new(),
                fix: Some(fix.clone()),
                chirality_confidence: request.chirality_confidence,
                chirality_locked: request.chirality_locked,
            });
        }
        Ok(clusters.into_iter().map(|cluster| cluster.cluster).collect())
    }

    fn classify_cluster_window(&self, request: &CallLocalizationRequest, cluster: &LocalizedCluster, masked: &[InterfererOnset]) -> Option<(ClassifierAnswer, f64)> {
        self.localizer.classify_cluster_window(request, cluster, masked)
    }
}

/// The localizer a run hands its hosts.
pub enum SimulatedClipLocalizer {
    Arrival(ArrivalClipLocalizer),
    Audio(Box<AudioClipLocalizer>),
}

impl SimulatedClipLocalizer {
    pub fn take_traces(&self) -> Vec<SolveTrace> {
        match self {
            Self::Arrival(localizer) => localizer.traces.take(),
            Self::Audio(localizer) => localizer.traces.take(),
        }
    }

    fn inner(&self) -> &dyn ClipSetLocalizer {
        match self {
            Self::Arrival(localizer) => localizer,
            Self::Audio(localizer) => localizer.as_ref(),
        }
    }
}

impl ClipSetLocalizer for SimulatedClipLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        self.inner().localize(request)
    }

    fn localize_clusters(&self, request: &CallLocalizationRequest) -> Result<Vec<LocalizedCluster>, String> {
        self.inner().localize_clusters(request)
    }

    fn classify_cluster_window(&self, request: &CallLocalizationRequest, cluster: &LocalizedCluster, masked: &[InterfererOnset]) -> Option<(ClassifierAnswer, f64)> {
        self.inner().classify_cluster_window(request, cluster, masked)
    }

    fn masked_cell_fraction(&self, request: &CallLocalizationRequest, cluster: &LocalizedCluster, masked: &[InterfererOnset]) -> f64 {
        self.inner().masked_cell_fraction(request, cluster, masked)
    }
}
