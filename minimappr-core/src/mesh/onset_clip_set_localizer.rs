//! The real host `ClipSetLocalizer`: decoded audio, onset clustering, LM.
//!
//! 1. decode every clip (`clip_codec`) and resample it onto the call's
//!    shared window on the island reference clock (`onset_clip_alignment`);
//! 2. run onset-frame SRP clustering on the window (`localize_onset_clusters`),
//!    which finds every source, rejects echoes and attaches S5 level evidence;
//! 3. polish each accepted cluster with `JointLmSeedRefiner` from its own
//!    reference-relative TDOAs, with every node's layout σ and clock σ;
//! 4. classify windows through a host-supplied callback (attribution).
//!
//! Unlike the Layer 2 arrival-time stand-in, nothing here knows which call
//! made which onset: two overlapping birds can be merged, split wrongly or
//! mislocated, exactly as on a phone.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::onset_clustering::attribution_decision::{AttributionThresholds, ClassifierAnswer, InterfererOnset};
use crate::onset_clustering::attribution_evidence::ClusterAttributionEvidence;
use crate::onset_clustering::{localize_onset_clusters, ClusterStatus, LocalizedOnsetCluster, OnsetClusteringConfig, OnsetClusteringInput};
use crate::protocol::NodeId;
use crate::srp_layout::lm_handoff::{LmSeed, PairTdoaObservation, TdoaSeedRefiner};

use super::call_pipeline_types::{CallFix, CallLocalizationRequest, ClipSetLocalizer, LocalizedCluster};
use super::config::DEVICE_SAMPLE_RATE_HZ;
use super::onset_clip_alignment::{align_call_clips, AlignedCallWindow, AlignedNodeWindow};
use super::srp_seed_refiner::{JointLmSeedRefiner, RefinedTdoaFix, SeedRefinerConfig};

/// Field clips are decimated 48 → 24 kHz (`clip_codec`).
pub const FIELD_CLIP_SAMPLE_RATE_HZ: u32 = 24_000;
/// `range_projection_mode` of a cluster with a full 3-D position (onset refinement).
const RANGE_REFINED_MODE: &str = "range_refined";
/// Huber knee recorded on seeds built here (unused by the refiner's loss).
const SEED_HUBER_KNEE: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct OnsetClipLocalizerConfig {
    pub onset: OnsetClusteringConfig,
    pub sound_speed_mps: f64,
    /// GCC-PHAT lag interpolation inside onset clustering.
    pub interpolation_factor: usize,
    /// Range a bearing-only cluster is projected to (server default 50 m).
    pub far_field_default_range_m: f64,
    /// Attribution's audibility threshold (S5 evidence); None skips evidence.
    pub attribution_audible_snr_db: Option<f64>,
    /// A clip covering less of the shared window than this is not used.
    pub minimum_clip_coverage_fraction: f64,
    /// Floor on a pair's onset-TDOA σ before clock σ is added: a little under
    /// one 24 kHz sample (41.7 µs); sub-sample GCC-PHAT interpolation on a
    /// clean onset does better, a smeared onset worse (χ² inflation covers it).
    pub pair_timing_std_floor_s: f64,
    /// Scale the covariance by χ²/dof when the fit is worse than declared.
    pub inflate_covariance_by_reduced_chi_square: bool,
    pub refiner: SeedRefinerConfig,
    /// Phones are held this far above the ground they stand on.
    pub assumed_node_height_above_ground_m: f64,
    /// Canopy band for the refiner's height prior (above ground).
    pub source_height_band_above_ground_m: [f64; 2],
    /// Fewer contributing nodes than this cannot give a 3-D fix.
    pub minimum_contributing_nodes: usize,
    /// A cluster whose loudest node's band SNR stays under this is a noise
    /// onset, not a source: the onset detector's own active-band threshold
    /// (`OnsetClusteringConfig::active_band_snr_db`). Measured: pure-noise
    /// windows give clusters at 2.5–5 dB that still fit a point at χ²/dof
    /// ≈ 2, so a goodness-of-fit test cannot reject them. Well under the
    /// 10 dB Stage-1 trigger, so faint long-range calls are not gated out.
    pub minimum_loudest_node_band_snr_db: f64,
}

impl Default for OnsetClipLocalizerConfig {
    fn default() -> Self {
        Self {
            onset: OnsetClusteringConfig::default(),
            sound_speed_mps: 343.0,
            interpolation_factor: 4,
            far_field_default_range_m: 50.0,
            attribution_audible_snr_db: Some(AttributionThresholds::default().audible_snr_db),
            minimum_clip_coverage_fraction: 0.5,
            pair_timing_std_floor_s: 25e-6,
            inflate_covariance_by_reduced_chi_square: true,
            refiner: SeedRefinerConfig::default(),
            assumed_node_height_above_ground_m: 1.4,
            source_height_band_above_ground_m: [0.0, 20.0],
            minimum_contributing_nodes: 3,
            minimum_loudest_node_band_snr_db: OnsetClusteringConfig::default().active_band_snr_db,
        }
    }
}

/// What a classifier callback sees: the cluster, the onsets to mask, and the
/// classifying node's window audio on the reference timeline (None if that
/// node's clip was unusable).
pub struct ClusterClassificationInput<'call> {
    pub request: &'call CallLocalizationRequest,
    pub cluster: &'call LocalizedCluster,
    pub masked: &'call [InterfererOnset],
    pub window_start_reference_s: f64,
    pub sample_rate_hz: u32,
    pub classifying_node_audio: Option<&'call AlignedNodeWindow>,
}

/// (answer, masked share of the window). None = cannot classify here.
/// `Send` so a localizer can live inside a shell object shared across threads
/// (the phone FFI device sits behind a mutex used by capture and tick threads).
pub type ClusterWindowClassifier = dyn Fn(&ClusterClassificationInput<'_>) -> Option<(ClassifierAnswer, f64)> + Send;

/// One localized cluster plus where its led onsets were heard, on the
/// reference clock: provenance for logs and simulator scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetLocalizedCluster {
    pub cluster: LocalizedCluster,
    pub led_onsets_reference_s: Vec<(NodeId, f64)>,
}

pub struct OnsetClipSetLocalizer {
    pub config: OnsetClipLocalizerConfig,
    classifier: Option<Box<ClusterWindowClassifier>>,
    /// The last call's aligned window, so attribution reruns do not decode again.
    aligned_cache: RefCell<Option<(AlignedWindowKey, AlignedCallWindow)>>,
}

/// What an aligned window depends on. Keyed on the request's window and clip
/// set, not the call id alone: a localizer can outlive a host tenure (the
/// mesh sim shares one across hosts), and call-id uniqueness across terms is
/// the host's invariant, not this cache's.
#[derive(Debug, Clone, PartialEq)]
struct AlignedWindowKey {
    call_id: u64,
    island_id: u32,
    centre_reference_s_bits: u64,
    half_window_s_bits: u64,
    clip_nodes: Vec<NodeId>,
}

impl AlignedWindowKey {
    fn of(request: &CallLocalizationRequest) -> Self {
        Self {
            call_id: request.call_id,
            island_id: request.island_id,
            centre_reference_s_bits: request.centre_reference_s.to_bits(),
            half_window_s_bits: request.half_window_s.to_bits(),
            clip_nodes: request.clips.iter().map(|clip| clip.node_id).collect(),
        }
    }
}

impl std::fmt::Debug for OnsetClipSetLocalizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnsetClipSetLocalizer")
            .field("config", &self.config)
            .field("has_classifier", &self.classifier.is_some())
            .finish()
    }
}

fn parse_node_id(text: &str) -> Option<NodeId> {
    text.parse::<u32>().ok().map(NodeId)
}

impl OnsetClipSetLocalizer {
    pub fn new(config: OnsetClipLocalizerConfig) -> Self {
        Self { config, classifier: None, aligned_cache: RefCell::new(None) }
    }

    pub fn with_classifier(mut self, classifier: Box<ClusterWindowClassifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    fn aligned_window(&self, request: &CallLocalizationRequest) -> AlignedCallWindow {
        let key = AlignedWindowKey::of(request);
        if let Some((cached_key, window)) = self.aligned_cache.borrow().as_ref() {
            if *cached_key == key {
                return window.clone();
            }
        }
        let window = align_call_clips(request, FIELD_CLIP_SAMPLE_RATE_HZ, self.config.minimum_clip_coverage_fraction);
        *self.aligned_cache.borrow_mut() = Some((key, window.clone()));
        window
    }

    fn clustering_input(&self, request: &CallLocalizationRequest, window: &AlignedCallWindow) -> OnsetClusteringInput {
        let mut input = OnsetClusteringInput {
            sensor_signals: BTreeMap::new(),
            sensor_positions_m: BTreeMap::new(),
            sensor_node_ids: BTreeMap::new(),
            sample_rate_hz: window.sample_rate_hz,
            sound_speed_mps: self.config.sound_speed_mps,
            interpolation_factor: self.config.interpolation_factor,
            far_field_default_range_m: self.config.far_field_default_range_m,
            consumed_onset_samples_by_node: BTreeMap::new(),
            attribution_audible_snr_db: self.config.attribution_audible_snr_db,
        };
        for node in &window.nodes {
            // One mono phone mic per node: sensor id = node id.
            let id = node.node_id.0.to_string();
            input.sensor_signals.insert(id.clone(), node.samples.iter().map(|sample| f64::from(*sample)).collect());
            input.sensor_positions_m.insert(id.clone(), request.clips[node.clip_index].position.position_m);
            input.sensor_node_ids.insert(id.clone(), id);
        }
        input
    }

    /// Every accepted cluster that yields a fix, strongest first. The window
    /// fails only when none does: unlike the stand-in, the strongest onset
    /// cluster is not necessarily the detected call (it can be a bearing-only
    /// or two-node cluster while a full-layout one follows).
    pub fn localize_window(&self, request: &CallLocalizationRequest) -> Result<Vec<OnsetLocalizedCluster>, String> {
        let window = self.aligned_window(request);
        if window.nodes.len() < self.config.minimum_contributing_nodes {
            return Err(format!("{} usable clips (rejected: {:?})", window.nodes.len(), window.rejected_clips));
        }
        let result = localize_onset_clusters(&self.clustering_input(request, &window), &self.config.onset);
        let accepted: Vec<&LocalizedOnsetCluster> = result
            .clusters
            .iter()
            .filter(|cluster| cluster.status == ClusterStatus::Accepted && self.audible_somewhere(cluster))
            .collect();
        if accepted.is_empty() {
            return Err(format!("no accepted onset cluster ({} onsets, {} frames)", result.onset_count, result.frame_count));
        }
        let mut failures = Vec::new();
        let mut clusters = Vec::new();
        for cluster in accepted {
            match self.refine_cluster(request, &window, cluster) {
                Ok(localized) => clusters.push(localized),
                Err(reason) => failures.push(reason),
            }
        }
        if clusters.is_empty() {
            return Err(failures.join("; "));
        }
        Ok(clusters)
    }

    /// Without evidence (attribution off) the cluster cannot be judged and is kept.
    fn audible_somewhere(&self, cluster: &LocalizedOnsetCluster) -> bool {
        cluster.attribution_evidence.as_ref().is_none_or(|evidence| {
            evidence.node_band_snr_db.values().any(|snr| *snr >= self.config.minimum_loudest_node_band_snr_db)
        })
    }

    fn refine_cluster(
        &self,
        request: &CallLocalizationRequest,
        window: &AlignedCallWindow,
        cluster: &LocalizedOnsetCluster,
    ) -> Result<OnsetLocalizedCluster, String> {
        let nodes: Vec<&AlignedNodeWindow> = window.nodes.iter().collect();
        let node_index = |id: &str| nodes.iter().position(|node| node.node_id.0.to_string() == id);
        let reference_index = node_index(&cluster.reference_sensor).ok_or("cluster reference is not a clip node")?;
        let measured: Vec<(usize, f64)> =
            cluster.tdoa_s.iter().filter_map(|(sensor, tdoa)| Some((node_index(sensor)?, *tdoa))).filter(|(index, _)| *index != reference_index).collect();
        if measured.len() + 1 < self.config.minimum_contributing_nodes {
            let fix = self.seed_only_fix(request, &nodes, cluster).ok_or_else(|| format!("cluster has {} nodes with a TDOA", measured.len() + 1))?;
            return Ok(self.localized(request, window, cluster, fix));
        }
        let clip_of = |index: usize| &request.clips[nodes[index].clip_index];
        let observations: Vec<PairTdoaObservation> = measured
            .iter()
            .map(|&(index, tdoa_s)| {
                let clock_variance = clip_of(index).clock.offset_sigma_s.powi(2) + clip_of(reference_index).clock.offset_sigma_s.powi(2);
                PairTdoaObservation {
                    sensor_a_position_m: clip_of(index).position.position_m,
                    sensor_b_position_m: clip_of(reference_index).position.position_m,
                    sensor_a_node: index,
                    sensor_b_node: reference_index,
                    // tdoa_s[s] = arrival(s) − arrival(reference) = (|x − s| − |x − ref|) / c.
                    measured_tdoa_s: tdoa_s,
                    relative_peak_coherence: 1.0,
                    weight: 1.0,
                    lag_resolution_s: (self.config.pair_timing_std_floor_s.powi(2) + clock_variance).sqrt(),
                }
            })
            .collect();
        let mean_node_height = nodes.iter().map(|node| request.clips[node.clip_index].position.position_m[2]).sum::<f64>() / nodes.len() as f64;
        let ground_m = mean_node_height - self.config.assumed_node_height_above_ground_m;
        let [band_low, band_high] = self.config.source_height_band_above_ground_m;
        let seed = LmSeed {
            initial_position_m: cluster.position_m,
            far_field_direction: None,
            observations,
            sound_speed_mps: self.config.sound_speed_mps,
            source_height_bounds_m: [ground_m + band_low, ground_m + band_high],
            huber_knee: SEED_HUBER_KNEE,
            srp_score: 0.0,
        };
        let refiner = JointLmSeedRefiner {
            config: self.config.refiner.clone(),
            node_position_std_m: nodes.iter().map(|node| request.clips[node.clip_index].position.std_m).collect(),
        };
        let refined = refiner.refine(&seed).map_err(|error| format!("{error:?}"))?;
        let fix = self.call_fix(&refined, &seed, &nodes, reference_index);
        Ok(self.localized(request, window, cluster, fix))
    }

    /// Onset refinement keeps the SRP seed (and no TDOAs) when its LM point
    /// scores worse than the seed. Such a cluster still has a full-layout
    /// position; publish it with the pipeline's own covariance (resolution
    /// floor + spread of its frame peaks) plus the layout σ, which moves a
    /// fix about one-for-one. A bearing-only seed is not a fix.
    fn seed_only_fix(&self, request: &CallLocalizationRequest, nodes: &[&AlignedNodeWindow], cluster: &LocalizedOnsetCluster) -> Option<CallFix> {
        let full_layout = cluster.range_projection_mode.as_deref() == Some(RANGE_REFINED_MODE);
        let covariance = cluster.covariance_m2.filter(|_| full_layout && nodes.len() >= self.config.minimum_contributing_nodes)?;
        let layout_variance =
            nodes.iter().map(|node| request.clips[node.clip_index].position.std_m.powi(2)).sum::<f64>() / nodes.len() as f64;
        let covariance_m2: [[f64; 3]; 3] =
            std::array::from_fn(|row| std::array::from_fn(|column| covariance[row][column] + if row == column { layout_variance } else { 0.0 }));
        let mut used_nodes: Vec<NodeId> = nodes.iter().map(|node| node.node_id).collect();
        used_nodes.sort();
        let as_refined = RefinedTdoaFix {
            position_m: cluster.position_m,
            covariance_m2,
            chi_square: 0.0,
            degrees_of_freedom: 0,
            rejected_observation_indices: Vec::new(),
        };
        Some(CallFix {
            position_m: cluster.position_m,
            covariance_m2,
            horizontal_major_sigma_m: as_refined.horizontal_major_sigma_m(),
            used_nodes,
            rejected_nodes: Vec::new(),
        })
    }

    fn localized(&self, request: &CallLocalizationRequest, window: &AlignedCallWindow, cluster: &LocalizedOnsetCluster, fix: CallFix) -> OnsetLocalizedCluster {
        let classifying_node = parse_node_id(&cluster.loudest_node_id).unwrap_or(request.classifying_node);
        let led_onsets_reference_s = cluster
            .template_node_ids
            .iter()
            .zip(&cluster.onset_samples)
            .filter_map(|(node, sample)| Some((parse_node_id(node)?, window.reference_seconds_of_sample(*sample as f64))))
            .collect();
        OnsetLocalizedCluster {
            cluster: LocalizedCluster {
                attribution_evidence: cluster.attribution_evidence.as_ref().map(|evidence| reference_clock_evidence(evidence, window)),
                fix,
                classifying_node,
                attributed_species: None,
                attribution_tier: None,
            },
            led_onsets_reference_s,
        }
    }

    fn call_fix(&self, refined: &RefinedTdoaFix, seed: &LmSeed, nodes: &[&AlignedNodeWindow], reference_index: usize) -> CallFix {
        let mut fix = refined.clone();
        if self.config.inflate_covariance_by_reduced_chi_square && refined.degrees_of_freedom > 0 {
            let reduced = refined.chi_square / refined.degrees_of_freedom as f64;
            if reduced > 1.0 {
                fix.covariance_m2 = refined.covariance_m2.map(|row| row.map(|value| value * reduced));
            }
        }
        let rejected_nodes: Vec<NodeId> =
            refined.rejected_observation_indices.iter().map(|&index| nodes[seed.observations[index].sensor_a_node].node_id).collect();
        let mut used_nodes: Vec<NodeId> = std::iter::once(reference_index)
            .chain(seed.observations.iter().map(|observation| observation.sensor_a_node))
            .map(|index| nodes[index].node_id)
            .filter(|node| !rejected_nodes.contains(node))
            .collect();
        used_nodes.sort();
        CallFix {
            position_m: fix.position_m,
            covariance_m2: fix.covariance_m2,
            horizontal_major_sigma_m: fix.horizontal_major_sigma_m(),
            used_nodes,
            rejected_nodes,
        }
    }
}

/// Onset clustering reports arrivals on the 24 kHz window timeline; the host
/// attribution step reads them as 48 kHz reference-clock samples.
fn reference_clock_evidence(evidence: &ClusterAttributionEvidence, window: &AlignedCallWindow) -> ClusterAttributionEvidence {
    let mut converted = evidence.clone();
    for arrivals in converted.node_onset_arrivals.values_mut() {
        for arrival in arrivals.iter_mut() {
            let reference_s = window.reference_seconds_of_sample(arrival.span_sample as f64);
            arrival.span_sample = (reference_s * DEVICE_SAMPLE_RATE_HZ).round() as i64;
        }
    }
    converted
}

impl ClipSetLocalizer for OnsetClipSetLocalizer {
    fn localize(&self, request: &CallLocalizationRequest) -> Result<CallFix, String> {
        self.localize_window(request)?.into_iter().next().map(|cluster| cluster.cluster.fix).ok_or_else(|| "no source cluster".to_string())
    }

    fn localize_clusters(&self, request: &CallLocalizationRequest) -> Result<Vec<LocalizedCluster>, String> {
        Ok(self.localize_window(request)?.into_iter().map(|cluster| cluster.cluster).collect())
    }

    fn classify_cluster_window(
        &self,
        request: &CallLocalizationRequest,
        cluster: &LocalizedCluster,
        masked: &[InterfererOnset],
    ) -> Option<(ClassifierAnswer, f64)> {
        let classifier = self.classifier.as_ref()?;
        let window = self.aligned_window(request);
        classifier(&ClusterClassificationInput {
            request,
            cluster,
            masked,
            window_start_reference_s: window.window_start_reference_s,
            sample_rate_hz: window.sample_rate_hz,
            classifying_node_audio: window.node(cluster.classifying_node),
        })
    }
}

#[cfg(test)]
#[path = "onset_clip_set_localizer_tests.rs"]
mod tests;
