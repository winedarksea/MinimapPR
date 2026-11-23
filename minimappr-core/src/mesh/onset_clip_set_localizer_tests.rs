//! Real-audio tests for `OnsetClipSetLocalizer`: analytically delayed FM
//! calls sampled on each phone's own drifting clock, decimated and FLAC
//! encoded exactly as a phone would, then localized from the clip bytes.

use super::*;
use crate::clip_codec::field_clip::encode_field_clip;
use crate::clip_codec::{DecimationFilterId, FieldClipDecimator};
use crate::mesh::call_pipeline_types::ClipLocalizationInput;
use crate::mesh::clock_mapping::NodeClockMapping;
use crate::mesh::island_sync_engine::NodePositionForSolve;
use crate::protocol::detection_messages::ClipPayload;

const SOUND_SPEED_MPS: f64 = 343.0;
const NOISE_RMS: f64 = 0.0005;
const SOURCE_AMPLITUDE_AT_1M: f64 = 0.5;
const CALL_DURATION_S: f64 = 0.35;
const EMISSION_S: f64 = 100.0;
const HALF_WINDOW_S: f64 = 0.75;
const BUFFER_MARGIN_S: f64 = 0.1;
const PHONES_M: [[f64; 3]; 5] = [[0.0, 0.0, 1.4], [16.0, 2.0, 1.5], [7.0, 15.0, 1.3], [-9.0, 11.0, 1.4], [-6.0, -10.0, 1.5]];

struct TestCall {
    emission_s: f64,
    position_m: [f64; 3],
    sweep_hz: (f64, f64),
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn call_sample(call: &TestCall, time_since_emission_s: f64) -> f64 {
    if !(0.0..CALL_DURATION_S).contains(&time_since_emission_s) {
        return 0.0;
    }
    let t = time_since_emission_s;
    let rate = (call.sweep_hz.1 - call.sweep_hz.0) / CALL_DURATION_S;
    let fade = 0.1 * CALL_DURATION_S;
    let envelope = if t < fade {
        0.5 - 0.5 * (std::f64::consts::PI * t / fade).cos()
    } else if t > CALL_DURATION_S - fade {
        0.5 - 0.5 * (std::f64::consts::PI * (CALL_DURATION_S - t) / fade).cos()
    } else {
        1.0
    };
    (std::f64::consts::TAU * (call.sweep_hz.0 * t + 0.5 * rate * t * t)).sin() * envelope
}

/// Deterministic noise (sum of uniforms from an LCG).
struct Noise(u64);

impl Noise {
    fn gaussian(&mut self) -> f64 {
        (0..12)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            })
            .sum::<f64>()
            - 6.0
    }
}

fn phone_clock(index: usize) -> NodeClockMapping {
    NodeClockMapping {
        clock_epoch: 1,
        anchor_node_seconds: 90.0 + index as f64,
        offset_s: 0.004 * index as f64 - 0.008,
        rate_offset: (index as f64 - 2.0) * 20e-6,
        offset_sigma_s: 5e-6,
    }
}

/// One phone's clip of the reference window, rendered on its own clock.
fn render_clip(index: usize, calls: &[TestCall], window_start_s: f64, silent: bool) -> ClipLocalizationInput {
    let clock = phone_clock(index);
    let position = PHONES_M[index];
    let first_device = clock.node_sample_index_from_reference_seconds(window_start_s - BUFFER_MARGIN_S);
    let window_first = clock.node_sample_index_from_reference_seconds(window_start_s);
    let window_count = (2.0 * HALF_WINDOW_S * DEVICE_SAMPLE_RATE_HZ) as usize;
    let buffer_count = window_count + (2.0 * BUFFER_MARGIN_S * DEVICE_SAMPLE_RATE_HZ) as usize;
    let mut noise = Noise(17 + index as u64);
    let buffer: Vec<i16> = (0..buffer_count)
        .map(|offset| {
            let true_time = clock.reference_seconds_from_node_sample((first_device + offset as u64) as f64);
            let signal: f64 = if silent {
                0.0
            } else {
                calls
                    .iter()
                    .map(|call| {
                        let range = distance(call.position_m, position);
                        SOURCE_AMPLITUDE_AT_1M / range.max(1.0) * call_sample(call, true_time - call.emission_s - range / SOUND_SPEED_MPS)
                    })
                    .sum()
            };
            ((signal + NOISE_RMS * noise.gaussian()) * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16
        })
        .collect();
    let decimated = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut)
        .decimate_window_from_rolling_buffer(&buffer, first_device, window_first, window_count)
        .expect("window inside buffer");
    ClipLocalizationInput {
        node_id: NodeId(index as u32),
        payload: ClipPayload {
            request_id: index as u32,
            node_id: NodeId(index as u32),
            clip_container: encode_field_clip(&decimated).expect("encode"),
            onsets: Vec::new(),
            sensor_frames: Vec::new(),
        },
        clock,
        position: NodePositionForSolve { position_m: position, std_m: 0.05 },
        walked_at_call: false,
        island_id: 1,
        island_epoch: 1,
        clock_epoch_at_request: 1,
    }
}

fn request_for(calls: &[TestCall], centre_s: f64, silent_nodes: &[usize]) -> CallLocalizationRequest {
    let window_start_s = centre_s - HALF_WINDOW_S;
    CallLocalizationRequest {
        call_id: 7,
        island_id: 1,
        classifying_node: NodeId(0),
        centre_reference_s: centre_s,
        half_window_s: HALF_WINDOW_S,
        clips: (0..PHONES_M.len()).map(|index| render_clip(index, calls, window_start_s, silent_nodes.contains(&index))).collect(),
        chirality_confidence: None,
        chirality_locked: false,
        region_context: None,
    }
}

fn horizontal_error(fix: &CallFix, truth: [f64; 3]) -> f64 {
    (fix.position_m[0] - truth[0]).hypot(fix.position_m[1] - truth[1])
}

fn localizer() -> OnsetClipSetLocalizer {
    OnsetClipSetLocalizer::new(OnsetClipLocalizerConfig { sound_speed_mps: SOUND_SPEED_MPS, ..OnsetClipLocalizerConfig::default() })
}

#[test]
fn one_bird_from_flac_clips_on_drifting_clocks() {
    let bird = [4.0, 5.0, 6.0];
    let calls = [TestCall { emission_s: EMISSION_S, position_m: bird, sweep_hz: (2_500.0, 4_000.0) }];
    let request = request_for(&calls, EMISSION_S + 0.3, &[]);
    let clusters = localizer().localize_window(&request).expect("fix");
    let fix = &clusters[0].cluster.fix;
    let error = horizontal_error(fix, bird);
    assert!(error < 1.0, "error {error} m, fix {:?}", fix.position_m);
    assert!(error <= 4.0 * fix.horizontal_major_sigma_m + 0.5, "σ {}", fix.horizontal_major_sigma_m);
    assert_eq!(fix.used_nodes.len(), PHONES_M.len(), "{fix:?}");
    let evidence = clusters[0].cluster.attribution_evidence.as_ref().expect("evidence");
    // Evidence arrivals are 48 kHz reference-clock samples near the true arrival.
    let arrival_0 = EMISSION_S + distance(bird, PHONES_M[0]) / SOUND_SPEED_MPS;
    let first = evidence.node_onset_arrivals["0"][0].span_sample as f64 / DEVICE_SAMPLE_RATE_HZ;
    assert!((first - arrival_0).abs() < 0.05, "evidence arrival {first} vs {arrival_0}");
    let (node, onset_s) = clusters[0].led_onsets_reference_s[0];
    let truth_s = EMISSION_S + distance(bird, PHONES_M[node.0 as usize]) / SOUND_SPEED_MPS;
    assert!((onset_s - truth_s).abs() < 0.05, "led onset {onset_s} vs {truth_s}");
}

fn two_bird_calls(second_delay_s: f64) -> ([[f64; 3]; 2], [TestCall; 2]) {
    let birds = [[-8.0, 6.0, 5.0], [11.0, -3.0, 7.0]];
    let calls = [
        TestCall { emission_s: EMISSION_S, position_m: birds[0], sweep_hz: (2_200.0, 3_200.0) },
        TestCall { emission_s: EMISSION_S + second_delay_s, position_m: birds[1], sweep_hz: (5_000.0, 6_500.0) },
    ];
    (birds, calls)
}

/// Which onsets belong to which bird is decided from the audio alone: two
/// birds 20 m apart calling 0.5 s apart in one window are two clusters.
#[test]
fn two_calls_in_one_window_are_separated_from_audio_alone() {
    let (birds, calls) = two_bird_calls(0.5);
    let request = request_for(&calls, EMISSION_S + 0.5, &[]);
    let clusters = localizer().localize_window(&request).expect("fixes");
    let fixes: Vec<[f64; 3]> = clusters.iter().map(|cluster| cluster.cluster.fix.position_m).collect();
    assert_eq!(clusters.len(), 2, "{fixes:?}");
    for bird in birds {
        let best = clusters.iter().map(|cluster| horizontal_error(&cluster.cluster.fix, bird)).fold(f64::INFINITY, f64::min);
        assert!(best < 1.5, "no cluster within 1.5 m of {bird:?}: {fixes:?}");
    }
}

/// The failure mode the arrival-time stand-in cannot have. When the second
/// call starts while the first is still sounding (0.15 s into a 0.35 s call),
/// its onset frames correlate against the first bird's ongoing sweep and are
/// clustered with it: measured, one cluster on bird 0 (holding onsets of
/// both) plus an echo. What must hold even then: no fix lands between the
/// birds, and the fix that does come out is on a bird.
#[test]
fn overlapping_calls_can_merge_but_never_make_a_phantom_between_birds() {
    let (birds, calls) = two_bird_calls(0.15);
    let request = request_for(&calls, EMISSION_S + 0.4, &[]);
    let clusters = localizer().localize_window(&request).expect("fixes");
    for cluster in &clusters {
        let nearest = birds.iter().map(|bird| horizontal_error(&cluster.cluster.fix, *bird)).fold(f64::INFINITY, f64::min);
        assert!(nearest < 1.5, "phantom fix {:?}", cluster.cluster.fix.position_m);
    }
    let onset_nodes_heard = clusters[0].led_onsets_reference_s.len();
    assert!(onset_nodes_heard >= PHONES_M.len(), "{:?}", clusters[0].led_onsets_reference_s);
}

#[test]
fn silent_clips_give_no_fix_and_a_broken_clip_is_left_out() {
    let calls = [TestCall { emission_s: EMISSION_S, position_m: [4.0, 5.0, 6.0], sweep_hz: (2_500.0, 4_000.0) }];
    let silent = request_for(&calls, EMISSION_S + 0.3, &[0, 1, 2, 3, 4]);
    assert!(localizer().localize_window(&silent).is_err());

    let mut broken = request_for(&calls, EMISSION_S + 0.3, &[]);
    broken.clips[3].payload.clip_container.truncate(12);
    let fix = localizer().localize(&broken).expect("four good clips still fix");
    assert!(!fix.used_nodes.contains(&NodeId(3)), "{fix:?}");
    assert!(horizontal_error(&fix, [4.0, 5.0, 6.0]) < 1.5, "{fix:?}");
}

#[test]
fn classifier_callback_sees_the_classifying_nodes_audio() {
    let calls = [TestCall { emission_s: EMISSION_S, position_m: [4.0, 5.0, 6.0], sweep_hz: (2_500.0, 4_000.0) }];
    let request = request_for(&calls, EMISSION_S + 0.3, &[]);
    let localizer = localizer().with_classifier(Box::new(|input: &ClusterClassificationInput<'_>| {
        let audio = input.classifying_node_audio?;
        let peak = audio.samples.iter().fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        Some((ClassifierAnswer { label: format!("node {}", audio.node_id.0), confidence: f64::from(peak), scores: Vec::new() }, 0.0))
    }));
    let clusters = localizer.localize_clusters(&request).expect("fix");
    let (answer, masked) = localizer.classify_cluster_window(&request, &clusters[0], &[]).expect("answer");
    assert_eq!(answer.label, format!("node {}", clusters[0].classifying_node.0));
    assert!(answer.confidence > 10.0 * NOISE_RMS, "window peak {}", answer.confidence);
    assert_eq!(masked, 0.0);
}

