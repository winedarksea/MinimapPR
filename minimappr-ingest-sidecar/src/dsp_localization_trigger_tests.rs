use super::*;

const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_SAMPLES: usize = 320;
const QUIET_SITE_NOISE_RMS: f32 = 1.0e-4;
const STREAM: &str = "cluster::site::audio";
const NODE: &str = "node-a";
const DEFAULT_COOLDOWN_SECONDS: f64 = 0.8;

/// The same-event window of a lone node: the cooldown.
fn plain_cooldown(_last_trigger_node_id: &str) -> f64 {
    DEFAULT_COOLDOWN_SECONDS
}

fn trigger_config(relative_enabled: bool) -> DspWorkerConfig {
    let mut config = DspWorkerConfig::default();
    config
        .localization_solve
        .localization_trigger_noise_floor_relative_enabled = relative_enabled;
    config
}

fn white_noise(mut seed: u32, len: usize, rms: f32) -> Vec<f32> {
    // Uniform in [-1, 1] has RMS 1/sqrt(3).
    let uniform_to_unit_rms = 3f32.sqrt();
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as i32 as f32) / (i32::MAX as f32) * uniform_to_unit_rms * rms
        })
        .collect()
}

fn tetra_frame(seed: u32, rms: f32) -> Vec<Vec<f32>> {
    (0..4)
        .map(|channel| white_noise(seed.wrapping_add(channel * 7919), FRAME_SAMPLES, rms))
        .collect()
}

/// Ten seconds of steady quiet-site noise so the node floor settles.
fn settle_quiet_floor(trigger: &mut LocalizationEventTrigger) -> usize {
    (0..500)
        .filter(|frame| {
            trigger
                .observe_node_frame(
                    NODE,
                    &tetra_frame(*frame, QUIET_SITE_NOISE_RMS),
                    SAMPLE_RATE_HZ,
                )
                .relatively_loud
        })
        .count()
}

#[test]
fn relative_trigger_fires_on_a_quiet_site_below_the_absolute_gate() {
    let config = trigger_config(true);
    let mut trigger = LocalizationEventTrigger::new(&config);
    assert_eq!(settle_quiet_floor(&mut trigger), 0);

    // 12 dB over the floor, still ~40x under the 0.015 absolute gate.
    let faint_call = tetra_frame(9_001, QUIET_SITE_NOISE_RMS * 4.0);
    assert!(window_rms(&faint_call[0]) < config.localization_rms_gate);
    let relatively_loud = trigger
        .observe_node_frame(NODE, &faint_call, SAMPLE_RATE_HZ)
        .relatively_loud;
    assert!(relatively_loud);
    assert_eq!(
        trigger.evaluate(STREAM, 1, &faint_call, relatively_loud, plain_cooldown),
        Some(LocalizationTriggerSource::NoiseFloorRelative)
    );
    assert_eq!(
        LocalizationTriggerSource::NoiseFloorRelative.attempt_kind(),
        SolveAttemptKind::EventTriggerNoiseFloorRelative
    );
}

#[test]
fn absolute_gate_still_fires_and_wins_a_tie() {
    let config = trigger_config(true);
    let mut trigger = LocalizationEventTrigger::new(&config);
    settle_quiet_floor(&mut trigger);
    let loud_call = tetra_frame(9_002, 0.05);
    let relatively_loud = trigger
        .observe_node_frame(NODE, &loud_call, SAMPLE_RATE_HZ)
        .relatively_loud;
    assert!(relatively_loud);
    assert_eq!(
        trigger.evaluate(STREAM, 1, &loud_call, relatively_loud, plain_cooldown),
        Some(LocalizationTriggerSource::AbsoluteRms)
    );
    // Absolute alone (relative off) is today's behaviour.
    let absolute_only = LocalizationEventTrigger::new(&trigger_config(false));
    assert_eq!(
        absolute_only.evaluate(STREAM, 1, &loud_call, false, plain_cooldown),
        Some(LocalizationTriggerSource::AbsoluteRms)
    );
}

#[test]
fn disabled_relative_trigger_still_tracks_the_floor_but_never_fires() {
    let config = trigger_config(false);
    let mut trigger = LocalizationEventTrigger::new(&config);
    settle_quiet_floor(&mut trigger);
    let faint_call = tetra_frame(9_003, QUIET_SITE_NOISE_RMS * 4.0);
    assert!(
        !trigger
            .observe_node_frame(NODE, &faint_call, SAMPLE_RATE_HZ)
            .relatively_loud
    );
    assert_eq!(trigger.evaluate(STREAM, 1, &faint_call, false, plain_cooldown), None);
    let layout = crate::dsp_worker::solve_layout_test_fixture::tetra_nodes_layout(&[(
        NODE,
        [0.0, 0.0, 0.0],
    )]);
    let floor = trigger.node_noise_floor_snapshot(&layout)[NODE];
    assert!(
        (floor / QUIET_SITE_NOISE_RMS - 1.0).abs() < 0.5,
        "floor {floor}"
    );
}

#[test]
fn cooldown_starts_only_when_the_trigger_is_committed() {
    let config = trigger_config(true);
    let mut trigger = LocalizationEventTrigger::new(&config);
    let loud_call = tetra_frame(9_004, 0.05);
    let evaluate = |trigger: &LocalizationEventTrigger, audio_ns: u128| {
        trigger.evaluate(STREAM, audio_ns, &loud_call, false, plain_cooldown)
    };
    let cooldown_ns = (config.trigger_cooldown_seconds * 1e9) as u128;
    // Decided but suppressed downstream: the next frame may still trigger.
    assert!(evaluate(&trigger, 1_000).is_some());
    assert!(evaluate(&trigger, 2_000).is_some());
    assert_eq!(trigger.last_dispatched_trigger_ns(STREAM), None);

    trigger.commit_dispatched_trigger(STREAM, 2_000, NODE);
    assert!(evaluate(&trigger, 2_000 + cooldown_ns - 1).is_none());
    assert!(evaluate(&trigger, 2_000 + cooldown_ns).is_some());
    // Either order: an earlier onset inside the window is the same event.
    assert!(evaluate(&trigger, 1_500).is_none());
    trigger.forget_stream(STREAM);
    assert_eq!(trigger.last_dispatched_trigger_ns(STREAM), None);
}

/// E7 cares about the false-trigger rate. Measured over two minutes of
/// pure noise (20 ms frames, 0.8 s cooldown): steady and +-3 dB gusts never
/// fire; +-6 dB gusts on a 4 s cycle fire 149 frames but only 7 solves after
/// the cooldown, mostly before the fluctuation allowance has built up.
#[test]
fn relative_trigger_rate_on_pure_noise_stays_low() {
    let config = trigger_config(true);
    let frames = 120 * SAMPLE_RATE_HZ as usize / FRAME_SAMPLES;
    let cooldown_frames = (config.trigger_cooldown_seconds * f64::from(SAMPLE_RATE_HZ)
        / FRAME_SAMPLES as f64) as usize;
    let gust_period_frames = 4.0 * SAMPLE_RATE_HZ as f32 / FRAME_SAMPLES as f32;
    let solves_on_gusts = |swing_db: f32| {
        let mut trigger = LocalizationEventTrigger::new(&config);
        let mut last_solve_frame: Option<usize> = None;
        let mut solves = 0;
        for frame in 0..frames {
            let phase = std::f32::consts::TAU * frame as f32 / gust_period_frames;
            let rms = QUIET_SITE_NOISE_RMS * 10f32.powf(swing_db * phase.sin() / 20.0);
            let fired = trigger
                .observe_node_frame(NODE, &tetra_frame(frame as u32, rms), SAMPLE_RATE_HZ)
                .relatively_loud;
            if fired && last_solve_frame.is_none_or(|last| frame - last >= cooldown_frames) {
                solves += 1;
                last_solve_frame = Some(frame);
            }
        }
        solves
    };
    assert_eq!(solves_on_gusts(0.0), 0);
    assert_eq!(solves_on_gusts(3.0), 0);
    let strong_gust_solves = solves_on_gusts(6.0);
    assert!(
        strong_gust_solves <= 8,
        "strong gust solves {strong_gust_solves}"
    );
}

/// A far node's onset of the same event arrives up to the inter-node lag
/// later; it must not reopen (and supersede) the solve waiting for it, while
/// the trigger node itself may fire again after the plain cooldown.
#[test]
fn a_far_nodes_onset_inside_the_inter_node_lag_is_the_same_event() {
    let config = trigger_config(true);
    let mut trigger = LocalizationEventTrigger::new(&config);
    let loud_call = tetra_frame(9_005, 0.05);
    let lag_seconds = 4.0;
    let window_seconds = |last_trigger_node_id: &str, trigger_node_id: &str| {
        DEFAULT_COOLDOWN_SECONDS + if last_trigger_node_id == trigger_node_id { 0.0 } else { lag_seconds }
    };
    let seconds_ns = |seconds: f64| (seconds * 1e9) as u128;
    trigger.commit_dispatched_trigger(STREAM, seconds_ns(10.0), NODE);
    let far_onset = |audio_ns| {
        trigger.evaluate(STREAM, audio_ns, &loud_call, false, |last| window_seconds(last, "node-far"))
    };
    assert!(far_onset(seconds_ns(10.0 + lag_seconds)).is_none());
    assert!(far_onset(seconds_ns(10.0 + lag_seconds + DEFAULT_COOLDOWN_SECONDS + 0.01)).is_some());
    let same_node_onset = |audio_ns| {
        trigger.evaluate(STREAM, audio_ns, &loud_call, false, |last| window_seconds(last, NODE))
    };
    assert!(same_node_onset(seconds_ns(10.5)).is_none());
    assert!(same_node_onset(seconds_ns(10.0 + DEFAULT_COOLDOWN_SECONDS + 0.01)).is_some());
}

/// A 5 ms chip 12 dB over the floor, at the end of an otherwise quiet frame.
fn tetra_frame_with_chip(seed: u32) -> Vec<Vec<f32>> {
    let chip_samples = (SAMPLE_RATE_HZ as usize) / 200;
    let mut frame = tetra_frame(seed, QUIET_SITE_NOISE_RMS);
    for (channel_index, channel) in frame.iter_mut().enumerate() {
        let chip = white_noise(
            seed.wrapping_add(31 * channel_index as u32),
            chip_samples,
            QUIET_SITE_NOISE_RMS * 4.0,
        );
        channel[FRAME_SAMPLES - chip_samples..].copy_from_slice(&chip);
    }
    frame
}

#[test]
fn short_block_excess_candidate_admits_a_chip_only_when_enabled() {
    let mut enabled_config = trigger_config(true);
    enabled_config
        .localization_solve
        .localization_trigger_short_block_excess_enabled = true;
    let mut short_block = LocalizationEventTrigger::new(&enabled_config);
    let mut whole_frame = LocalizationEventTrigger::new(&trigger_config(true));
    assert_eq!(settle_quiet_floor(&mut short_block), 0);
    assert_eq!(settle_quiet_floor(&mut whole_frame), 0);

    let chip_frame = tetra_frame_with_chip(77_001);
    assert!(
        !whole_frame
            .observe_node_frame(NODE, &chip_frame, SAMPLE_RATE_HZ)
            .relatively_loud
    );
    assert!(
        short_block
            .observe_node_frame(NODE, &chip_frame, SAMPLE_RATE_HZ)
            .relatively_loud
    );
    // The floor follows the frame level either way.
    assert_eq!(short_block.node_floor_rms(NODE), whole_frame.node_floor_rms(NODE));
}
