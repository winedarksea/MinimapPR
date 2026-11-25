use super::*;

const SAMPLE_RATE_HZ: u32 = 48_000;
const NOISE_RMS: f32 = 1.0e-4;
const NODE: &str = "tetra-a";
const WINDOW_SECONDS: f64 = 0.08;
const FRAME_SAMPLES: usize = 4_800;

fn noise(mut seed: u32, len: usize, rms: f32) -> Vec<f32> {
    let uniform_to_unit_rms = 3f32.sqrt();
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as i32 as f32) / (i32::MAX as f32) * uniform_to_unit_rms * rms
        })
        .collect()
}

/// A 100 ms tetra frame of ambient noise with a loud tone starting at `onset_sample`.
fn frame_with_onset(seed: u32, onset_sample: usize, tone_rms: f32) -> Vec<Vec<f32>> {
    (0..4)
        .map(|channel| {
            let mut samples = noise(seed.wrapping_add(channel * 7919), FRAME_SAMPLES, NOISE_RMS);
            for (index, sample) in samples.iter_mut().enumerate().skip(onset_sample) {
                let phase =
                    2.0 * std::f32::consts::PI * 1_500.0 * index as f32 / SAMPLE_RATE_HZ as f32;
                *sample += tone_rms * std::f32::consts::SQRT_2 * phase.sin();
            }
            samples
        })
        .collect()
}

fn settled_trigger() -> LocalizationEventTrigger {
    let mut trigger = LocalizationEventTrigger::new(&DspWorkerConfig::default());
    for frame in 0..100 {
        let quiet = frame_with_onset(frame, FRAME_SAMPLES, 0.0);
        assert!(
            !trigger
                .observe_node_frame(NODE, &quiet, SAMPLE_RATE_HZ)
                .relatively_loud
        );
    }
    trigger
}

#[test]
fn onset_late_in_the_trigger_frame_is_located_to_a_block() {
    let mut trigger = settled_trigger();
    // The E5 100 m case: the burst reaches the array 8.5 ms before the frame ends.
    let onset_sample = FRAME_SAMPLES - 408;
    let node_frame = trigger.observe_node_frame(
        NODE,
        &frame_with_onset(9_001, onset_sample, 3.0e-3),
        SAMPLE_RATE_HZ,
    );
    assert!(node_frame.relatively_loud);
    let onset_seconds = node_frame.onset_offset_seconds.unwrap();
    let true_onset_seconds = onset_sample as f64 / f64::from(SAMPLE_RATE_HZ);
    assert!(
        (onset_seconds - true_onset_seconds).abs() <= ONSET_SCAN_BLOCK_SECONDS,
        "{onset_seconds}"
    );
}

#[test]
fn relative_trigger_window_starts_at_the_onset_instead_of_ending_at_the_frame() {
    let frame_end_ns: i128 = 10_000_000_000;
    let buffered_center_ns = frame_end_ns - 40_000_000;
    let node_frame = NodeFrameTrigger {
        relatively_loud: true,
        onset_offset_seconds: Some(0.0915),
        frame_seconds: 0.1,
    };
    let center = event_window_center_time_ns(
        Some(LocalizationTriggerSource::NoiseFloorRelative),
        node_frame,
        frame_end_ns,
        buffered_center_ns,
        WINDOW_SECONDS,
        true,
    );
    // Window [onset, onset + 80 ms]: onset 8.5 ms before the frame end.
    assert_eq!(center, frame_end_ns - 8_500_000 + 40_000_000);

    // The absolute gate judged the buffered window itself: it stays.
    let absolute = event_window_center_time_ns(
        Some(LocalizationTriggerSource::AbsoluteRms),
        node_frame,
        frame_end_ns,
        buffered_center_ns,
        WINDOW_SECONDS,
        true,
    );
    assert_eq!(absolute, buffered_center_ns);
    // Without the coverage wait no later audio can be read.
    let no_wait = event_window_center_time_ns(
        Some(LocalizationTriggerSource::NoiseFloorRelative),
        node_frame,
        frame_end_ns,
        buffered_center_ns,
        WINDOW_SECONDS,
        false,
    );
    assert_eq!(no_wait, buffered_center_ns);
}

#[test]
fn reboot_and_silence_reset_the_node_floor() {
    let stale_after_ns: u128 = 3_600_000_000_000;
    let mut trigger = settled_trigger();
    trigger.observe_node_session(NODE, Some(4), 1_000, stale_after_ns);
    trigger.observe_node_session(NODE, Some(4), 2_000, stale_after_ns);
    assert!(trigger.node_floor_rms(NODE).is_some());

    trigger.observe_node_session(NODE, Some(5), 3_000, stale_after_ns);
    assert!(
        trigger.node_floor_rms(NODE).is_none(),
        "reboot keeps the old floor"
    );

    let mut trigger = settled_trigger();
    trigger.observe_node_session(NODE, Some(4), 1_000, stale_after_ns);
    trigger.observe_node_session(NODE, Some(4), 1_000 + stale_after_ns, stale_after_ns);
    assert!(
        trigger.node_floor_rms(NODE).is_none(),
        "silence keeps the old floor"
    );

    let mut trigger = settled_trigger();
    trigger.observe_node_session(NODE, None, 1_000, stale_after_ns);
    trigger.evict_stale_node_floors(1_000 + stale_after_ns, stale_after_ns);
    assert!(
        trigger.node_floor_rms(NODE).is_none(),
        "purge keeps the old floor"
    );
}
