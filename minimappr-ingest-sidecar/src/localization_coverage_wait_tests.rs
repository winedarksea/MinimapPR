use super::*;

const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_SAMPLES: usize = 512;
const FRAME_DURATION_NS: i128 = 32_000_000;
const STREAM_START_NS: i128 = 1_000_000_000_000;
const BUFFER_SECONDS: f64 = 4.0;
const WINDOW_SECONDS: f64 = FRAME_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
const MIN_COVERAGE_RATIO: f64 = 0.85;
const MILLISECOND_NS: u128 = 1_000_000;

fn wait_config() -> LocalizationCoverageWaitConfig {
    LocalizationCoverageWaitConfig::from_seconds(true, 0.5, 1.0)
}

/// Two "nodes" of two channels each, with `frames_by_node[n]` frames buffered.
fn buffers_with_frames(frames_by_node: [usize; 2]) -> Vec<SensorStreamBuffer> {
    let mut buffers = Vec::new();
    for frame_count in frames_by_node {
        for _ in 0..2 {
            let mut buffer = SensorStreamBuffer::new(SAMPLE_RATE_HZ, BUFFER_SECONDS);
            append_frames(&mut buffer, 0, frame_count);
            buffers.push(buffer);
        }
    }
    buffers
}

fn append_frames(buffer: &mut SensorStreamBuffer, first_frame: usize, frame_count: usize) {
    for frame_index in first_frame..first_frame + frame_count {
        let start_sample_index = (frame_index * FRAME_SAMPLES) as i64;
        buffer
            .append(
                STREAM_START_NS + frame_index as i128 * FRAME_DURATION_NS,
                &vec![0.25; FRAME_SAMPLES],
                Some(start_sample_index),
                Some(start_sample_index + FRAME_SAMPLES as i64),
            )
            .unwrap();
    }
}

/// Window covering frame `frame_index` exactly (centred like the worker does).
fn span_for_frame(frame_index: usize) -> LocalizationWindowSpan {
    let frame_end_ns = STREAM_START_NS + (frame_index as i128 + 1) * FRAME_DURATION_NS;
    LocalizationWindowSpan {
        center_time_ns: frame_end_ns - FRAME_DURATION_NS / 2,
        window_seconds: WINDOW_SECONDS,
        min_coverage_ratio: MIN_COVERAGE_RATIO,
    }
}

fn observed_tracker(buffers: &[SensorStreamBuffer], now_ns: u128) -> ChannelStreamActivityTracker {
    let mut tracker = ChannelStreamActivityTracker::default();
    tracker.observe(buffers, now_ns);
    tracker
}

#[test]
fn all_channels_covered_is_ready_without_waiting() {
    let buffers = buffers_with_frames([10, 10]);
    let tracker = observed_tracker(&buffers, 0);
    let decision =
        evaluate_window_coverage(&buffers, span_for_frame(9), &tracker, wait_config(), 0, 0);
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_ALL_COVERED
        }
    );
}

#[test]
fn late_but_advancing_node_is_waited_for() {
    // Node b is one frame behind node a: the race this module exists for.
    let buffers = buffers_with_frames([10, 9]);
    let tracker = observed_tracker(&buffers, 0);
    let decision = evaluate_window_coverage(
        &buffers,
        span_for_frame(9),
        &tracker,
        wait_config(),
        5 * MILLISECOND_NS,
        5 * MILLISECOND_NS,
    );
    assert_eq!(decision, CoverageWaitDecision::Wait);
}

#[test]
fn node_that_stops_advancing_is_released_after_stall_budget() {
    let buffers = buffers_with_frames([10, 9]);
    let tracker = observed_tracker(&buffers, 0);
    let config = wait_config();
    let just_under_budget_ns = config.stall_budget_ns - MILLISECOND_NS;
    assert_eq!(
        evaluate_window_coverage(
            &buffers,
            span_for_frame(9),
            &tracker,
            config,
            just_under_budget_ns,
            just_under_budget_ns
        ),
        CoverageWaitDecision::Wait
    );
    assert_eq!(
        evaluate_window_coverage(
            &buffers,
            span_for_frame(9),
            &tracker,
            config,
            config.stall_budget_ns,
            config.stall_budget_ns
        ),
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_STALLED
        }
    );
}

#[test]
fn dead_node_costs_nothing_on_later_windows() {
    // The tracker persists, so once node b's last advance is older than the
    // stall budget a fresh window is released immediately (waited = 0).
    let mut buffers = buffers_with_frames([10, 9]);
    let mut tracker = observed_tracker(&buffers, 0);
    for buffer in buffers.iter_mut().take(2) {
        append_frames(buffer, 10, 20);
    }
    let config = wait_config();
    let later_now_ns = config.stall_budget_ns + 100 * MILLISECOND_NS;
    tracker.observe(&buffers, later_now_ns);
    let decision = evaluate_window_coverage(
        &buffers,
        span_for_frame(29),
        &tracker,
        config,
        0,
        later_now_ns,
    );
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_STALLED
        }
    );
}

#[test]
fn total_cap_releases_even_while_a_node_keeps_advancing() {
    let buffers = buffers_with_frames([10, 9]);
    let config = LocalizationCoverageWaitConfig::from_seconds(true, 10.0, 0.2);
    // Node b "advanced" just now, so only the cap can release this window.
    let tracker = observed_tracker(&buffers, config.total_cap_ns);
    let decision = evaluate_window_coverage(
        &buffers,
        span_for_frame(9),
        &tracker,
        config,
        config.total_cap_ns,
        config.total_cap_ns,
    );
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_TOTAL_CAP
        }
    );
}

#[test]
fn node_lagging_by_more_audio_than_the_cap_is_skipped_immediately() {
    // 40 frames x 32 ms = 1.28 s of audio behind: cannot catch up in 1.0 s.
    let buffers = buffers_with_frames([50, 10]);
    let tracker = observed_tracker(&buffers, 0);
    let decision =
        evaluate_window_coverage(&buffers, span_for_frame(49), &tracker, wait_config(), 0, 0);
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_STALLED
        }
    );
}

#[test]
fn channels_that_never_streamed_are_ignored() {
    let mut buffers = buffers_with_frames([10, 10]);
    buffers.push(SensorStreamBuffer::new(SAMPLE_RATE_HZ, BUFFER_SECONDS));
    let tracker = observed_tracker(&buffers, 0);
    let decision =
        evaluate_window_coverage(&buffers, span_for_frame(9), &tracker, wait_config(), 0, 0);
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_ALL_COVERED
        }
    );
}

#[test]
fn gap_inside_an_already_delivered_window_is_not_waited_on() {
    // Node b delivered frames 0..8 and 10, but frame 9 was lost in transit.
    let mut buffers = buffers_with_frames([11, 9]);
    for buffer in buffers.iter_mut().skip(2) {
        append_frames(buffer, 10, 1);
    }
    let tracker = observed_tracker(&buffers, 0);
    let decision =
        evaluate_window_coverage(&buffers, span_for_frame(9), &tracker, wait_config(), 0, 0);
    assert_eq!(
        decision,
        CoverageWaitDecision::Ready {
            stop_reason: STOP_REASON_ALL_COVERED
        }
    );
}

#[test]
fn disabled_config_never_waits() {
    let buffers = buffers_with_frames([10, 9]);
    let tracker = observed_tracker(&buffers, 0);
    let disabled = LocalizationCoverageWaitConfig::from_seconds(false, 0.5, 1.0);
    let decision = evaluate_window_coverage(&buffers, span_for_frame(9), &tracker, disabled, 0, 0);
    assert!(matches!(decision, CoverageWaitDecision::Ready { .. }));
}

#[test]
fn config_from_seconds_clamps_non_finite_and_negative_values_to_zero() {
    let config = LocalizationCoverageWaitConfig::from_seconds(true, -1.0, f64::NAN);
    assert_eq!(config.stall_budget_ns, 0);
    assert_eq!(config.total_cap_ns, 0);
    let defaults = LocalizationCoverageWaitConfig::from_seconds(true, 0.5, 1.0);
    assert_eq!(defaults.stall_budget_ns, 500 * MILLISECOND_NS);
    assert_eq!(defaults.total_cap_ns, 1_000 * MILLISECOND_NS);
}
