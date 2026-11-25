use super::*;
use serde_json::Value;

const PARITY_FIXTURE_JSON: &str =
    include_str!("../../tests/fixtures/timeline_discipline_parity_cases.json");

fn policy_from_json(value: &Value) -> TimelineDisciplinePolicy {
    TimelineDisciplinePolicy {
        step_reanchor_enabled: value["step_reanchor_enabled"].as_bool().unwrap(),
        step_threshold_ns: i128::from(value["step_threshold_ns"].as_i64().unwrap()),
        step_confirm_frames: value["step_confirm_frames"].as_u64().unwrap() as usize,
        drift_slew_threshold_ns: i128::from(value["drift_slew_threshold_ns"].as_i64().unwrap()),
        hard_limit_ns: i128::from(value["hard_limit_ns"].as_i64().unwrap()),
    }
}

fn correction_from_json(value: &Value) -> Option<TimelineCorrection> {
    if value.is_null() {
        return None;
    }
    let kind = match value["kind"].as_str().unwrap() {
        "step_reanchor" => TimelineCorrectionKind::StepReanchor,
        "drift_slew" => TimelineCorrectionKind::DriftSlew,
        other => panic!("unknown correction kind {other}"),
    };
    Some(TimelineCorrection {
        kind,
        origin_shift_ns: i128::from(value["origin_shift_ns"].as_i64().unwrap()),
    })
}

fn optional_ns(value: &Value) -> Option<i128> {
    value.as_i64().map(i128::from)
}

#[test]
fn decisions_match_the_shared_parity_fixture() {
    let fixture: Value = serde_json::from_str(PARITY_FIXTURE_JSON).unwrap();
    let default_policy = &fixture["default_policy"];
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let policy = policy_from_json(case.get("policy").unwrap_or(default_policy));
        let mut discipline = TimelineDiscipline::new(policy);
        for (op_index, op) in case["ops"].as_array().unwrap().iter().enumerate() {
            let actual = if let Some(frame) = op.get("frame") {
                discipline.observe_frame(
                    optional_ns(&frame["residual_ns"]).unwrap(),
                    optional_ns(&frame["detected_step_ns"]),
                )
            } else {
                discipline.drift_slew(optional_ns(&op["slew"]["windowed_residual_ns"]))
            };
            assert_eq!(
                actual,
                correction_from_json(&op["expect"]),
                "{name}: op {op_index}"
            );
        }
    }
}

#[test]
fn policy_from_seconds_scales_the_slew_threshold_off_the_hard_limit() {
    let policy = TimelineDisciplinePolicy::from_seconds(true, 0.002, 0, 0.5, 0.010);
    assert_eq!(policy.step_threshold_ns, 2_000_000);
    assert_eq!(policy.step_confirm_frames, 1, "at least the flagged frame");
    assert_eq!(policy.drift_slew_threshold_ns, 5_000_000);
    assert_eq!(policy.hard_limit_ns, 10_000_000);
}

#[test]
fn a_cancelled_step_leaves_no_pending_state() {
    let policy = TimelineDisciplinePolicy::from_seconds(true, 0.002, 3, 0.5, 0.010);
    let mut discipline = TimelineDiscipline::new(policy);
    assert_eq!(discipline.observe_frame(21_000_000, Some(21_000_000)), None);
    assert!(discipline.step_pending());
    discipline.cancel_pending_step();
    assert!(!discipline.step_pending());
    // Without a pending step, an unflagged frame starts nothing.
    assert_eq!(discipline.observe_frame(21_000_000, None), None);
    assert!(!discipline.step_pending());
}
