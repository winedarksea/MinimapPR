//! Call-path scenarios: baseline localization, readiness cap, Mic-only and
//! round-robin classification, faint bird heard by one classifier.
//! All use the arrival-time abstraction (no acoustics beyond SNR and timing).

use crate::device_spec::DeviceSpec;
use crate::invariants::{ensure, event_logs_converge, fixes_within_gate, fixes_within_reported_region, no_silent_detection_loss};
use crate::scenario::{ring_of_phones, ScenarioSpec};
use crate::scenario_outcome::ScenarioOutcome;

use super::{report, run_checked, ScenarioReport};

/// Test strategy gate inside the layout: horizontal P95 < 3 m.
pub const INSIDE_LAYOUT_GATE_M: f64 = 3.0;
pub const REPORTED_REGION_SIGMA_MULTIPLIER: f64 = 4.0;
pub const REPORTED_REGION_FLOOR_M: f64 = 0.5;
/// Mirrors must have caught up with what the host logged this long ago.
pub const CONVERGENCE_LAG_S: f64 = 3.0;
const PHONE_HEIGHT_M: f64 = 1.4;

/// Fixes, gate, region, no loss, convergence: the healthy-path checks.
/// `gate_m`: absolute accuracy gate for well-posed geometry; scenarios about
/// degraded geometry pass `None` and rely on the reported-region check.
pub(crate) fn healthy_path_checks(outcome: &ScenarioOutcome, expected_fixes: usize, gate_m: Option<f64>) -> Result<f64, String> {
    let fixes = outcome.scored_fixes();
    ensure(fixes.len() >= expected_fixes, || {
        let failures: Vec<_> = outcome.call_records().iter().filter_map(|r| r.failure.clone()).collect();
        format!("expected {expected_fixes} fixes, got {} (failures {failures:?})", fixes.len())
    })?;
    if let Some(gate_m) = gate_m {
        fixes_within_gate(&fixes, gate_m)?;
    }
    fixes_within_reported_region(&fixes, REPORTED_REGION_SIGMA_MULTIPLIER, REPORTED_REGION_FLOOR_M)?;
    no_silent_detection_loss(outcome)?;
    event_logs_converge(outcome, CONVERGENCE_LAG_S)?;
    Ok(fixes.iter().map(|f| f.horizontal_error_m).fold(0.0, f64::max))
}

pub fn baseline_localization(seed: u64) -> Result<ScenarioReport, String> {
    let spec = ScenarioSpec::new("baseline_localization", 70.0, ring_of_phones(5, [0.0, 0.0], 12.0, seed))
        .call(20.0, [3.0, 2.0, 6.0])
        .call(35.0, [-5.0, 6.0, 9.0])
        .call(50.0, [8.0, -7.0, 4.0]);
    report("baseline_localization", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let worst = healthy_path_checks(&outcome, 3, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("3/3 fixes, worst {worst:.2} m"))
    })())
}

/// 8 synced phones all ready for one call, cap 6: exactly 6 clips, the
/// classifier among them, the 2 others counted (never silently dropped).
pub fn readiness_cap(seed: u64) -> Result<ScenarioReport, String> {
    const PHONE_COUNT: usize = 8;
    const CAP: usize = 6;
    let mut spec = ScenarioSpec::new("readiness_cap", 50.0, ring_of_phones(PHONE_COUNT, [0.0, 0.0], 10.0, seed))
        .call(25.0, [2.0, -3.0, 5.0]);
    spec.mesh.clip_fetch.readiness_clip_cap = CAP;
    report("readiness_cap", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let records = outcome.call_records();
        let record = records.iter().find(|r| r.fix.is_some()).ok_or("call not localized")?;
        ensure(record.requested_nodes.len() == CAP, || format!("{} clips requested, cap {CAP}", record.requested_nodes.len()))?;
        ensure(record.requested_nodes.contains(&record.classifying_node), || "classifying node not fetched".into())?;
        ensure(record.readiness_excluded_by_cap == PHONE_COUNT - CAP, || format!("{} excluded by cap, expected {}", record.readiness_excluded_by_cap, PHONE_COUNT - CAP))?;
        let (_, host) = outcome.active_host().ok_or("no host")?;
        ensure(host.calls.counters.readiness_excluded_by_cap >= (PHONE_COUNT - CAP) as u64, || "cap exclusions not counted".into())?;
        let worst = healthy_path_checks(&outcome, 1, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("{CAP} of {PHONE_COUNT} fetched, {} counted, error {worst:.2} m", record.readiness_excluded_by_cap))
    })())
}

/// Two Mic-only phones still contribute clips; a 3-phone clump within 3 m
/// classifies each call about once, not three times.
pub fn mic_only_round_robin(seed: u64) -> Result<ScenarioReport, String> {
    const CLUMP: [usize; 3] = [3, 4, 5];
    const MIC_ONLY: [usize; 2] = [1, 2];
    const CALL_COUNT: usize = 8;
    const MAXIMUM_MEAN_CLUMP_CLASSIFICATIONS: f64 = 1.3;
    let mut devices = vec![
        DeviceSpec::android_at([-10.0, -6.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([10.0, -8.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-8.0, 9.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([8.0, 6.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([9.5, 7.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([7.6, 8.1, PHONE_HEIGHT_M]),
    ];
    for device in MIC_ONLY {
        devices[device].can_classify = false;
    }
    let mut spec = ScenarioSpec::new("mic_only_round_robin", 80.0, devices);
    for index in 0..CALL_COUNT {
        let angle = index as f64 * 0.9;
        spec = spec.call(20.0 + 6.0 * index as f64, [4.0 * angle.cos(), 4.0 * angle.sin(), 6.0]);
    }
    report("mic_only_round_robin", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        for device in MIC_ONLY {
            ensure(!outcome.metrics.generated_detections.iter().any(|(_, d, _)| *d == device), || format!("Mic-only phone {device} classified"))?;
            let contributed = outcome.call_records().iter().any(|r| r.fix.as_ref().is_some_and(|fix| fix.used_nodes.iter().any(|n| n.0 as usize == device)));
            ensure(contributed, || format!("Mic-only phone {device} never contributed a clip"))?;
        }
        let per_call: Vec<usize> = (0..CALL_COUNT)
            .map(|call| outcome.metrics.classifier_runs.get(&call).map_or(0, |runs| runs.iter().filter(|d| CLUMP.contains(d)).count()))
            .collect();
        let mean = per_call.iter().sum::<usize>() as f64 / CALL_COUNT as f64;
        ensure(mean <= MAXIMUM_MEAN_CLUMP_CLASSIFICATIONS && per_call.iter().all(|n| *n <= 2), || format!("clump classified {per_call:?} times per call"))?;
        let worst = healthy_path_checks(&outcome, CALL_COUNT, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("clump classifications per call {mean:.2}, worst {worst:.2} m"))
    })())
}

/// Bird 10 m from one phone, ~35 m from three: only the near phone's
/// Stage-1 and classifier fire, yet the readiness query brings in the far
/// phones' clips and the fix meets the gate.
pub fn faint_bird_one_classifier(seed: u64) -> Result<ScenarioReport, String> {
    const NEAR_PHONE: usize = 0;
    const FAINT_CALL_LEVEL_DB_SPL: f64 = 70.0;
    let devices = vec![
        DeviceSpec::android_at([0.0, 0.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-24.0, -10.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-22.0, 12.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-30.0, 2.0, PHONE_HEIGHT_M]),
    ];
    let spec = ScenarioSpec::new("faint_bird_one_classifier", 60.0, devices)
        .call_with_level(25.0, [9.0, 3.0, 4.0], FAINT_CALL_LEVEL_DB_SPL)
        .call_with_level(40.0, [8.0, -2.0, 5.0], FAINT_CALL_LEVEL_DB_SPL);
    report("faint_bird_one_classifier", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        for call in 0..2 {
            let triggered = outcome.metrics.stage1_triggers.get(&call).cloned().unwrap_or_default();
            ensure(triggered == vec![NEAR_PHONE], || format!("call {call}: Stage-1 fired on {triggered:?}"))?;
        }
        for record in outcome.call_records() {
            ensure(record.clip_nodes.len() == 4, || format!("call {}: clips from {:?}, wanted all 4", record.call_id, record.clip_nodes))?;
        }
        let worst = healthy_path_checks(&outcome, 2, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("1 classifier, 4 clips per call, worst {worst:.2} m"))
    })())
}
