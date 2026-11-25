//! `validate`: runs straight after `adb pull`, so a broken run is redone on
//! the spot rather than discovered at analysis time. Errors make a run
//! unusable; warnings degrade it (blind scan, assumed temperature, …).

use std::path::Path;

use crate::bundle_loader::PhoneBundle;
use crate::bundle_schema::BundleEvent;
use crate::run_analysis::load_manifest;

/// The WAV may trail the last timestamp by one capture buffer; more than
/// this many samples short means samples were silently lost.
const MISSING_SAMPLE_TOLERANCE: f64 = 4_800.0;
/// An anchor-fit residual above this suggests timestamp glitches or drops.
const ANCHOR_RESIDUAL_WARNING_NS: f64 = 1_000_000.0;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ValidationOutcome {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
}

impl ValidationOutcome {
    pub fn is_usable(&self) -> bool {
        self.errors.is_empty()
    }
}

pub fn validate_phone(directory: &Path, label: &str, outcome: &mut ValidationOutcome) {
    let bundle = match PhoneBundle::load(directory) {
        Ok(bundle) => bundle,
        Err(error) => {
            outcome.errors.push(format!("{label}: {error}"));
            return;
        }
    };
    let audio = &bundle.audio;
    if audio.sample_rate_hz != 48_000 || audio.channel_count != 1 {
        outcome.errors.push(format!("{label}: capture.wav is {} Hz × {} ch, expected 48000 × 1", audio.sample_rate_hz, audio.channel_count));
    }
    if audio.header_declared_data_bytes as usize != audio.actual_data_bytes {
        outcome.warnings.push(format!("{label}: WAV header claims {} data bytes, file has {} (app stopped uncleanly; file length used)", audio.header_declared_data_bytes, audio.actual_data_bytes));
    }
    let last_frame = bundle.events.iter().filter_map(|e| match e {
        BundleEvent::CaptureTimestamp { frame_position, .. } => Some(*frame_position),
        _ => None,
    }).max();
    match last_frame {
        Some(frame) if (audio.samples.len() as f64) + MISSING_SAMPLE_TOLERANCE < frame as f64 => outcome.errors.push(format!(
            "{label}: WAV has {} samples but capture timestamps reached frame {frame}: samples lost before the writer",
            audio.samples.len()
        )),
        Some(_) => {}
        None => outcome.warnings.push(format!("{label}: no capture_timestamp events: self picks fall back to a blind scan")),
    }
    if let Some(anchor) = bundle.capture_anchor {
        if anchor.residual_rms_ns > ANCHOR_RESIDUAL_WARNING_NS {
            outcome.warnings.push(format!("{label}: capture timestamp fit residual {:.2} ms (glitches or silent drops)", anchor.residual_rms_ns / 1e6));
        }
    }
    let emissions = bundle.emissions();
    let played = emissions.iter().filter(|e| e.3).count();
    if played == 0 {
        outcome.errors.push(format!("{label}: no played emissions in events.jsonl"));
    }
    if emissions.iter().any(|e| e.2.is_none()) {
        outcome.warnings.push(format!("{label}: some emissions have no target_ns (no AudioTrack timestamp yet)"));
    }
    let gaps = bundle.events.iter().filter(|e| matches!(e, BundleEvent::CaptureGap { .. })).count();
    let overflows: u64 = bundle.events.iter().filter_map(|e| match e {
        BundleEvent::WriterOverflow { dropped_samples, .. } => Some(*dropped_samples),
        _ => None,
    }).sum();
    if gaps > 0 || overflows > 0 {
        outcome.warnings.push(format!("{label}: {gaps} capture gaps, {overflows} samples dropped by the writer"));
    }
    if bundle.unparsed_event_lines > 0 {
        outcome.warnings.push(format!("{label}: {} unparseable events.jsonl lines", bundle.unparsed_event_lines));
    }
    if bundle.wall_clock_anchor().is_none() {
        outcome.warnings.push(format!("{label}: no wall_ms on capture timestamps: cross-phone pairing assumes simultaneous starts"));
    }
    if bundle.session.platform_name.is_empty() {
        outcome.warnings.push(format!("{label}: session.json has no platform_name"));
    }
    outcome.notes.push(format!(
        "{label}: {} {} (SDK {}), {:.1} s audio, {played} emissions, waveform {} {:.0}-{:.0} Hz, channel {}, anchors {}",
        bundle.session.device.manufacturer, bundle.session.platform_name, bundle.session.device.sdk_int, bundle.duration_s(),
        bundle.session.schedule.waveform.kind, bundle.session.schedule.waveform.band_low_hz, bundle.session.schedule.waveform.band_high_hz,
        bundle.session.schedule.channel, bundle.capture_anchor.map_or(0, |a| a.anchor_count)
    ));
}

pub fn validate_run(run_directory: &Path) -> ValidationOutcome {
    let mut outcome = ValidationOutcome::default();
    let manifest = match load_manifest(run_directory) {
        Ok(manifest) => manifest,
        Err(error) => {
            outcome.errors.push(error);
            return outcome;
        }
    };
    if manifest.temperature_c.is_none() {
        outcome.warnings.push("manifest: no temperature_c (20 °C assumed)".into());
    }
    if manifest.tape_distance_m.is_none() {
        outcome.notes.push("manifest: no tape_distance_m (range is reported without an error)".into());
    }
    for (role, assignment) in &manifest.roles {
        validate_phone(&run_directory.join(&assignment.dir), &format!("{role} ({})", assignment.dir), &mut outcome);
    }
    if manifest.roles.len() != 2 {
        outcome.errors.push(format!("manifest lists {} roles, expected A and B", manifest.roles.len()));
    }
    outcome
}
