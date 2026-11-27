//! Writes `report.md`, `report.json` and `rounds.csv` for a run, and the
//! cross-run summary (gate table, range-vs-band table, channel map) that
//! `summarize` builds from many `report.json` files.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::gate_evaluation::{merge_gate_items, GateItem};
use crate::channel_map::ChannelOffset;
use crate::pair_ranging_series::CompensationMode;
use crate::pick_structure::PickStructure;
use crate::run_analysis::RunReport;

/// Bundles are validated to be 48 kHz capture.
const CAPTURE_RATE_HZ: f64 = 48_000.0;

fn optional(value: Option<f64>, digits: usize) -> String {
    value.map_or("—".into(), |v| format!("{v:.digits$}"))
}

fn lead_modes(structure: &PickStructure) -> String {
    structure.lead_histogram.iter().take(3).map(|(lead, count)| format!("{lead}:{count}")).collect::<Vec<_>>().join(" ")
}

fn pick_structure_row(out: &mut String, label: &str, structure: &PickStructure) {
    let _ = writeln!(out, "| {label} | {} | {} | {} | {} | {}{} | {} | {} | {} | {} |",
        structure.count, optional(structure.median_margin_db, 1), structure.margin_at_risk_count, structure.not_strongest_count,
        lead_modes(structure), if structure.lead_bimodal { " **bimodal**" } else { "" }, structure.precursor_rejected_count,
        structure.double_arrival_count, structure.window_edge_count, optional(structure.suggested_self_path_m, 3));
}

const PICK_STRUCTURE_HEADER: &str = "| picks | n | median margin dB | at risk (≤ −7 dB) | not strongest | lead modes (samples:n) | precursor rej | double | window edge | suggested d_ii m |\n|---|---|---|---|---|---|---|---|---|---|";

fn render_pick_diagnostics(out: &mut String, report: &RunReport) {
    let _ = writeln!(out, "\n## Pick structure\n\nMargin = picked/strongest envelope (0 dB: the strongest peak was the pick; −10 dB is the acceptance edge). Self rows: this phone's own chirps; remote rows: the other phone's chirps heard here.\n\n{PICK_STRUCTURE_HEADER}");
    for phone in &report.phones {
        pick_structure_row(out, &format!("{} {} self", phone.role, phone.platform_name), &phone.self_pick_structure);
        pick_structure_row(out, &format!("{} {} hears other", phone.role, phone.platform_name), &phone.remote_pick_structure);
    }
    let cycled = report.channel_evidence.iter().any(|e| e.channels.len() > 1);
    let _ = writeln!(out, "\n## Channel evidence\n\nΔ against each emitter's reference channel. Self Δ = d_c − d_ref; cross Δ (other phone's pick against this phone's output frame) = x_c − x_ref, the axial speaker offset in the colinear tape layout.{}\n",
        if cycled { "" } else { " Single-channel run: the Δ columns are zero by construction." });
    let _ = writeln!(out, "| emitter | channel | self n | self Δ µs | self Δ mm | self σ smp | cross n | cross Δ µs | cross Δ mm | cross σ smp | remote SNR dB | remote margin dB | remote at risk | remote lead modes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for emitter in &report.channel_evidence {
        for channel in &emitter.channels {
            let offset_cells = |offset: &Option<ChannelOffset>| match offset {
                Some(o) => format!("{} | {:+.1} | {:+.1} | {:.2}", o.count, o.offset_us, o.offset_mm, o.robust_sigma_samples),
                None => "— | — | — | —".into(),
            };
            let reference_marker = if channel.channel == emitter.reference_channel { " (ref)" } else { "" };
            let _ = writeln!(out, "| {} {} | {}{} | {} | {} | {} | {} | {} | {} |",
                emitter.emitter_role, emitter.emitter_platform, channel.channel, reference_marker, offset_cells(&channel.self_offset), offset_cells(&channel.remote_offset),
                optional(channel.median_remote_snr_db, 1), optional(channel.remote_picks.median_margin_db, 1), channel.remote_picks.margin_at_risk_count, lead_modes(&channel.remote_picks));
        }
    }
    let _ = writeln!(out, "\n### Remote picks by the listener's last own channel\n\n| emitter channel | listener last played | n | median margin dB | at risk | lead modes | double |\n|---|---|---|---|---|---|---|");
    for emitter in &report.channel_evidence {
        for channel in &emitter.channels {
            for (last, structure) in &channel.remote_picks_by_listener_last_channel {
                let _ = writeln!(out, "| {} {} {} | {} | {} | {} | {} | {}{} | {} |", emitter.emitter_role, emitter.emitter_platform, channel.channel, last, structure.count,
                    optional(structure.median_margin_db, 1), structure.margin_at_risk_count, lead_modes(structure), if structure.lead_bimodal { " **bimodal**" } else { "" }, structure.double_arrival_count);
            }
        }
    }
    if !report.distance_by_channel.is_empty() {
        let _ = writeln!(out, "\n## Range by channel ({})\n\n| A channel | B channel | rounds | median m | robust σ mm | error vs tape mm |\n|---|---|---|---|---|---|", report.distance_by_channel_mode);
        for pair in &report.distance_by_channel {
            let stats = &pair.statistics;
            let _ = writeln!(out, "| {} | {} | {} | {:.4} | {:.1} | {} |", pair.a_channel, pair.b_channel, stats.count, stats.median_m, stats.robust_sigma_m * 1e3, optional(stats.error_vs_tape_m.map(|e| e * 1e3), 1));
        }
    }
    let _ = writeln!(out, "\n## Microphone path (active_microphones)\n");
    for phone in &report.phones {
        let path = &phone.microphone_path;
        if path.event_count == 0 {
            let _ = writeln!(out, "- {} {}: no events (app predates mic-path logging)", phone.role, phone.platform_name);
            continue;
        }
        let after: Vec<String> = path.changes_after_channel.iter().map(|(channel, count)| format!("{channel}: {count}")).collect();
        let _ = writeln!(out, "- {} {}: {} events, {} distinct states, {} changes (by preceding emission channel: {})", phone.role, phone.platform_name, path.event_count, path.states.len(), path.change_count, if after.is_empty() { "—".into() } else { after.join(", ") });
        for observation in &path.states {
            let state = &observation.state;
            let _ = writeln!(out, "  - ×{} {:.0}–{:.0} s: mics [{}], input {}, source {}, silenced {}, effects client [{}] device [{}]",
                observation.event_count, observation.first_s, observation.last_s, state.microphones.join(", "), state.routed_input.as_deref().unwrap_or("—"),
                state.client_audio_source.map_or("—".into(), |s| s.to_string()), state.client_silenced.map_or("—".into(), |s| s.to_string()),
                state.client_effects.join(", "), state.device_effects.join(", "));
        }
    }
}

pub fn render_markdown(report: &RunReport, gates: &[GateItem]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Sync spike run: {} ({})\n", report.run_label, report.matrix_run);
    let _ = writeln!(out, "- Waveform {} {:.0}–{:.0} Hz, {:.0} ms; period {} ms", report.waveform.kind, report.waveform.band_low_hz, report.waveform.band_high_hz, report.waveform.duration_s * 1e3, report.period_ms);
    let _ = writeln!(out, "- Tape {} m; {:.1} °C{} → c = {:.2} m/s", optional(report.tape_distance_m, 3), report.temperature_c, if report.temperature_assumed { " (assumed)" } else { "" }, report.speed_of_sound_mps);
    let _ = writeln!(out, "- d_ii A/B {:.3}/{:.3} m; speaker correction {}", report.self_paths_m[0], report.self_paths_m[1], if report.speaker_correction_applied { "applied" } else { "not applied (no pose/speaker vectors)" });
    let _ = writeln!(out, "- Rounds paired: {} ({} more left out: a capture/playback break overlaps them)\n", report.round_count, report.rounds_excluded_for_capture_breaks);
    let _ = writeln!(out, "## Phones\n\n| role | model | SDK | source | dur s | emissions | self det/miss | self σ µs | max shift µs | remote det | remote SNR dB | gaps | silent drops | timeline breaks | self-pick hops | overflows | blind |");
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for phone in &report.phones {
        let _ = writeln!(out, "| {} | {} | {} | {} | {:.0} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            phone.role, phone.platform_name, phone.sdk_int, phone.capture_source, phone.duration_s, phone.emissions_played,
            phone.onset.detected, phone.onset.missed, optional(phone.onset.overall_sigma_us, 1), optional(phone.onset.max_segment_shift_us, 1),
            phone.remote_detected, optional(phone.median_remote_snr_db, 1), phone.capture_gaps, phone.capture_continuity.silent_drop_count,
            phone.capture_continuity.self_interval_break_count, phone.capture_continuity.self_pick_outlier_count, phone.writer_overflows, phone.blind_scanned);
    }
    for phone in &report.phones {
        for gap in phone.capture_continuity.breaks.iter().filter(|gap| gap.kind != crate::capture_continuity::BreakKind::SelfPickOutlier) {
            let _ = writeln!(out, "- {} {:?} at {:.2}–{:.2} s: {:+.1} samples ({:+.3} ms)", phone.role, gap.kind, gap.start_sample / CAPTURE_RATE_HZ, gap.end_sample / CAPTURE_RATE_HZ, gap.discrepancy_samples, gap.discrepancy_samples / CAPTURE_RATE_HZ * 1e3);
        }
    }
    let _ = writeln!(out, "\n## Range\n\n| mode | rounds | median m | robust σ mm | std mm | outliers | error vs tape mm |\n|---|---|---|---|---|---|---|");
    for stats in &report.distance {
        let _ = writeln!(out, "| {} | {} | {:.4} | {:.1} | {:.1} | {} | {} |", stats.mode, stats.count, stats.median_m, stats.robust_sigma_m * 1e3, stats.standard_deviation_m * 1e3, stats.outlier_count, optional(stats.error_vs_tape_m.map(|e| e * 1e3), 1));
    }
    if let Some(implied) = report.implied_mean_self_path_m {
        let _ = writeln!(out, "\nImplied (d_AA + d_BB)/2 from the tape: **{implied:.4} m**");
    }
    render_pick_diagnostics(&mut out, report);
    if let Some(clock) = &report.clock {
        let _ = writeln!(out, "\n## Clock (θ = B − A, from {})\n\n- Duration {:.0} s, rate {} ppm", clock.source_mode, clock.duration_s, optional(clock.rate_offset_ppm, 2));
        for (name, summary) in [("linear fit", &clock.linear_fit), ("quadratic fit", &clock.quadratic_fit), ("filter, every round", &clock.filter_every_round), ("filter, heartbeat cadence", &clock.filter_heartbeat_cadence)] {
            if let Some(s) = summary {
                let _ = writeln!(out, "- {name}: rms {:.2} µs, p95 {:.2} µs, max {:.2} µs (n = {})", s.rms_s * 1e6, s.p95_abs_s * 1e6, s.max_abs_s * 1e6, s.count);
            }
        }
        let _ = writeln!(out, "- filter outliers rejected {}, silent gaps {}; θ steps stitched out at capture breaks {}", clock.filter_outliers_rejected, clock.filter_silent_gaps, clock.segment_steps_removed);
    }
    let _ = writeln!(out, "\n## Gate items\n\n| item | models | status | evidence |\n|---|---|---|---|");
    for gate in gates {
        let _ = writeln!(out, "| {} | {} | {:?} | {} |", gate.id, gate.models.join(", "), gate.status, gate.evidence);
    }
    out
}

pub fn render_rounds_csv(report: &RunReport) -> String {
    let mut out = String::from("a_emission,b_emission,a_channel,b_channel,a_time_s,a_hears_a_s,a_hears_b_s,b_hears_a_s,b_hears_b_s,snr_aa,snr_ab,snr_ba,snr_bb");
    for mode in CompensationMode::ALL {
        let _ = write!(out, ",distance_{0}_m,offset_{0}_s", mode.name());
    }
    out.push('\n');
    for ranged in &report.ranged_rounds {
        let round = &ranged.round;
        let t = &round.times;
        let channel = |c: &Option<String>| c.clone().unwrap_or_default();
        let _ = write!(out, "{},{},{},{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.1},{:.1},{:.1},{:.1}", round.a_emission_index, round.b_emission_index,
            channel(&round.channels[0]), channel(&round.channels[1]), round.a_time_s,
            t.a_hears_a_s, t.a_hears_b_s, t.b_hears_a_s, t.b_hears_b_s, round.snr_db[0], round.snr_db[1], round.snr_db[2], round.snr_db[3]);
        for mode in CompensationMode::ALL {
            match ranged.result(mode) {
                Some(result) => { let _ = write!(out, ",{:.5},{:.9}", result.distance_m, result.clock_offset_b_minus_a_s); }
                None => out.push_str(",,"),
            }
        }
        out.push('\n');
    }
    out
}

pub fn write_run_outputs(report: &RunReport, gates: &[GateItem], output_directory: &Path) -> Result<(), String> {
    fs::create_dir_all(output_directory).map_err(|e| e.to_string())?;
    let json = serde_json::json!({ "report": report, "gate_items": gates });
    fs::write(output_directory.join("report.json"), serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fs::write(output_directory.join("report.md"), render_markdown(report, gates)).map_err(|e| e.to_string())?;
    fs::write(output_directory.join("rounds.csv"), render_rounds_csv(report)).map_err(|e| e.to_string())
}

/// Cross-run summary from `report.json` files.
pub fn render_summary(report_jsons: &[serde_json::Value], required_model_count: usize) -> String {
    let gates: Vec<GateItem> = report_jsons
        .iter()
        .filter_map(|json| serde_json::from_value::<Vec<GateItem>>(json.get("gate_items")?.clone()).ok())
        .flatten()
        .collect();
    let mut out = String::from("# Phase 1 sync spike summary\n\n## Gate\n\n| item | status | models | runs |\n|---|---|---|---|\n");
    for merged in merge_gate_items(&gates, required_model_count) {
        let _ = writeln!(out, "| {} | {:?} | {} | {} |", merged.id, merged.status, merged.models.join(", "), merged.runs.join("; "));
    }
    out.push_str("\n## Range vs band and distance\n\n| run | waveform | band Hz | tape m | mode | median m | error mm | robust σ mm | rounds | remote SNR A/B dB |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for json in report_jsons {
        let Some(report) = json.get("report") else { continue };
        let waveform = &report["waveform"];
        let snrs: Vec<String> = report["phones"].as_array().map_or(Vec::new(), |phones| {
            phones.iter().map(|p| p["median_remote_snr_db"].as_f64().map_or("—".into(), |s| format!("{s:.0}"))).collect()
        });
        for stats in report["distance"].as_array().into_iter().flatten() {
            let _ = writeln!(out, "| {} {} | {} | {:.0}–{:.0} | {} | {} | {:.4} | {} | {:.1} | {} | {} |",
                report["matrix_run"].as_str().unwrap_or(""), report["run_label"].as_str().unwrap_or(""),
                waveform["kind"].as_str().unwrap_or(""), waveform["band_low_hz"].as_f64().unwrap_or(0.0), waveform["band_high_hz"].as_f64().unwrap_or(0.0),
                optional(report["tape_distance_m"].as_f64(), 2), stats["mode"].as_str().unwrap_or(""), stats["median_m"].as_f64().unwrap_or(f64::NAN),
                optional(stats["error_vs_tape_m"].as_f64().map(|e| e * 1e3), 1), stats["robust_sigma_m"].as_f64().unwrap_or(f64::NAN) * 1e3,
                stats["count"].as_u64().unwrap_or(0), snrs.join("/"));
        }
    }
    out.push_str("\n## Speaker channel map\n\n| run | emitter | channel | orientation | listener | median SNR dB | heard |\n|---|---|---|---|---|---|---|\n");
    for json in report_jsons {
        let Some(report) = json.get("report") else { continue };
        for observation in report["channel_map"].as_array().into_iter().flatten() {
            let _ = writeln!(out, "| {} {} | {} | {} | {} | {} | {} | {}/{} |",
                report["matrix_run"].as_str().unwrap_or(""), report["run_label"].as_str().unwrap_or(""),
                observation["emitter_platform"].as_str().unwrap_or(""), observation["emitter_channel"].as_str().unwrap_or(""),
                observation["emitter_orientation"].as_str().unwrap_or(""), observation["listener_platform"].as_str().unwrap_or(""),
                optional(observation["median_remote_snr_db"].as_f64(), 1), observation["remote_detected"], observation["remote_expected"]);
        }
    }
    out
}
