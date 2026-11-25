//! Writes `report.md`, `report.json` and `rounds.csv` for a run, and the
//! cross-run summary (gate table, range-vs-band table, channel map) that
//! `summarize` builds from many `report.json` files.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::gate_evaluation::{merge_gate_items, GateItem};
use crate::pair_ranging_series::CompensationMode;
use crate::run_analysis::RunReport;

fn optional(value: Option<f64>, digits: usize) -> String {
    value.map_or("—".into(), |v| format!("{v:.digits$}"))
}

pub fn render_markdown(report: &RunReport, gates: &[GateItem]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Sync spike run: {} ({})\n", report.run_label, report.matrix_run);
    let _ = writeln!(out, "- Waveform {} {:.0}–{:.0} Hz, {:.0} ms; period {} ms", report.waveform.kind, report.waveform.band_low_hz, report.waveform.band_high_hz, report.waveform.duration_s * 1e3, report.period_ms);
    let _ = writeln!(out, "- Tape {} m; {:.1} °C{} → c = {:.2} m/s", optional(report.tape_distance_m, 3), report.temperature_c, if report.temperature_assumed { " (assumed)" } else { "" }, report.speed_of_sound_mps);
    let _ = writeln!(out, "- d_ii A/B {:.3}/{:.3} m; speaker correction {}", report.self_paths_m[0], report.self_paths_m[1], if report.speaker_correction_applied { "applied" } else { "not applied (no pose/speaker vectors)" });
    let _ = writeln!(out, "- Rounds paired: {}\n", report.round_count);
    let _ = writeln!(out, "## Phones\n\n| role | model | SDK | source | dur s | emissions | self det/miss | self σ µs | max shift µs | remote det | remote SNR dB | gaps | overflows | blind |");
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for phone in &report.phones {
        let _ = writeln!(out, "| {} | {} | {} | {} | {:.0} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |",
            phone.role, phone.platform_name, phone.sdk_int, phone.capture_source, phone.duration_s, phone.emissions_played,
            phone.onset.detected, phone.onset.missed, optional(phone.onset.overall_sigma_us, 1), optional(phone.onset.max_segment_shift_us, 1),
            phone.remote_detected, optional(phone.median_remote_snr_db, 1), phone.capture_gaps, phone.writer_overflows, phone.blind_scanned);
    }
    let _ = writeln!(out, "\n## Range\n\n| mode | rounds | median m | robust σ mm | std mm | outliers | error vs tape mm |\n|---|---|---|---|---|---|---|");
    for stats in &report.distance {
        let _ = writeln!(out, "| {} | {} | {:.4} | {:.1} | {:.1} | {} | {} |", stats.mode, stats.count, stats.median_m, stats.robust_sigma_m * 1e3, stats.standard_deviation_m * 1e3, stats.outlier_count, optional(stats.error_vs_tape_m.map(|e| e * 1e3), 1));
    }
    if let Some(implied) = report.implied_mean_self_path_m {
        let _ = writeln!(out, "\nImplied (d_AA + d_BB)/2 from the tape: **{implied:.4} m**");
    }
    if let Some(clock) = &report.clock {
        let _ = writeln!(out, "\n## Clock (θ = B − A, from {})\n\n- Duration {:.0} s, rate {} ppm", clock.source_mode, clock.duration_s, optional(clock.rate_offset_ppm, 2));
        for (name, summary) in [("linear fit", &clock.linear_fit), ("quadratic fit", &clock.quadratic_fit), ("filter, every round", &clock.filter_every_round), ("filter, heartbeat cadence", &clock.filter_heartbeat_cadence)] {
            if let Some(s) = summary {
                let _ = writeln!(out, "- {name}: rms {:.2} µs, p95 {:.2} µs, max {:.2} µs (n = {})", s.rms_s * 1e6, s.p95_abs_s * 1e6, s.max_abs_s * 1e6, s.count);
            }
        }
        let _ = writeln!(out, "- filter outliers rejected {}, silent gaps {}", clock.filter_outliers_rejected, clock.filter_silent_gaps);
    }
    let _ = writeln!(out, "\n## Gate items\n\n| item | models | status | evidence |\n|---|---|---|---|");
    for gate in gates {
        let _ = writeln!(out, "| {} | {} | {:?} | {} |", gate.id, gate.models.join(", "), gate.status, gate.evidence);
    }
    out
}

pub fn render_rounds_csv(report: &RunReport) -> String {
    let mut out = String::from("a_emission,b_emission,a_time_s,a_hears_a_s,a_hears_b_s,b_hears_a_s,b_hears_b_s,snr_aa,snr_ab,snr_ba,snr_bb");
    for mode in CompensationMode::ALL {
        let _ = write!(out, ",distance_{0}_m,offset_{0}_s", mode.name());
    }
    out.push('\n');
    for ranged in &report.ranged_rounds {
        let round = &ranged.round;
        let t = &round.times;
        let _ = write!(out, "{},{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.1},{:.1},{:.1},{:.1}", round.a_emission_index, round.b_emission_index, round.a_time_s,
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
