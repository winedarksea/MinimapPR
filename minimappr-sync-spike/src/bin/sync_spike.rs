//! `sync_spike validate|analyse|analyse-single|summarize`.
//!
//!   sync_spike validate <run_dir>
//!   sync_spike analyse <run_dir> [--out <dir>]          (default: <run_dir>/analysis)
//!   sync_spike analyse-single <phone_dir>
//!   sync_spike summarize <out.md> <report.json>...

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimappr_sync_spike::analysis_config::AnalysisConfig;
use minimappr_sync_spike::bundle_validation::validate_run;
use minimappr_sync_spike::gate_evaluation::evaluate_run;
use minimappr_sync_spike::report_writer::{render_markdown, render_summary, write_run_outputs};
use minimappr_sync_spike::run_analysis::{analyse_run, analyse_single};

const USAGE: &str = "usage: sync_spike validate <run_dir> | analyse <run_dir> [--out <dir>] | analyse-single <phone_dir> | summarize <out.md> <report.json>...";

fn validate(run_directory: &Path) -> ExitCode {
    let outcome = validate_run(run_directory);
    for note in &outcome.notes {
        println!("note: {note}");
    }
    for warning in &outcome.warnings {
        println!("WARNING: {warning}");
    }
    for error in &outcome.errors {
        println!("ERROR: {error}");
    }
    println!("{}", if outcome.is_usable() { "VALID" } else { "INVALID" });
    if outcome.is_usable() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn analyse(run_directory: &Path, output_directory: Option<PathBuf>) -> Result<(), String> {
    let config = AnalysisConfig::default();
    let report = analyse_run(run_directory, &config)?;
    let gates = evaluate_run(&report, &config.gates);
    let output = output_directory.unwrap_or_else(|| run_directory.join("analysis"));
    write_run_outputs(&report, &gates, &output)?;
    println!("{}", render_markdown(&report, &gates));
    println!("wrote {}", output.display());
    Ok(())
}

fn summarize(output: &Path, inputs: &[String]) -> Result<(), String> {
    let jsons: Vec<serde_json::Value> = inputs
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
        })
        .collect::<Result<_, String>>()?;
    let summary = render_summary(&jsons, AnalysisConfig::default().gates.required_model_count);
    std::fs::write(output, &summary).map_err(|e| e.to_string())?;
    println!("{summary}");
    Ok(())
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("validate") if arguments.len() == 2 => return validate(Path::new(&arguments[1])),
        Some("analyse") if arguments.len() >= 2 => {
            let output = arguments.iter().position(|a| a == "--out").and_then(|i| arguments.get(i + 1)).map(PathBuf::from);
            analyse(Path::new(&arguments[1]), output)
        }
        Some("analyse-single") if arguments.len() == 2 => analyse_single(Path::new(&arguments[1]), &AnalysisConfig::default())
            .and_then(|summary| serde_json::to_string_pretty(&summary).map_err(|e| e.to_string()))
            .map(|json| println!("{json}")),
        Some("summarize") if arguments.len() >= 3 => summarize(Path::new(&arguments[1]), &arguments[2..]),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
