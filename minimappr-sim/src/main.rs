//! `minimappr-sim` CLI.
//!
//! ```text
//! minimappr-sim render-bundle --scenario x.toml --out bundle.zip [--seed N] [--stored]
//! minimappr-sim bench --scenario x.toml [--repeats 5]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use minimappr_sim::bundle::{write_bundle, BundleCompression};
use minimappr_sim::{render_scenario, Scenario};

#[derive(Parser)]
#[command(
    name = "minimappr-sim",
    version,
    about = "MinimapPR Field Mode Layer 1 acoustic simulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render a scenario into a MinimapPR calibration-bundle zip.
    RenderBundle {
        #[arg(long)]
        scenario: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Override the scenario's seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Store audio uncompressed (faster to write, larger zip).
        #[arg(long)]
        stored: bool,
    },
    /// Time planning + rendering (excluding bundle I/O) over several runs.
    Bench {
        #[arg(long)]
        scenario: PathBuf,
        #[arg(long, default_value_t = 5)]
        repeats: usize,
    },
}

fn load(path: &Path, seed: Option<u64>) -> Result<Scenario, String> {
    let mut scenario = Scenario::from_path(path)?;
    if let Some(seed) = seed {
        scenario.seed = seed;
    }
    Ok(scenario)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::RenderBundle {
            scenario,
            out,
            seed,
            stored,
        } => {
            let scenario = load(&scenario, seed)?;
            let (plan, rendered, timing) = render_scenario(&scenario)?;
            let write_start = Instant::now();
            let compression = if stored {
                BundleCompression::Stored
            } else {
                BundleCompression::Deflated
            };
            write_bundle(&out, &scenario, &plan, &rendered, compression)?;
            eprintln!(
                "rendered {} devices, {} emissions, {:.1} s @ {} Hz: plan {:.1} ms, render {:.1} ms, write {:.1} ms -> {}",
                plan.devices.len(),
                plan.emissions.len(),
                scenario.duration_s,
                scenario.sample_rate_hz,
                milliseconds(timing.planning),
                milliseconds(timing.rendering),
                milliseconds(write_start.elapsed()),
                out.display()
            );
            Ok(())
        }
        Command::Bench { scenario, repeats } => {
            let scenario = load(&scenario, None)?;
            let mut totals = Vec::with_capacity(repeats);
            for run_index in 0..repeats.max(1) {
                let (plan, _, timing) = render_scenario(&scenario)?;
                let total = timing.planning + timing.rendering;
                eprintln!(
                    "run {run_index}: {} devices / {} channels / {} emissions: plan {:.1} ms + render {:.1} ms = {:.1} ms",
                    plan.devices.len(),
                    plan.devices.iter().map(|d| d.mic_offsets_m.len()).sum::<usize>(),
                    plan.emissions.len(),
                    milliseconds(timing.planning),
                    milliseconds(timing.rendering),
                    milliseconds(total)
                );
                totals.push(total);
            }
            totals.sort();
            println!(
                "{}: {:.1} s audio, best {:.1} ms, median {:.1} ms over {} runs",
                scenario.name,
                scenario.duration_s,
                milliseconds(totals[0]),
                milliseconds(totals[totals.len() / 2]),
                totals.len()
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("minimappr-sim: {message}");
            ExitCode::FAILURE
        }
    }
}
