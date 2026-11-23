//! Nightly Layer 2 runner: every scenario on N fresh seeds.
//!
//! cargo run --release -p minimappr-mesh-sim --example nightly_mesh_seeds -- \
//!     --seeds 200 [--start-seed S] [--scenario NAME[,NAME…]] [--threads T] [--audio]
//!
//! `--audio` runs in rendered-audio mode (closeout D2): Layer 1 audio clips
//! and the real onset-clustering localizer instead of the arrival-time
//! stand-in. It is slow (seconds per run), so it is nightly-only.
//! `--audio-waveform sweep|trill|harsh` (implies `--audio`) picks the Layer 1
//! call waveform; `--no-ground` (implies `--audio`) drops the ground reflection.
//!
//! Seeds are `start, start+1, …`; the start defaults to the current Unix
//! time so each night draws fresh seeds, and is printed so a run can be
//! repeated. Each failure prints the seed and the exact command that
//! reproduces it; the exit code is 1 if anything failed. A fixed failing
//! seed joins `PULL_REQUEST_SEEDS` after its fix (test strategy §Seeds).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use minimappr_mesh_sim::audio_world::{AcousticMode, AudioCallWaveform, AudioModeConfig};
use minimappr_mesh_sim::{run_scenario_by_name_in_mode, run_scenario_by_name_with_audio, SCENARIO_NAMES};

const DEFAULT_SEED_COUNT: u64 = 200;
const DEFAULT_THREAD_COUNT: usize = 8;

struct Options {
    seed_count: u64,
    start_seed: u64,
    scenarios: Option<Vec<String>>,
    threads: usize,
    mode: AcousticMode,
    waveform: AudioCallWaveform,
    ground_reflection: bool,
}

fn parse_options() -> Options {
    let mut options = Options {
        seed_count: DEFAULT_SEED_COUNT,
        start_seed: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs()),
        scenarios: None,
        mode: AcousticMode::ArrivalTimes,
        waveform: AudioCallWaveform::FmSweep,
        ground_reflection: true,
        threads: std::thread::available_parallelism().map_or(DEFAULT_THREAD_COUNT, |n| n.get()),
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--seeds" => options.seed_count = value().parse().expect("--seeds takes a count"),
            "--start-seed" => options.start_seed = value().parse().expect("--start-seed takes an integer"),
            "--scenario" => options.scenarios = Some(value().split(',').map(str::to_string).collect()),
            "--audio" => options.mode = AcousticMode::RenderedAudio,
            "--no-ground" => {
                options.mode = AcousticMode::RenderedAudio;
                options.ground_reflection = false;
            }
            "--audio-waveform" => {
                options.mode = AcousticMode::RenderedAudio;
                options.waveform = match value().as_str() {
                    "sweep" => AudioCallWaveform::FmSweep,
                    "trill" => AudioCallWaveform::Trill,
                    "harsh" => AudioCallWaveform::Harsh,
                    other => panic!("--audio-waveform takes sweep|trill|harsh, not {other}"),
                };
            }
            "--threads" => options.threads = value().parse().expect("--threads takes a count"),
            other => panic!("unknown flag {other}"),
        }
    }
    options
}

fn main() {
    let options = parse_options();
    let scenarios: Vec<&str> = SCENARIO_NAMES
        .iter()
        .copied()
        .filter(|name| options.scenarios.as_ref().is_none_or(|wanted| wanted.iter().any(|w| w == name)))
        .collect();
    let jobs: Vec<(&str, u64)> = (0..options.seed_count)
        .flat_map(|offset| scenarios.iter().map(move |name| (*name, options.start_seed + offset)))
        .collect();
    println!(
        "nightly mesh sim ({:?}): {} scenarios x {} seeds from start seed {} on {} threads",
        options.mode,
        scenarios.len(),
        options.seed_count,
        options.start_seed,
        options.threads
    );
    let started = Instant::now();
    let next_job = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..options.threads.max(1) {
            scope.spawn(|| loop {
                let index = next_job.fetch_add(1, Ordering::Relaxed);
                let Some(&(name, seed)) = jobs.get(index) else { break };
                let outcome = std::panic::catch_unwind(|| match options.mode {
                    AcousticMode::RenderedAudio => {
                        run_scenario_by_name_with_audio(name, seed, AudioModeConfig {
                            call_waveform: options.waveform,
                            ground_reflection_enabled: options.ground_reflection,
                            ..AudioModeConfig::default()
                        })
                    }
                    AcousticMode::ArrivalTimes => run_scenario_by_name_in_mode(name, seed, options.mode),
                });
                let failure = match outcome {
                    Ok(Ok(_)) => None,
                    Ok(Err(message)) => Some(message),
                    Err(_) => Some(format!("{name} seed {seed}: panicked")),
                };
                if let Some(message) = failure {
                    failures.lock().expect("failure list").push((name, seed, message));
                }
            });
        }
    });
    let mut failures = failures.into_inner().expect("failure list");
    failures.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    for name in &scenarios {
        let failed = failures.iter().filter(|failure| failure.0 == *name).count() as u64;
        println!("scenario {name}: {} / {} passed", options.seed_count - failed, options.seed_count);
    }
    for (name, seed, message) in &failures {
        println!("FAIL {message}\n  reproduce: MINIMAPPR_SIM_SEED={seed} cargo test -p minimappr-mesh-sim {name}");
    }
    println!(
        "{} runs, {} failed, {:.1} s",
        jobs.len(),
        failures.len(),
        started.elapsed().as_secs_f64()
    );
    if !failures.is_empty() {
        std::process::exit(1);
    }
}
