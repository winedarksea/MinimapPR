//! Starts the relay Worker under `wrangler dev --local` (miniflare/workerd)
//! for integration tests. Strictly local: no login, no deploy, no remote
//! bindings. Returns `None` (and says why, loudly) when the toolchain is
//! missing so `cargo test --workspace` still passes on machines without
//! Node; set `MINIMAPPR_REQUIRE_RELAY_INTEGRATION=1` to turn a skip into a
//! failure.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// First `wrangler dev` runs `worker-build --release` (~30-90 s cold).
const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(500);

pub struct LocalRelayWorker {
    pub http_base_url: String,
    child: Child,
    pub log_path: PathBuf,
    state_directory: PathBuf,
}

impl Drop for LocalRelayWorker {
    fn drop(&mut self) {
        // wrangler runs workerd as a child; kill the whole process group.
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{}", self.child.id())])
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.state_directory);
        }
    }
}

fn skip(reason: &str) -> Option<LocalRelayWorker> {
    let banner = format!("SKIPPED relay Worker integration test: {reason}");
    if std::env::var("MINIMAPPR_REQUIRE_RELAY_INTEGRATION").as_deref() == Ok("1") {
        panic!("{banner} (MINIMAPPR_REQUIRE_RELAY_INTEGRATION=1)");
    }
    eprintln!("\n========== {banner} ==========\n");
    None
}

fn free_local_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn command_succeeds(program: &str, arguments: &[&str], directory: &Path) -> bool {
    Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn wrangler_command(worker_directory: &Path) -> Command {
    let mut command = Command::new(worker_directory.join("node_modules/.bin/wrangler"));
    command
        .current_dir(worker_directory)
        // No telemetry, no interactive prompts, no account lookups.
        .env("WRANGLER_SEND_METRICS", "false")
        .env("CI", "1")
        .env_remove("CLOUDFLARE_API_TOKEN")
        .env_remove("CLOUDFLARE_ACCOUNT_ID");
    command
}

/// Starts the Worker with `vars` overriding wrangler.toml `[vars]`.
pub async fn start_local_relay_worker(vars: &[(&str, &str)]) -> Option<LocalRelayWorker> {
    let worker_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../field-relay-worker");
    if !command_succeeds("npm", &["--version"], &worker_directory) {
        return skip("npm is not installed");
    }
    if !command_succeeds("worker-build", &["--version"], &worker_directory) {
        return skip("worker-build is not installed (cargo install worker-build@0.7.5)");
    }
    if !worker_directory.join("node_modules/.bin/wrangler").exists()
        && !command_succeeds("npm", &["install", "--no-audit", "--no-fund"], &worker_directory)
    {
        return skip("`npm install` of the pinned wrangler failed (offline?)");
    }

    let state_directory =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("relay-worker-state-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&state_directory);
    std::fs::create_dir_all(&state_directory).unwrap();
    let state_argument = state_directory.to_str().unwrap().to_string();
    let migrated = wrangler_command(&worker_directory)
        .args(["d1", "migrations", "apply", "JOIN_CODES_DB", "--local", "--persist-to", &state_argument])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    assert!(migrated, "local D1 migrations failed to apply");

    let port = free_local_port();
    let inspector_port = free_local_port();
    let log_path = state_directory.join("wrangler-dev.log");
    let log_file = std::fs::File::create(&log_path).unwrap();
    let mut command = wrangler_command(&worker_directory);
    command.args([
        "dev",
        "--local",
        "--ip",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--inspector-port",
        &inspector_port.to_string(),
        "--persist-to",
        &state_argument,
        "--test-scheduled",
        "--show-interactive-dev-session=false",
    ]);
    for (name, value) in vars {
        command.args(["--var", &format!("{name}:{value}")]);
    }
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let child = command
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
        .expect("spawn wrangler dev");
    let mut worker = LocalRelayWorker {
        http_base_url: format!("http://127.0.0.1:{port}"),
        child,
        log_path,
        state_directory: state_directory.clone(),
    };

    let http = reqwest::Client::new();
    let started = Instant::now();
    loop {
        if let Ok(Some(status)) = worker.child.try_wait() {
            let log = std::fs::read_to_string(&worker.log_path).unwrap_or_default();
            panic!("wrangler dev exited early ({status}):\n{log}");
        }
        let health = http.get(format!("{}/v1/health", worker.http_base_url)).send().await;
        if health.is_ok_and(|response| response.status().is_success()) {
            return Some(worker);
        }
        if started.elapsed() > WORKER_STARTUP_TIMEOUT {
            let log = std::fs::read_to_string(&worker.log_path).unwrap_or_default();
            panic!("wrangler dev not healthy after {WORKER_STARTUP_TIMEOUT:?}:\n{log}");
        }
        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}

/// Fires the cron handler (`--test-scheduled` exposes `/__scheduled`).
pub async fn trigger_scheduled_cleanup(worker: &LocalRelayWorker) {
    let response = reqwest::get(format!("{}/__scheduled?cron=*+*+*+*+*", worker.http_base_url))
        .await
        .expect("scheduled trigger");
    assert!(response.status().is_success());
}
