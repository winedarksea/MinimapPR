//! CPU time of the calling thread, for the resource gate's time budgets.
//!
//! Wall time on a shared machine (parallel test binaries, other agents, a
//! busy CI runner) inflated SRP solves 2–3× and failed the 1 s budget on
//! seeds that take 0.5 s idle. Every timed region (arrival LM, layout,
//! SRP localize, a mesh session) runs on the calling thread, so thread CPU
//! time measures the work itself. Off Unix this falls back to wall time.

use std::time::Instant;

/// A started measurement; `elapsed_seconds` reads CPU seconds used since.
pub struct ThreadCpuStopwatch {
    started_cpu_seconds: Option<f64>,
    started_wall: Instant,
}

impl ThreadCpuStopwatch {
    pub fn start() -> Self {
        Self { started_cpu_seconds: thread_cpu_seconds(), started_wall: Instant::now() }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        match (self.started_cpu_seconds, thread_cpu_seconds()) {
            (Some(start), Some(now)) => now - start,
            _ => self.started_wall.elapsed().as_secs_f64(),
        }
    }

    /// Whether `elapsed_seconds` is CPU time (else wall time).
    pub fn measures_cpu_time() -> bool {
        thread_cpu_seconds().is_some()
    }
}

#[cfg(unix)]
fn thread_cpu_seconds() -> Option<f64> {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `time` is a valid, writable timespec for the call's duration.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    (status == 0).then_some(time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9)
}

#[cfg(not(unix))]
fn thread_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleeping_costs_no_cpu_time() {
        let stopwatch = ThreadCpuStopwatch::start();
        std::thread::sleep(std::time::Duration::from_millis(50));
        if ThreadCpuStopwatch::measures_cpu_time() {
            assert!(stopwatch.elapsed_seconds() < 0.025, "{}", stopwatch.elapsed_seconds());
        }
    }
}
