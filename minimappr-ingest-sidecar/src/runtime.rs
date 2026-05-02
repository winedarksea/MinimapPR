use rayon::ThreadPool;
use std::sync::OnceLock;

static DSP_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Returns the global DSP rayon thread pool, sized to `physical_cpus - 1` (min 1).
/// All CPU-bound DSP work (SRP-PHAT, FFT, beamforming) runs here, leaving Tokio
/// worker threads free for HTTP and I/O.
pub fn dsp_pool() -> &'static ThreadPool {
    DSP_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get_physical().saturating_sub(1).max(1))
            .thread_name(|i| format!("dsp-worker-{i}"))
            .build()
            .expect("failed to build DSP rayon pool")
    })
}
