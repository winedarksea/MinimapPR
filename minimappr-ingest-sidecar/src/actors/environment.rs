use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct EnvSample {
    pub t_ns: u64,
    pub temp_c: f32,
    /// Relative humidity in percent (0–100).
    pub rh_pct: f32,
}

/// Per-node ring buffer of recent environmental readings, used for sound-speed
/// interpolation when binary payload environment data is absent.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentCache {
    inner: Arc<RwLock<HashMap<String, Vec<EnvSample>>>>,
}

const RING_SIZE: usize = 64;

impl EnvironmentCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn update(&self, node_id: &str, sample: EnvSample) {
        let mut map = self.inner.write().await;
        let ring = map.entry(node_id.to_string()).or_default();
        ring.push(sample);
        if ring.len() > RING_SIZE {
            ring.remove(0);
        }
    }

    /// Returns `(temp_c, humidity_fraction)` linearly interpolated at `query_ns`
    /// for `node_id`, or `None` if no readings are stored.
    pub async fn interpolate(&self, node_id: &str, query_ns: u64) -> Option<(f32, f32)> {
        let map = self.inner.read().await;
        let ring = map.get(node_id)?;
        if ring.is_empty() {
            return None;
        }
        // Binary-search for the insertion point of query_ns.
        let idx = ring.partition_point(|s| s.t_ns <= query_ns);

        let (temp_c, rh_pct) = if idx == 0 {
            let s = &ring[0];
            (s.temp_c, s.rh_pct)
        } else if idx >= ring.len() {
            let s = ring.last().unwrap();
            (s.temp_c, s.rh_pct)
        } else {
            let before = &ring[idx - 1];
            let after = &ring[idx];
            if after.t_ns == before.t_ns {
                (before.temp_c, before.rh_pct)
            } else {
                let alpha = (query_ns - before.t_ns) as f32 / (after.t_ns - before.t_ns) as f32;
                let alpha = alpha.clamp(0.0, 1.0);
                (
                    before.temp_c + alpha * (after.temp_c - before.temp_c),
                    before.rh_pct + alpha * (after.rh_pct - before.rh_pct),
                )
            }
        };

        // Return humidity as a fraction (0.0–1.0) to match the rest of the codebase.
        Some((temp_c, rh_pct / 100.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn interpolate_empty_returns_none() {
        let cache = EnvironmentCache::new();
        assert!(cache.interpolate("node1", 1_000_000).await.is_none());
    }

    #[tokio::test]
    async fn interpolate_unknown_node_returns_none() {
        let cache = EnvironmentCache::new();
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 1000,
                    temp_c: 20.0,
                    rh_pct: 50.0,
                },
            )
            .await;
        assert!(cache.interpolate("other", 1000).await.is_none());
    }

    #[tokio::test]
    async fn interpolate_before_first_clamps_to_first() {
        let cache = EnvironmentCache::new();
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 1000,
                    temp_c: 20.0,
                    rh_pct: 50.0,
                },
            )
            .await;
        let (t, h) = cache.interpolate("n1", 0).await.unwrap();
        assert_eq!(t, 20.0);
        assert!(
            (h - 0.5).abs() < 1e-6,
            "humidity fraction expected 0.5, got {h}"
        );
    }

    #[tokio::test]
    async fn interpolate_after_last_clamps_to_last() {
        let cache = EnvironmentCache::new();
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 1000,
                    temp_c: 25.0,
                    rh_pct: 60.0,
                },
            )
            .await;
        let (t, h) = cache.interpolate("n1", 9999).await.unwrap();
        assert_eq!(t, 25.0);
        assert!(
            (h - 0.6).abs() < 1e-6,
            "humidity fraction expected 0.6, got {h}"
        );
    }

    #[tokio::test]
    async fn interpolate_midpoint() {
        let cache = EnvironmentCache::new();
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 0,
                    temp_c: 20.0,
                    rh_pct: 40.0,
                },
            )
            .await;
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 1000,
                    temp_c: 30.0,
                    rh_pct: 60.0,
                },
            )
            .await;
        let (t, h) = cache.interpolate("n1", 500).await.unwrap();
        assert!((t - 25.0).abs() < 1e-4, "temp expected 25, got {t}");
        assert!((h - 0.5).abs() < 1e-4, "humidity expected 0.5, got {h}");
    }

    #[tokio::test]
    async fn interpolate_exact_sample_time() {
        let cache = EnvironmentCache::new();
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 500,
                    temp_c: 22.0,
                    rh_pct: 55.0,
                },
            )
            .await;
        cache
            .update(
                "n1",
                EnvSample {
                    t_ns: 1500,
                    temp_c: 28.0,
                    rh_pct: 65.0,
                },
            )
            .await;
        let (t, h) = cache.interpolate("n1", 500).await.unwrap();
        assert!((t - 22.0).abs() < 1e-4);
        assert!((h - 0.55).abs() < 1e-4);
    }

    #[tokio::test]
    async fn ring_evicts_oldest_when_full() {
        let cache = EnvironmentCache::new();
        for i in 0u64..=(RING_SIZE as u64) {
            cache
                .update(
                    "n1",
                    EnvSample {
                        t_ns: i * 1000,
                        temp_c: i as f32,
                        rh_pct: 50.0,
                    },
                )
                .await;
        }
        // After RING_SIZE+1 insertions the oldest entry (t_ns=0, temp=0) is evicted.
        let (t, _) = cache.interpolate("n1", 0).await.unwrap();
        assert!(
            (t - 1.0).abs() < 1e-4,
            "expected evicted first sample, oldest should be temp=1.0, got {t}"
        );
    }
}
