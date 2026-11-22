//! Per-island clip budget: a token bucket on clips fetched, so a dawn
//! chorus cannot turn into unbounded clip traffic at the host. Calls that
//! find the bucket short are localized with fewer clips or skipped, and
//! every shortfall is counted.

use super::MeshMilliseconds;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

#[derive(Debug, Clone)]
pub struct IslandClipBudget {
    tokens: f64,
    burst: f64,
    refill_per_s: f64,
    last_refill_ms: MeshMilliseconds,
}

impl IslandClipBudget {
    pub fn new(burst: f64, refill_per_s: f64, now_ms: MeshMilliseconds) -> Self {
        Self {
            tokens: burst,
            burst,
            refill_per_s,
            last_refill_ms: now_ms,
        }
    }

    fn refill(&mut self, now_ms: MeshMilliseconds) {
        let elapsed_s = now_ms.saturating_sub(self.last_refill_ms) as f64 / MILLISECONDS_PER_SECOND;
        self.tokens = (self.tokens + elapsed_s * self.refill_per_s).min(self.burst);
        self.last_refill_ms = now_ms;
    }

    /// Grants up to `wanted` clips (whole tokens only).
    pub fn take(&mut self, wanted: usize, now_ms: MeshMilliseconds) -> usize {
        self.refill(now_ms);
        let granted = (self.tokens.floor() as usize).min(wanted);
        self.tokens -= granted as f64;
        granted
    }

    pub fn available(&mut self, now_ms: MeshMilliseconds) -> usize {
        self.refill(now_ms);
        self.tokens.floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_caps_bursts_and_refills() {
        let mut budget = IslandClipBudget::new(6.0, 1.0, 0);
        assert_eq!(budget.take(4, 0), 4);
        assert_eq!(budget.take(4, 0), 2);
        assert_eq!(budget.take(4, 0), 0);
        assert_eq!(budget.take(4, 3_000), 3);
        assert_eq!(budget.available(100_000), 6);
    }
}
