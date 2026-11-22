//! Exponential reconnect backoff (deterministic; no jitter source needed in
//! the sans-IO core — an adapter can add jitter when it schedules).

use crate::transport_config::RelayTransportConfig;

#[derive(Debug, Clone)]
pub struct ReconnectBackoff {
    initial_delay_milliseconds: u64,
    maximum_delay_milliseconds: u64,
    multiplier: u64,
    next_delay_milliseconds: u64,
}

impl ReconnectBackoff {
    pub fn new(config: &RelayTransportConfig) -> Self {
        Self {
            initial_delay_milliseconds: config.reconnect_initial_backoff_milliseconds,
            maximum_delay_milliseconds: config.reconnect_maximum_backoff_milliseconds,
            multiplier: u64::from(config.reconnect_backoff_multiplier.max(1)),
            next_delay_milliseconds: config.reconnect_initial_backoff_milliseconds,
        }
    }

    /// Delay before the next attempt; grows until reset by a welcome.
    pub fn next_delay_milliseconds(&mut self) -> u64 {
        let delay = self.next_delay_milliseconds;
        self.next_delay_milliseconds = delay
            .saturating_mul(self.multiplier)
            .min(self.maximum_delay_milliseconds);
        delay
    }

    pub fn reset(&mut self) {
        self.next_delay_milliseconds = self.initial_delay_milliseconds;
    }
}
