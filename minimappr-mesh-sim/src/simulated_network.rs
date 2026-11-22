//! Seeded simulated network: the cloud relay (needs a cell link at both
//! ends, enforces a per-session message cap) and the host's local hotspot
//! (range-limited around the host, joined after a delay). Latency jitter
//! reorders messages, losses are Bernoulli, bulk clips serialise on the
//! sender's uplink below control messages, and scripted drops/disconnects
//! cut individual devices. Every delivered byte is counted by message kind
//! and by whether it crossed a constrained link.

use std::collections::BTreeMap;

use minimappr_core::mesh::host_transport_policy::CellLinkQuality;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::acoustic_world::{distance, AcousticWorld};
use crate::seeded_random::purpose_stream;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkProfile {
    pub base_latency_ms: f64,
    pub jitter_ms: f64,
    pub loss_probability: f64,
    pub bulk_bandwidth_bytes_per_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkSpec {
    pub relay: LinkProfile,
    pub local: LinkProfile,
    /// Satellite / bandwidth-constrained cell link.
    pub constrained_bandwidth_bytes_per_s: f64,
    pub constrained_extra_latency_ms: f64,
    /// Phone SoftAP in foliage (plan: 20–40 m).
    pub hotspot_range_m: f64,
    pub hotspot_join_delay_s: f64,
    pub relay_message_cap: Option<u64>,
}

impl Default for NetworkSpec {
    fn default() -> Self {
        Self {
            relay: LinkProfile { base_latency_ms: 90.0, jitter_ms: 60.0, loss_probability: 0.0, bulk_bandwidth_bytes_per_s: 250_000.0 },
            local: LinkProfile { base_latency_ms: 8.0, jitter_ms: 6.0, loss_probability: 0.0, bulk_bandwidth_bytes_per_s: 1_500_000.0 },
            constrained_bandwidth_bytes_per_s: 2_000.0,
            constrained_extra_latency_ms: 600.0,
            hotspot_range_m: 35.0,
            hotspot_join_delay_s: 3.0,
            relay_message_cap: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActiveTransport {
    Relay,
    LocalHotspot { host_device: usize, started_s: f64 },
    Unavailable,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetworkStats {
    pub messages_sent: u64,
    pub messages_lost: u64,
    pub messages_unreachable: u64,
    pub relay_messages: u64,
    pub bytes_by_kind: BTreeMap<u16, u64>,
    pub clip_payloads_delivered_or_lost: u64,
    pub clip_bytes_over_constrained_links: u64,
}

#[derive(Debug, Clone)]
pub struct SimulatedNetwork {
    pub spec: NetworkSpec,
    pub transport: ActiveTransport,
    pub cell: Vec<CellLinkQuality>,
    pub alive: Vec<bool>,
    relay_dropped_until_s: Vec<f64>,
    disconnected_until_s: Vec<f64>,
    extra_bulk_delay_s: Vec<f64>,
    loss_override: Option<f64>,
    uplink_busy_until_s: Vec<f64>,
    pub relay_capped: bool,
    rng: ChaCha8Rng,
    pub stats: NetworkStats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Route {
    profile: LinkProfile,
    constrained: bool,
    via_relay: bool,
}

impl SimulatedNetwork {
    pub fn new(seed: u64, spec: NetworkSpec, device_count: usize) -> Self {
        Self {
            spec,
            transport: ActiveTransport::Relay,
            cell: vec![CellLinkQuality::Unconstrained; device_count],
            alive: vec![true; device_count],
            relay_dropped_until_s: vec![f64::NEG_INFINITY; device_count],
            disconnected_until_s: vec![f64::NEG_INFINITY; device_count],
            extra_bulk_delay_s: vec![0.0; device_count],
            loss_override: None,
            uplink_busy_until_s: vec![0.0; device_count],
            relay_capped: false,
            rng: purpose_stream(seed, "network"),
            stats: NetworkStats::default(),
        }
    }

    pub fn set_loss_override(&mut self, probability: Option<f64>) {
        self.loss_override = probability;
    }

    pub fn drop_from_relay(&mut self, device: usize, until_s: f64) {
        self.relay_dropped_until_s[device] = until_s;
    }

    pub fn disconnect(&mut self, device: usize, until_s: f64) {
        self.disconnected_until_s[device] = until_s;
    }

    pub fn set_extra_bulk_delay(&mut self, device: usize, extra_s: f64) {
        self.extra_bulk_delay_s[device] = extra_s;
    }

    fn on_relay(&self, device: usize, time_s: f64) -> bool {
        self.cell[device] != CellLinkQuality::NoSignal && time_s >= self.relay_dropped_until_s[device] && !self.relay_capped
    }

    fn on_hotspot(&self, device: usize, host_device: usize, started_s: f64, world: &AcousticWorld, time_s: f64) -> bool {
        if device == host_device {
            return true;
        }
        let in_range = distance(world.position(device, time_s), world.position(host_device, time_s)) <= self.spec.hotspot_range_m;
        in_range && time_s >= started_s + self.spec.hotspot_join_delay_s
    }

    fn attached(&self, device: usize, world: &AcousticWorld, time_s: f64) -> bool {
        if !self.alive[device] || time_s < self.disconnected_until_s[device] {
            return false;
        }
        match self.transport {
            ActiveTransport::Relay => self.on_relay(device, time_s),
            ActiveTransport::LocalHotspot { host_device, started_s } => {
                self.alive[host_device] && self.on_hotspot(device, host_device, started_s, world, time_s)
            }
            ActiveTransport::Unavailable => false,
        }
    }

    /// Whether the device has a working session link right now.
    pub fn link_up(&self, device: usize, world: &AcousticWorld, time_s: f64) -> bool {
        self.attached(device, world, time_s)
    }

    fn route(&self, from: usize, to: usize, world: &AcousticWorld, time_s: f64) -> Option<Route> {
        if !self.attached(from, world, time_s) || !self.attached(to, world, time_s) {
            return None;
        }
        match self.transport {
            ActiveTransport::Relay => {
                let constrained = self.cell[from] == CellLinkQuality::Constrained || self.cell[to] == CellLinkQuality::Constrained;
                let mut profile = self.spec.relay;
                if constrained {
                    profile.bulk_bandwidth_bytes_per_s = self.spec.constrained_bandwidth_bytes_per_s;
                    profile.base_latency_ms += self.spec.constrained_extra_latency_ms;
                }
                Some(Route { profile, constrained, via_relay: true })
            }
            _ => Some(Route { profile: self.spec.local, constrained: false, via_relay: false }),
        }
    }

    /// Returns the delivery time, or `None` if unreachable or lost.
    pub fn send(&mut self, from: usize, to: usize, frame_bytes: usize, kind: u16, bulk: bool, is_clip: bool, world: &AcousticWorld, time_s: f64) -> Option<f64> {
        self.stats.messages_sent += 1;
        if from == to {
            if !self.alive[from] {
                return None;
            }
            *self.stats.bytes_by_kind.entry(kind).or_default() += frame_bytes as u64;
            return Some(time_s);
        }
        let Some(route) = self.route(from, to, world, time_s) else {
            self.stats.messages_unreachable += 1;
            return None;
        };
        if route.via_relay {
            self.stats.relay_messages += 1;
            if self.spec.relay_message_cap.is_some_and(|cap| self.stats.relay_messages >= cap) {
                self.relay_capped = true;
            }
        }
        if is_clip {
            self.stats.clip_payloads_delivered_or_lost += 1;
            if route.constrained {
                self.stats.clip_bytes_over_constrained_links += frame_bytes as u64;
            }
        }
        let loss = self.loss_override.unwrap_or(route.profile.loss_probability);
        if self.rng.random::<f64>() < loss {
            self.stats.messages_lost += 1;
            return None;
        }
        *self.stats.bytes_by_kind.entry(kind).or_default() += frame_bytes as u64;
        let latency_s = (route.profile.base_latency_ms + route.profile.jitter_ms * self.rng.random::<f64>()) / MILLISECONDS_PER_SECOND;
        if !bulk {
            return Some(time_s + latency_s);
        }
        let start = time_s.max(self.uplink_busy_until_s[from]);
        let finish = start + frame_bytes as f64 / route.profile.bulk_bandwidth_bytes_per_s;
        self.uplink_busy_until_s[from] = finish;
        Some(finish + latency_s + self.extra_bulk_delay_s[from])
    }
}
