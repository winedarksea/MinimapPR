//! Which transport the host runs (plan §Networking: Auto, Local only,
//! Online only). Auto uses the relay while the host has an unconstrained
//! cell link and the session is under its relay caps; a lost, constrained
//! (satellite, bandwidth-limited) or capped link moves it to the local
//! hotspot. Online-only falls back to detection-only. A switch is always
//! announced and logged, so no detection is lost silently. Auto does not
//! flip back to the relay on its own: every switch makes phones rejoin.

use crate::protocol::wire_codes::{SessionChangeReasonCode, SessionTransportCode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellLinkQuality {
    Unconstrained,
    /// Satellite or bandwidth-constrained: treated as no signal.
    Constrained,
    NoSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionTransportPreference {
    Auto,
    LocalOnly,
    OnlineOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransportDecision {
    pub transport: SessionTransportCode,
    pub reason: SessionChangeReasonCode,
    pub clip_fetch_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct HostTransportPolicy {
    preference: SessionTransportPreference,
    active: SessionTransportCode,
    cell: CellLinkQuality,
    relay_cap_reached: bool,
}

impl HostTransportPolicy {
    pub fn new(preference: SessionTransportPreference, cell: CellLinkQuality) -> (Self, TransportDecision) {
        let mut policy = Self {
            preference,
            active: SessionTransportCode::SoloDetectionOnly,
            cell,
            relay_cap_reached: false,
        };
        let reason = match cell {
            CellLinkQuality::Constrained => SessionChangeReasonCode::HostCellLinkConstrained,
            _ => SessionChangeReasonCode::SessionStarted,
        };
        let transport = policy.desired();
        policy.active = transport;
        let decision = policy.decision(reason);
        (policy, decision)
    }

    /// Continue an inherited transport (successor host).
    pub fn inherit(preference: SessionTransportPreference, cell: CellLinkQuality, active: SessionTransportCode) -> Self {
        Self {
            preference,
            active,
            cell,
            relay_cap_reached: false,
        }
    }

    fn relay_usable(&self) -> bool {
        self.cell == CellLinkQuality::Unconstrained && !self.relay_cap_reached
    }

    fn desired(&self) -> SessionTransportCode {
        match self.preference {
            SessionTransportPreference::LocalOnly => SessionTransportCode::LocalHotspot,
            SessionTransportPreference::OnlineOnly if self.relay_usable() => SessionTransportCode::CloudRelay,
            SessionTransportPreference::OnlineOnly => SessionTransportCode::SoloDetectionOnly,
            SessionTransportPreference::Auto if self.active == SessionTransportCode::LocalHotspot => {
                SessionTransportCode::LocalHotspot
            }
            SessionTransportPreference::Auto if self.relay_usable() => SessionTransportCode::CloudRelay,
            SessionTransportPreference::Auto => SessionTransportCode::LocalHotspot,
        }
    }

    fn decision(&self, reason: SessionChangeReasonCode) -> TransportDecision {
        TransportDecision {
            transport: self.active,
            reason,
            clip_fetch_enabled: self.clip_fetch_enabled(),
        }
    }

    fn switch_if_needed(&mut self, reason: SessionChangeReasonCode) -> Option<TransportDecision> {
        let desired = self.desired();
        if desired == self.active {
            return None;
        }
        self.active = desired;
        Some(self.decision(reason))
    }

    pub fn on_cell_link(&mut self, quality: CellLinkQuality) -> Option<TransportDecision> {
        self.cell = quality;
        let reason = match quality {
            CellLinkQuality::Unconstrained => SessionChangeReasonCode::HostCellLinkRestored,
            CellLinkQuality::Constrained => SessionChangeReasonCode::HostCellLinkConstrained,
            CellLinkQuality::NoSignal => SessionChangeReasonCode::HostCellLinkLost,
        };
        self.switch_if_needed(reason)
    }

    pub fn on_relay_cap_reached(&mut self) -> Option<TransportDecision> {
        self.relay_cap_reached = true;
        self.switch_if_needed(SessionChangeReasonCode::RelayMessageCapReached)
    }

    pub fn active(&self) -> SessionTransportCode {
        self.active
    }

    /// No clips in detection-only mode, and never over a constrained link.
    pub fn clip_fetch_enabled(&self) -> bool {
        match self.active {
            SessionTransportCode::SoloDetectionOnly => false,
            SessionTransportCode::CloudRelay => self.cell == CellLinkQuality::Unconstrained,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_moves_to_hotspot_on_constrained_or_capped_relay() {
        let (mut policy, start) = HostTransportPolicy::new(SessionTransportPreference::Auto, CellLinkQuality::Unconstrained);
        assert_eq!(start.transport, SessionTransportCode::CloudRelay);
        let switch = policy.on_cell_link(CellLinkQuality::Constrained).expect("switches");
        assert_eq!(switch.transport, SessionTransportCode::LocalHotspot);
        assert_eq!(switch.reason, SessionChangeReasonCode::HostCellLinkConstrained);
        assert!(policy.on_cell_link(CellLinkQuality::Unconstrained).is_none());

        let (mut online, _) = HostTransportPolicy::new(SessionTransportPreference::OnlineOnly, CellLinkQuality::Unconstrained);
        let capped = online.on_relay_cap_reached().expect("switches");
        assert_eq!(capped.transport, SessionTransportCode::SoloDetectionOnly);
        assert!(!capped.clip_fetch_enabled);
    }
}
