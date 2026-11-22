//! Identifiers and forward-compatible small enums shared by message bodies.
//!
//! Nested enums travel as a single `u8` code with an `Unrecognized(code)`
//! fallback instead of a serde enum: postcard is not self-describing, so an
//! unknown serde variant inside a body would make the *whole* message
//! undecodable on an older peer. A raw code degrades to "unknown" instead.

use serde::{Deserialize, Serialize};

/// Session-scoped node id assigned by the host at join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Recipient value meaning "every node in the session".
    pub const BROADCAST: NodeId = NodeId(u32::MAX);
}

/// 128-bit session id (from the QR code / Worker registration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub [u8; 16]);

/// Defines a `u8`-coded enum with an `Unrecognized(u8)` catch-all.
macro_rules! wire_code_enum {
    (
        $(#[$enum_meta:meta])*
        $name:ident { $( $(#[$variant_meta:meta])* $variant:ident = $code:literal ),* $(,)? }
    ) => {
        $(#[$enum_meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(from = "u8", into = "u8")]
        pub enum $name {
            $( $(#[$variant_meta])* $variant, )*
            /// Code from a newer peer; carried through unchanged.
            Unrecognized(u8),
        }

        impl From<u8> for $name {
            fn from(code: u8) -> Self {
                match code {
                    $( $code => Self::$variant, )*
                    other => Self::Unrecognized(other),
                }
            }
        }

        impl From<$name> for u8 {
            fn from(value: $name) -> u8 {
                match value {
                    $( $name::$variant => $code, )*
                    $name::Unrecognized(other) => other,
                }
            }
        }
    };
}

wire_code_enum! {
    /// What kind of device a node is.
    NodeKind {
        Phone = 1,
        /// MinimapPR tetrahedral or XVF3800 array node.
        ArrayNode = 2,
        /// iOS PWA (never eligible to host).
        BrowserPwa = 3,
        /// At-home MinimapPR server acting as host.
        HomeServer = 4,
    }
}

wire_code_enum! {
    /// Phone placement reported in heartbeats.
    PhonePlacement {
        Unknown = 0,
        Pocket = 1,
        Surface = 2,
        Handheld = 3,
    }
}

wire_code_enum! {
    /// Walking/still state from the step detector.
    MotionState {
        Unknown = 0,
        Still = 1,
        Walking = 2,
        Handling = 3,
    }
}

wire_code_enum! {
    /// Role a joiner asks for (tokens per role, as in pcplayerpicker).
    SessionRole {
        Joiner = 1,
        ViewOnly = 2,
        Host = 3,
    }
}

wire_code_enum! {
    /// Bird-entity lifecycle shown on every map.
    BirdEntityState {
        HeardNearby = 1,
        DirectionKnown = 2,
        Pinpointed = 3,
        Expired = 4,
    }
}

wire_code_enum! {
    /// Reserved group status (Phase 2 reserves the wire format; UI later).
    GroupStatusCode {
        TakingBreak = 1,
        HeadingHome = 2,
        RequestAssistance = 3,
    }
}

wire_code_enum! {
    /// Which flight of the join-code key exchange a message carries.
    KeyExchangeFlight {
        /// Joiner -> host: SPAKE2 message A.
        JoinerSpakeMessage = 1,
        /// Host -> joiner: SPAKE2 message B (reveals nothing about the code).
        HostSpakeMessage = 2,
        /// Joiner -> host: joiner key-confirmation tag (joiner proves first,
        /// so the host counts every wrong guess before revealing anything).
        JoinerConfirmation = 3,
        /// Host -> joiner: session key sealed under the exchanged key.
        SealedSessionKey = 4,
        /// Host -> joiner: attempt rejected (code wrong or attempts exhausted).
        Rejected = 5,
    }
}

wire_code_enum! {
    /// W3-mesh-sim: why a node's audio or clock status changed.
    NodeStatusChangeCode {
        /// Input route changed (e.g. Bluetooth HFP mic): hard clock invalidation.
        InputRouteChanged = 1,
        /// Output-only route change; chirps stay on the built-in speaker.
        OutputRouteChangedSpeakerPinned = 2,
        /// Output route moved chirps off the speaker: node becomes listen-only.
        OutputRouteChangedListenOnly = 3,
        /// Capture gap or overrun reported by the OS: hard clock invalidation.
        CaptureGapReported = 4,
        /// Own chirp missing from the self-recording: excluded from ranging.
        MediaVolumeMuted = 5,
        /// Low thermal headroom: node stopped classifying (Mic-only).
        ThermalMicOnly = 6,
        /// Audio session interrupted (call, another recorder).
        AudioSessionInterrupted = 7,
    }
}

wire_code_enum! {
    /// W3-mesh-sim: which transport carries the session right now.
    SessionTransportCode {
        CloudRelay = 1,
        LocalHotspot = 2,
        /// No shared transport: every node runs solo, detection-only.
        SoloDetectionOnly = 3,
    }
}

wire_code_enum! {
    /// W3-mesh-sim: why the transport or the host changed.
    SessionChangeReasonCode {
        SessionStarted = 1,
        RelayMessageCapReached = 2,
        HostCellLinkLost = 3,
        /// Satellite or bandwidth-constrained cell link: treated as no signal.
        HostCellLinkConstrained = 4,
        HostSilent = 5,
        HostThermalHeadroomLow = 6,
        HostLeftSession = 7,
        HostCellLinkRestored = 8,
    }
}

/// One classifier score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesScore {
    /// Index into the session's agreed label table.
    pub label_id: u32,
    pub score: f32,
}

/// Frequency band in Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrequencyBandHz {
    pub low_hz: f32,
    pub high_hz: f32,
}
