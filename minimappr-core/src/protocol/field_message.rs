//! The versioned Field Mode message enum and its postcard wire framing.
//!
//! Frame = postcard(`WireHeader { major, minor, kind }`) ++ postcard(body).
//! A sealed envelope or transport frame supplies the outer length.
//!
//! Forward-compatibility rules (postcard is not self-describing):
//! - A different `major` is rejected as `UnsupportedMajorVersion`.
//! - An unknown `kind` decodes to `DecodedFrame::UnknownKind` with its raw
//!   body, so hosts can log/relay it and peers keep running.
//! - Within a major, bodies only grow by *appending* fields; a decoder
//!   ignores trailing bytes it does not understand (`take_from_bytes`).
//!   A field that cannot be appended gets a new kind id instead.
//! - Kind ids are never reused or renumbered.

use serde::{Deserialize, Serialize};

use super::detection_messages::{
    BirdEntityUpdate, ClipPayload, ClipPayloadChunk, ClipRequest, DetectionEvent, DoaBearing,
    LocoOcclusionRequest, LocoRescoredSpecies, QueuedDetectionUpload, ReadinessAnswer,
    ReadinessQuery, TapToHearPayload, TapToHearRequest,
};
use super::session_messages::{
    ChirpArrivalReport, ChirpSchedule, ClockModel, GroupStatus, Heartbeat, IslandMembership,
    JoinAccepted, JoinCodeKeyExchange, JoinRequest, LayoutModel, PeopleNearbySummary,
};

pub const FIELD_PROTOCOL_MAJOR_VERSION: u16 = 1;
pub const FIELD_PROTOCOL_MINOR_VERSION: u16 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireHeader {
    pub major_version: u16,
    pub minor_version: u16,
    pub message_kind_id: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMessageDecodeError {
    MalformedHeader,
    UnsupportedMajorVersion(u16),
    MalformedBody { message_kind_id: u16 },
}

impl std::fmt::Display for FieldMessageDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "field message decode error: {self:?}")
    }
}

impl std::error::Error for FieldMessageDecodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMessageEncodeError(pub String);

impl std::fmt::Display for FieldMessageEncodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "field message encode error: {}", self.0)
    }
}

impl std::error::Error for FieldMessageEncodeError {}

/// Result of decoding one frame.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedFrame {
    Known {
        minor_version: u16,
        message: FieldMessage,
    },
    /// Kind from a newer peer: keep the bytes, do not fail.
    UnknownKind {
        minor_version: u16,
        message_kind_id: u16,
        body: Vec<u8>,
    },
}

macro_rules! field_messages {
    ($( $kind_id:literal => $variant:ident($payload:ty) ),* $(,)?) => {
        /// Every Field Mode message kind (plan §Message protocol).
        #[derive(Debug, Clone, PartialEq)]
        pub enum FieldMessage {
            $( $variant($payload), )*
        }

        /// All kind ids this build understands.
        pub const KNOWN_MESSAGE_KIND_IDS: &[u16] = &[$( $kind_id, )*];

        impl FieldMessage {
            pub fn message_kind_id(&self) -> u16 {
                match self {
                    $( Self::$variant(_) => $kind_id, )*
                }
            }

            fn encode_body(&self) -> Result<Vec<u8>, postcard::Error> {
                match self {
                    $( Self::$variant(body) => postcard::to_allocvec(body), )*
                }
            }

            /// `None` for an unknown kind; trailing bytes are ignored.
            fn decode_body(
                message_kind_id: u16,
                body: &[u8],
            ) -> Option<Result<Self, postcard::Error>> {
                match message_kind_id {
                    $( $kind_id => Some(
                        postcard::take_from_bytes::<$payload>(body)
                            .map(|(decoded, _ignored_newer_fields)| Self::$variant(decoded)),
                    ), )*
                    _ => None,
                }
            }
        }
    };
}

field_messages! {
    1 => JoinRequest(JoinRequest),
    2 => JoinAccepted(JoinAccepted),
    3 => JoinCodeKeyExchange(JoinCodeKeyExchange),
    4 => ChirpSchedule(ChirpSchedule),
    5 => ChirpArrivalReport(ChirpArrivalReport),
    6 => IslandMembership(IslandMembership),
    7 => DetectionEvent(DetectionEvent),
    8 => ReadinessQuery(ReadinessQuery),
    9 => ReadinessAnswer(ReadinessAnswer),
    10 => DoaBearing(DoaBearing),
    11 => ClipRequest(ClipRequest),
    12 => ClipPayload(ClipPayload),
    13 => ClipPayloadChunk(ClipPayloadChunk),
    14 => LocoOcclusionRequest(LocoOcclusionRequest),
    15 => LocoRescoredSpecies(LocoRescoredSpecies),
    16 => TapToHearRequest(TapToHearRequest),
    17 => TapToHearPayload(TapToHearPayload),
    18 => LayoutModel(LayoutModel),
    19 => ClockModel(ClockModel),
    20 => BirdEntityUpdate(BirdEntityUpdate),
    21 => QueuedDetectionUpload(QueuedDetectionUpload),
    22 => Heartbeat(Heartbeat),
    23 => PeopleNearbySummary(PeopleNearbySummary),
    // Reserved in Phase 2 so the status UI can land without a wire change.
    24 => GroupStatus(GroupStatus),
}

impl FieldMessage {
    /// Encodes header + body at this build's protocol version.
    pub fn encode_frame(&self) -> Result<Vec<u8>, FieldMessageEncodeError> {
        let header = WireHeader {
            major_version: FIELD_PROTOCOL_MAJOR_VERSION,
            minor_version: FIELD_PROTOCOL_MINOR_VERSION,
            message_kind_id: self.message_kind_id(),
        };
        let mut frame = postcard::to_allocvec(&header)
            .map_err(|error| FieldMessageEncodeError(error.to_string()))?;
        frame.extend(
            self.encode_body()
                .map_err(|error| FieldMessageEncodeError(error.to_string()))?,
        );
        Ok(frame)
    }
}

/// Decodes one frame, tolerating unknown kinds and appended fields.
pub fn decode_field_message_frame(frame: &[u8]) -> Result<DecodedFrame, FieldMessageDecodeError> {
    let (header, body) = postcard::take_from_bytes::<WireHeader>(frame)
        .map_err(|_| FieldMessageDecodeError::MalformedHeader)?;
    if header.major_version != FIELD_PROTOCOL_MAJOR_VERSION {
        return Err(FieldMessageDecodeError::UnsupportedMajorVersion(
            header.major_version,
        ));
    }
    match FieldMessage::decode_body(header.message_kind_id, body) {
        Some(Ok(message)) => Ok(DecodedFrame::Known {
            minor_version: header.minor_version,
            message,
        }),
        Some(Err(_)) => Err(FieldMessageDecodeError::MalformedBody {
            message_kind_id: header.message_kind_id,
        }),
        None => Ok(DecodedFrame::UnknownKind {
            minor_version: header.minor_version,
            message_kind_id: header.message_kind_id,
            body: body.to_vec(),
        }),
    }
}
