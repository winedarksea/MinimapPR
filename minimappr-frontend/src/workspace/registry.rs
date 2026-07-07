#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawerId {
    Nodes,
    Tracks,
    Detections,
    Alerts,
    Zones,
    Audio,
    Rf,
    Seismic,
    Speech,
}

impl DrawerId {
    pub fn title(self) -> &'static str {
        match self {
            Self::Nodes => "Nodes",
            Self::Tracks => "Tracks",
            Self::Detections => "Detections",
            Self::Alerts => "Alerts",
            Self::Zones => "Zones",
            Self::Audio => "Audio",
            Self::Rf => "RF",
            Self::Seismic => "Seismic",
            Self::Speech => "Speech",
        }
    }
}
