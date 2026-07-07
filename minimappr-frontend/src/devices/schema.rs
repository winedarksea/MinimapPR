use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Radar24G,
    SdrRf,
    Seismic,
    SpeechNode,
}

impl DeviceKind {
    pub const ALL: [Self; 4] = [Self::Radar24G, Self::SdrRf, Self::Seismic, Self::SpeechNode];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Radar24G => "radar_24g",
            Self::SdrRf => "sdr_rf",
            Self::Seismic => "seismic",
            Self::SpeechNode => "speech_node",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Radar24G => "24G radar",
            Self::SdrRf => "SDR / RF",
            Self::Seismic => "Seismic",
            Self::SpeechNode => "Speech node",
        }
    }

    pub fn drawer_label(self) -> &'static str {
        match self {
            Self::Radar24G => "Radar",
            Self::SdrRf => "RF",
            Self::Seismic => "Seismic",
            Self::SpeechNode => "Speech",
        }
    }

    pub fn parse(value: &str) -> Self {
        match value {
            "radar_24g" => Self::Radar24G,
            "sdr_rf" => Self::SdrRf,
            "seismic" => Self::Seismic,
            "speech_node" => Self::SpeechNode,
            _ => Self::Radar24G,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeviceRecord {
    pub id: String,
    pub kind: DeviceKind,
    pub label: String,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub alt_m: Option<f64>,
    pub yaw_deg: Option<f64>,
    pub pitch_deg: Option<f64>,
    pub created_ns: i64,
}

impl DeviceRecord {
    pub fn display_label(&self) -> String {
        if self.label.trim().is_empty() {
            self.id.clone()
        } else {
            self.label.clone()
        }
    }

    pub fn orientation_label(&self) -> String {
        match (self.yaw_deg, self.pitch_deg) {
            (Some(yaw), Some(pitch)) => format!("yaw {yaw:.1} deg, pitch {pitch:.1} deg"),
            (Some(yaw), None) => format!("yaw {yaw:.1} deg"),
            (None, Some(pitch)) => format!("pitch {pitch:.1} deg"),
            (None, None) => "-".to_string(),
        }
    }
}
