use leptos::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct RfEmitterEstimate {
    pub emitter_id: String,
    pub rssi_dbm: f64,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub drone_likelihood: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RfSpectrumFrame {
    pub device_id: String,
    pub center_mhz: f64,
    pub bins: Vec<f64>,
    pub emitters: Vec<RfEmitterEstimate>,
    pub updated_ns: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SeismicTrace {
    pub device_id: String,
    pub samples: Vec<f64>,
    pub peak_velocity_mms: f64,
    pub event_count: u32,
    pub updated_ns: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptLine {
    pub line_id: String,
    pub device_id: String,
    pub text: String,
    pub confidence: f64,
    pub timestamp_ns: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AcousticMapSample {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AcousticMapLayer {
    pub layer_id: String,
    pub data_type: String,
    pub time_window: String,
    pub storey: Option<String>,
    pub bounds: Vec<Vec<f64>>,
    pub samples: Vec<AcousticMapSample>,
    pub updated_ns: i64,
}

#[derive(Clone)]
pub struct ModalityFeeds {
    pub rf_spectra: RwSignal<Vec<RfSpectrumFrame>>,
    pub seismic_traces: RwSignal<Vec<SeismicTrace>>,
    pub speech_lines: RwSignal<VecDeque<TranscriptLine>>,
    pub acoustic_maps: RwSignal<Vec<AcousticMapLayer>>,
}

impl ModalityFeeds {
    pub fn new() -> Self {
        Self {
            rf_spectra: RwSignal::new(vec![]),
            seismic_traces: RwSignal::new(vec![]),
            speech_lines: RwSignal::new(VecDeque::new()),
            acoustic_maps: RwSignal::new(vec![]),
        }
    }
}
