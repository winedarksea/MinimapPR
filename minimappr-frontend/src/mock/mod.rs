use crate::state::{
    AcousticMapLayer, AcousticMapSample, AppState, NodeStatus, RfEmitterEstimate, RfSpectrumFrame,
    SeismicTrace, TranscriptLine,
};
use futures::StreamExt;
use gloo_timers::future::IntervalStream;
use js_sys::Date;
use leptos::prelude::{GetUntracked, Set, Update};
use wasm_bindgen_futures::spawn_local;

const MAX_TRANSCRIPT_LINES: usize = 48;

pub fn start_mock_feeds(state: AppState) {
    if !mock_feeds_enabled() {
        return;
    }
    spawn_local(async move {
        let mut tick: u64 = 0;
        let mut interval = IntervalStream::new(1_500);
        while interval.next().await.is_some() {
            tick = tick.wrapping_add(1);
            let nodes = state.nodes.get_untracked();
            publish_rf_frames(&state, &nodes, tick);
            publish_seismic_traces(&state, &nodes, tick);
            publish_speech_lines(&state, &nodes, tick);
            publish_acoustic_map(&state, tick);
        }
    });
}

fn mock_feeds_enabled() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let search = window.location().search().unwrap_or_default();
    if search.contains("mock=1") || search.contains("mock=true") {
        crate::prefs::set(crate::prefs::KEY_MOCK_FEEDS, "1");
        return true;
    }
    crate::prefs::get(crate::prefs::KEY_MOCK_FEEDS)
        .map(|value| value == "1" || value == "true")
        .unwrap_or(false)
}

fn has_capability(node: &NodeStatus, capability: &str) -> bool {
    node.has_capability(capability)
}

fn node_label(node: &NodeStatus) -> String {
    node.metadata
        .as_ref()
        .and_then(|metadata| serde_json::to_value(metadata).ok())
        .and_then(|metadata| {
            metadata
                .get("label")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| node.node_id.clone())
}

fn publish_rf_frames(state: &AppState, nodes: &[NodeStatus], tick: u64) {
    let frames = nodes
        .iter()
        .filter(|node| has_capability(node, "sdr") || has_capability(node, "radar_24g"))
        .map(|node| rf_frame_for_node(node, tick))
        .collect::<Vec<_>>();
    state.modality.rf_spectra.set(frames);
}

fn publish_seismic_traces(state: &AppState, nodes: &[NodeStatus], tick: u64) {
    let traces = nodes
        .iter()
        .filter(|node| has_capability(node, "seismic"))
        .map(|node| seismic_trace_for_node(node, tick))
        .collect::<Vec<_>>();
    state.modality.seismic_traces.set(traces);
}

fn publish_speech_lines(state: &AppState, nodes: &[NodeStatus], tick: u64) {
    let speech_nodes = nodes
        .iter()
        .filter(|node| has_capability(node, "speech"))
        .collect::<Vec<_>>();
    if speech_nodes.is_empty() {
        state.modality.speech_lines.set(Default::default());
        return;
    }

    state.modality.speech_lines.update(|lines| {
        for node in speech_nodes {
            if !(tick + stable_seed(&node.node_id)).is_multiple_of(3) {
                continue;
            }
            lines.push_front(transcript_line_for_node(node, tick));
        }
        while lines.len() > MAX_TRANSCRIPT_LINES {
            lines.pop_back();
        }
    });
}

fn publish_acoustic_map(state: &AppState, tick: u64) {
    let node_positions = state
        .nodes
        .get_untracked()
        .into_iter()
        .filter_map(|node| {
            node.position_geo
                .map(|geo| (node.node_id, geo.lat, geo.lon, node.rms_history))
        })
        .collect::<Vec<_>>();
    if node_positions.is_empty() {
        state.modality.acoustic_maps.set(vec![]);
        return;
    }

    let mut samples = Vec::new();
    let mut min_lat = f64::INFINITY;
    let mut min_lon = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut max_lon = f64::NEG_INFINITY;

    for (node_id, lat, lon, rms_history) in node_positions {
        let seed = stable_seed(&node_id);
        let rms = rms_history
            .as_ref()
            .and_then(|values| values.last().copied())
            .unwrap_or(0.12);
        for row in -1_i32..=1 {
            for col in -1_i32..=1 {
                let sample_lat = lat + row as f64 * 0.00016;
                let sample_lon = lon + col as f64 * 0.00016;
                let phase = tick as f64 * 0.19 + seed as f64 * 0.0003 + row as f64 + col as f64;
                let falloff = 1.0 - ((row.abs() + col.abs()) as f64 * 0.18);
                let value = (rms * 0.65 + ((phase.sin() + 1.0) * 0.18)) * falloff;
                samples.push(AcousticMapSample {
                    lat: sample_lat,
                    lon: sample_lon,
                    value: value.clamp(0.0, 1.0),
                });
                min_lat = min_lat.min(sample_lat);
                min_lon = min_lon.min(sample_lon);
                max_lat = max_lat.max(sample_lat);
                max_lon = max_lon.max(sample_lon);
            }
        }
    }

    state.modality.acoustic_maps.set(vec![AcousticMapLayer {
        layer_id: "mock-acoustic-spl".to_string(),
        data_type: "spl_density".to_string(),
        time_window: "live_mock".to_string(),
        storey: None,
        bounds: vec![
            vec![min_lat, min_lon],
            vec![min_lat, max_lon],
            vec![max_lat, max_lon],
            vec![max_lat, min_lon],
        ],
        samples,
        updated_ns: now_ns(),
    }]);
}

fn rf_frame_for_node(node: &NodeStatus, tick: u64) -> RfSpectrumFrame {
    let seed = stable_seed(&node.node_id);
    let phase = tick as f64 * 0.23 + seed as f64 * 0.001;
    let bins = (0..36)
        .map(|index| {
            let carrier = ((index as f64 * 0.41 + phase).sin() + 1.0) * 0.18;
            let pulse_center = ((seed + tick) % 36) as i32;
            let distance = (index - pulse_center).unsigned_abs() as f64;
            let pulse = (1.0 - distance / 8.0).max(0.0) * 0.7;
            (0.08 + carrier + pulse).clamp(0.0, 1.0)
        })
        .collect::<Vec<_>>();

    let emitter_id = format!("RF-{:04X}", (seed as u16) ^ ((tick as u16) << 2));
    let lat = node
        .position_geo
        .as_ref()
        .map(|geo| geo.lat + ((phase * 0.7).sin() * 0.00025));
    let lon = node
        .position_geo
        .as_ref()
        .map(|geo| geo.lon + ((phase * 0.5).cos() * 0.00025));
    RfSpectrumFrame {
        device_id: node.node_id.clone(),
        center_mhz: if has_capability(node, "radar_24g") {
            24_125.0
        } else {
            915.0 + (seed % 5) as f64 * 2.0
        },
        bins,
        emitters: vec![RfEmitterEstimate {
            emitter_id,
            rssi_dbm: -82.0 + ((phase.sin() + 1.0) * 18.0),
            lat,
            lon,
            drone_likelihood: (((phase * 0.6).cos() + 1.0) * 0.5).clamp(0.0, 1.0),
        }],
        updated_ns: now_ns(),
    }
}

fn seismic_trace_for_node(node: &NodeStatus, tick: u64) -> SeismicTrace {
    let seed = stable_seed(&node.node_id);
    let phase = tick as f64 * 0.31 + seed as f64 * 0.0007;
    let samples = (0..48)
        .map(|index| {
            let base = ((index as f64 * 0.35 + phase).sin() + 1.0) * 0.08;
            let event_center = ((tick + seed) % 48) as i32;
            let distance = (index - event_center).unsigned_abs() as f64;
            let impulse = (1.0 - distance / 5.0).max(0.0) * 0.65;
            (base + impulse).clamp(0.0, 1.0)
        })
        .collect::<Vec<_>>();
    let peak = samples.iter().copied().fold(0.0, f64::max);

    SeismicTrace {
        device_id: node.node_id.clone(),
        samples,
        peak_velocity_mms: peak * 14.0,
        event_count: ((tick + seed) % 7) as u32,
        updated_ns: now_ns(),
    }
}

fn transcript_line_for_node(node: &NodeStatus, tick: u64) -> TranscriptLine {
    let phrases = [
        "voice activity bearing east",
        "short command-like utterance",
        "vehicle cabin speech fragment",
        "uncertain human vocalization",
        "radio-adjacent speech burst",
    ];
    let seed = stable_seed(&node.node_id);
    let phrase = phrases[((seed + tick) as usize) % phrases.len()];
    TranscriptLine {
        line_id: format!("{}-{tick}", node.node_id),
        device_id: node.node_id.clone(),
        text: format!("{}: {phrase}", node_label(node)),
        confidence: 0.52 + (((tick + seed) % 37) as f64 / 100.0),
        timestamp_ns: now_ns(),
    }
}

fn stable_seed(value: &str) -> u64 {
    value.bytes().fold(0_u64, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(byte as u64)
    })
}

fn now_ns() -> i64 {
    (Date::now() * 1_000_000.0) as i64
}
