//! Orchestrates mic capture -> Stage-1 -> BirdNET -> feed/relay.
//!
//! Audio arrives on the main thread from the worklet (20 ms chunks); Stage-1
//! runs inline (cheap), BirdNET runs one window at a time off an async task
//! so a slow inference only queues windows (bounded, oldest dropped and
//! counted) and never stalls capture.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::prelude::{GetUntracked, Set, Update};
use serde::Serialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen_futures::spawn_local;

use super::birdnet_input_resampler::WindowedSincResampler;
use super::browser_bindings::{
    classify_with_birdnet, load_birdnet, publish_diagnostic, start_capture, stop_capture, BirdnetLoadOptions,
    CaptureStartOptions,
};
use super::field_configuration::*;
use super::field_message_builders::build_detection_event;
use super::field_signals::{FieldSignals, LatestClassification};
use super::relay_session_controller::{local_peer_id, send_sealed_message};
use super::stage1_capture_pipeline::{ClassificationWindow, Stage1CapturePipeline};
use crate::mock::field_shim::{self, FieldDetectionReport};
use crate::state::AppState;

/// Node label for detections before this phone has a relay peer id.
const UNJOINED_NODE_LABEL: &str = "this-phone";
const BYTES_PER_MEGABYTE: f64 = 1.0e6;

struct CaptureRuntime {
    pipeline: Stage1CapturePipeline,
    resampler: WindowedSincResampler,
    next_detection_sequence: u64,
    app_state: AppState,
}

thread_local! {
    /// Page-wide, not per capture runtime: a stop/start while a window is in
    /// flight must not start a second concurrent `session.run` on the one
    /// ORT session.
    static CLASSIFIER_BUSY: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_RUNTIME: RefCell<Option<Rc<RefCell<CaptureRuntime>>>> = const { RefCell::new(None) };
    static CHUNK_CALLBACK: RefCell<Option<Closure<dyn FnMut(f64, js_sys::Float32Array)>>> = const { RefCell::new(None) };
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClassificationDiagnostic<'a> {
    common_name: &'a str,
    scientific_name: &'a str,
    score: f32,
    inference_milliseconds: f64,
    resample_milliseconds: f64,
    clip_bytes: Option<usize>,
    window_first_device_sample_index: u64,
}

pub async fn load_classifier(signals: FieldSignals) {
    signals.birdnet_status.set("Loading BirdNET V3 FP16 (pruned)".to_string());
    let options = BirdnetLoadOptions {
        model_url: BIRDNET_MODEL_URL.to_string(),
        labels_url: BIRDNET_LABELS_URL.to_string(),
        ort_wasm_base_url: ORT_WASM_BASE_URL.to_string(),
    };
    match load_birdnet(&options).await {
        Ok(report) => {
            let offline_note = if report.model_stored_for_offline { "" } else { " (not stored offline: storage full)" };
            signals.birdnet_status.set(format!(
                "Loaded: {:.0} MB, {} labels, session {:.0} ms, {} thread(s){offline_note}",
                report.model_bytes / BYTES_PER_MEGABYTE,
                report.label_count,
                report.session_create_milliseconds,
                report.intra_op_thread_count
            ));
            publish_diagnostic("birdnetLoad", &report);
            signals.birdnet_load_report.set(Some(report));
            pump_classifier(signals);
        }
        // Mic-only mode: capture and Stage-1 still run (plan: slow phones join as Mic-only).
        Err(error) => signals.birdnet_status.set(format!("Model unavailable (Mic-only): {error}")),
    }
}

pub async fn start_microphone(signals: FieldSignals, app_state: AppState) {
    if signals.capture_running.get_untracked() {
        return;
    }
    signals.capture_status.set("Requesting microphone".to_string());
    let runtime_slot: Rc<RefCell<Option<Rc<RefCell<CaptureRuntime>>>>> = Rc::new(RefCell::new(None));
    let callback_slot = runtime_slot.clone();
    let on_chunk = Closure::<dyn FnMut(f64, js_sys::Float32Array)>::new(move |first_frame: f64, samples: js_sys::Float32Array| {
        let Some(runtime) = callback_slot.borrow().clone() else { return };
        on_capture_chunk(&runtime, first_frame as u64, &samples.to_vec(), signals);
        pump_classifier(signals);
    });
    let options = CaptureStartOptions {
        requested_sample_rate_hz: REQUESTED_CAPTURE_SAMPLE_RATE_HZ,
        chunk_sample_count: WORKLET_CHUNK_SAMPLE_COUNT,
        worklet_url: WORKLET_MODULE_URL.to_string(),
    };
    match start_capture(&options, &on_chunk).await {
        Ok(settings) => {
            let sample_rate_hz = settings.context_sample_rate_hz.round() as u32;
            let runtime = Rc::new(RefCell::new(CaptureRuntime {
                pipeline: Stage1CapturePipeline::new(sample_rate_hz),
                resampler: WindowedSincResampler::new(sample_rate_hz, BIRDNET_INPUT_SAMPLE_RATE_HZ),
                next_detection_sequence: 0,
                app_state: app_state.clone(),
            }));
            *runtime_slot.borrow_mut() = Some(runtime.clone());
            CAPTURE_RUNTIME.with(|slot| *slot.borrow_mut() = Some(runtime));
            CHUNK_CALLBACK.with(|slot| *slot.borrow_mut() = Some(on_chunk));
            let clip_note = if sample_rate_hz == CLIP_CODEC_REQUIRED_SAMPLE_RATE_HZ { "" } else { " (clips off: not 48 kHz)" };
            signals.capture_status.set(format!("Capturing at {sample_rate_hz} Hz from \"{}\"{clip_note}", settings.device_label));
            publish_diagnostic("captureSettings", &serde_json::json!({
                "contextSampleRateHz": settings.context_sample_rate_hz,
                "trackSampleRateHz": settings.track_sample_rate_hz,
                "deviceLabel": settings.device_label,
                "echoCancellation": settings.echo_cancellation,
                "noiseSuppression": settings.noise_suppression,
                "autoGainControl": settings.auto_gain_control,
            }));
            signals.capture_settings.set(Some(settings));
            signals.capture_running.set(true);
        }
        Err(error) => signals.capture_status.set(format!("Microphone failed: {error}")),
    }
}

pub async fn stop_microphone(signals: FieldSignals) {
    let _ = stop_capture().await;
    CHUNK_CALLBACK.with(|slot| slot.borrow_mut().take());
    CAPTURE_RUNTIME.with(|slot| slot.borrow_mut().take());
    signals.capture_running.set(false);
    signals.capture_status.set("Microphone off".to_string());
}

/// Device-clock index just past the newest captured sample (heartbeats).
pub fn device_sample_index_now() -> u64 {
    CAPTURE_RUNTIME.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(0, |runtime| runtime.borrow().pipeline.next_expected_device_sample_index())
    })
}

fn on_capture_chunk(runtime: &Rc<RefCell<CaptureRuntime>>, first_frame: u64, samples: &[f32], signals: FieldSignals) {
    let mut runtime = runtime.borrow_mut();
    let outcome = runtime.pipeline.push_chunk(first_frame, samples);
    let stage1 = runtime.pipeline.stage1_counters();
    let counters = runtime.pipeline.counters();
    if outcome.trigger_opened_count > 0 {
        signals.stage1_trigger_count.set(stage1.triggers_opened);
        publish_diagnostic("stage1", &serde_json::json!({
            "triggersOpened": stage1.triggers_opened,
            "onsetsDetected": stage1.onsets_detected,
            "framesEvaluated": stage1.frames_evaluated,
            "lastTriggerContrastDb": outcome.last_trigger_contrast_db,
        }));
    }
    if outcome.onset_count > 0 {
        signals.stage1_onset_count.set(stage1.onsets_detected);
    }
    let trigger_open = runtime.pipeline.trigger_is_open();
    if signals.stage1_trigger_open.get_untracked() != trigger_open {
        signals.stage1_trigger_open.set(trigger_open);
    }
    if counters.capture_gaps != signals.capture_gap_count.get_untracked() {
        signals.capture_gap_count.set(counters.capture_gaps);
        publish_diagnostic("captureGaps", &serde_json::json!({
            "count": counters.capture_gaps,
            "lastExpectedIndex": counters.last_gap_expected_index,
            "lastReceivedIndex": counters.last_gap_received_index,
            "samplesReceived": counters.samples_received,
        }));
    }
    if counters.windows_dropped_classifier_busy != signals.windows_dropped_count.get_untracked() {
        signals.windows_dropped_count.set(counters.windows_dropped_classifier_busy);
    }
    signals.stage1_last_contrast_db.set(runtime.pipeline.last_contrast_db());
}

/// Starts one classification if the model is loaded and a window waits.
fn pump_classifier(signals: FieldSignals) {
    if signals.birdnet_load_report.get_untracked().is_none() {
        return;
    }
    let Some(runtime) = CAPTURE_RUNTIME.with(|slot| slot.borrow().clone()) else { return };
    if CLASSIFIER_BUSY.with(Cell::get) {
        return;
    }
    let Some(window) = runtime.borrow_mut().pipeline.take_ready_window() else { return };
    CLASSIFIER_BUSY.with(|busy| busy.set(true));
    spawn_local(async move {
        classify_window(&runtime, window, signals).await;
        CLASSIFIER_BUSY.with(|busy| busy.set(false));
        pump_classifier(signals);
    });
}

async fn classify_window(
    runtime: &Rc<RefCell<CaptureRuntime>>,
    window: ClassificationWindow,
    signals: FieldSignals,
) {
    let resample_started = js_sys::Date::now();
    let samples_32_khz = runtime.borrow().resampler.resample(&window.samples);
    let resample_milliseconds = js_sys::Date::now() - resample_started;
    let classification = match classify_with_birdnet(&samples_32_khz, BIRDNET_TOP_CLASS_COUNT).await {
        Ok(classification) => classification,
        Err(error) => {
            signals.birdnet_status.set(format!("Inference failed: {error}"));
            return;
        }
    };
    let Some(best) = classification.top.first().cloned() else { return };
    let clip_bytes = runtime
        .borrow()
        .pipeline
        .encode_clip_centred_in(&window)
        .and_then(|encoded| encoded.ok())
        .map(|bytes| bytes.len());
    signals.last_clip_bytes.set(clip_bytes);
    signals.classification_count.update(|count| *count += 1);
    publish_diagnostic("lastClassification", &ClassificationDiagnostic {
        common_name: &best.common_name,
        scientific_name: &best.scientific_name,
        score: best.score,
        inference_milliseconds: classification.inference_milliseconds,
        resample_milliseconds,
        clip_bytes,
        window_first_device_sample_index: window.first_device_sample_index,
    });
    signals.latest_classification.set(Some(LatestClassification {
        common_name: best.common_name.clone(),
        scientific_name: best.scientific_name.clone(),
        score: best.score,
        inference_milliseconds: classification.inference_milliseconds,
        window_first_device_sample_index: window.first_device_sample_index,
    }));
    if best.score < BIRDNET_REPORT_MINIMUM_SCORE {
        return;
    }
    let sequence = {
        let mut state = runtime.borrow_mut();
        state.next_detection_sequence += 1;
        state.next_detection_sequence
    };
    let peer_id = local_peer_id();
    let app_state = runtime.borrow().app_state.clone();
    {
        field_shim::publish_detection(
            &app_state,
            FieldDetectionReport {
                event_id: format!("local-{sequence}"),
                reporting_node_label: peer_id.map_or(UNJOINED_NODE_LABEL.to_string(), |id| format!("peer-{id}")),
                label: best.common_name.to_lowercase(),
                confidence: f64::from(best.score),
                has_clip: clip_bytes.is_some(),
            },
            LOCAL_DETECTION_FEED_CAPACITY,
        );
    }
    if let (Some(peer_id), true) = (peer_id, signals.relay_session_key_installed.get_untracked()) {
        let message = build_detection_event(peer_id, sequence, &window, &classification);
        if let Err(error) = send_sealed_message(&message) {
            signals.log_relay(format!("detection not sent: {error}"));
        }
    }
}
