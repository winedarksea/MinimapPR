use crate::api::patch_config;
use crate::state::AppState;
use leptos::prelude::*;
use serde_json::json;
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Default)]
struct GroupState {
    dirty: bool,
    saving: bool,
    error: Option<String>,
    saved: bool,
    restart_required: Vec<String>,
}

#[component]
pub fn ConfigPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    view! {
        <div class="tab-pane">
            <div class="config-notice">
                "Saved settings persist across restarts (written to data/config.yml) and \
                 take precedence over environment variables."
            </div>
            {move || {
                let cfg = state.config.get();
                match cfg {
                    None => view! { <div class="empty-state">"Loading config…"</div> }.into_any(),
                    Some(c) => view! {
                        <AudioTriggerGroup
                            trigger_rms=c.trigger_rms
                            trigger_cooldown_seconds=c.trigger_cooldown_seconds
                            preprocess_enabled=c.preprocess_enabled
                            audio_highpass_hz=c.audio_highpass_hz
                            audio_lowpass_hz=c.audio_lowpass_hz
                        />
                        <LocalizationGroup
                            localization_window_seconds=c.localization_window_seconds
                            localization_algorithm=c.localization_algorithm.clone()
                            localization_strategy=c.localization_strategy.clone()
                            localization_band_min_hz=c.localization_band_min_hz
                            localization_band_max_hz=c.localization_band_max_hz
                        />
                        <ClassificationGroup
                            classification_audio_source=c.classification_audio_source.clone()
                            min_localization_confidence=c.min_localization_confidence
                            birdnet_enabled=c.birdnet_enabled
                            drone_head_enabled=c.drone_head_enabled
                            drone_head_min_confidence=c.drone_head_min_confidence
                            stt_enabled=c.stt_enabled
                            stt_trigger_min_confidence=c.stt_trigger_min_confidence
                            transcript_retention_seconds=c.transcript_retention_seconds
                            retention_yamnet_audio_seconds=c.retention_yamnet_audio_seconds
                            retention_birdnet_audio_seconds=c.retention_birdnet_audio_seconds
                            retention_drone_audio_seconds=c.retention_drone_audio_seconds
                            retention_alert_audio_seconds=c.retention_alert_audio_seconds
                            retention_detection_metadata_seconds=c.retention_detection_metadata_seconds
                            omni_scan_enabled=c.omni_scan_enabled
                            omni_scan_interval_seconds=c.omni_scan_interval_seconds
                            omni_scan_window_seconds=c.omni_scan_window_seconds
                            birdnet_chunked_dispatch_enabled=c.birdnet_chunked_dispatch_enabled
                            birdnet_trigger_min_confidence=c.birdnet_trigger_min_confidence
                            birdnet_geo_min_confidence=c.birdnet_geo_min_confidence
                            yamnet_min_confidence=c.yamnet_min_confidence
                            detection_min_confidence=c.detection_min_confidence
                            beamformer_type=c.beamformer_type.clone()
                        />
                        <TrackingFusionGroup
                            tracking_filter=c.tracking_filter.clone()
                            fusion_worker_count=c.fusion_worker_count
                            coordinate_mode=c.coordinate_mode.clone()
                        />
                    }.into_any(),
                }
            }}
        </div>
    }
}

// ── Input helpers ────────────────────────────────────────────────

fn num_input(
    label: &'static str,
    sig: RwSignal<String>,
    gs: RwSignal<GroupState>,
    min: f64,
    max: f64,
    step: f64,
) -> impl IntoView {
    view! {
        <div class="config-field">
            <label>{label}</label>
            <input
                type="number"
                min=min.to_string()
                max=max.to_string()
                step=step.to_string()
                prop:value=move || sig.get()
                on:input=move |ev| {
                    sig.set(event_target_value(&ev));
                    gs.update(|s| { s.dirty = true; s.saved = false; });
                }
            />
        </div>
    }
}

/// A retention control carries its data scope beside the number so operators
/// do not have to infer whether a value deletes audio or lightweight history.
fn retention_input(
    label: &'static str,
    description: &'static str,
    sig: RwSignal<String>,
    gs: RwSignal<GroupState>,
) -> impl IntoView {
    view! {
        <div class="retention-field">
            <label>
                <span class="retention-field-title">{label}</span>
                <span class="retention-field-description">{description}</span>
            </label>
            <div class="retention-input-wrap">
                <input
                    type="number"
                    min="1"
                    max="3650"
                    step="1"
                    prop:value=move || sig.get()
                    on:input=move |ev| {
                        sig.set(event_target_value(&ev));
                        gs.update(|s| { s.dirty = true; s.saved = false; });
                    }
                />
                <span aria-hidden="true">"days"</span>
            </div>
        </div>
    }
}

fn bool_input(label: &'static str, sig: RwSignal<bool>, gs: RwSignal<GroupState>) -> impl IntoView {
    view! {
        <div class="config-field">
            <label>{label}</label>
            <input
                type="checkbox"
                prop:checked=move || sig.get()
                on:change=move |ev| {
                    sig.set(event_target_checked(&ev));
                    gs.update(|s| { s.dirty = true; s.saved = false; });
                }
            />
        </div>
    }
}

fn select_input(
    label: &'static str,
    sig: RwSignal<String>,
    gs: RwSignal<GroupState>,
    options: &'static [&'static str],
) -> impl IntoView {
    view! {
        <div class="config-field">
            <label>{label}</label>
            <select
                on:change=move |ev| {
                    sig.set(event_target_value(&ev));
                    gs.update(|s| { s.dirty = true; s.saved = false; });
                }
            >
                {options.iter().map(|&opt| view! {
                    <option value=opt selected=move || sig.get() == opt>{opt}</option>
                }).collect_view()}
            </select>
        </div>
    }
}

/// Select input supporting per-option disabled state with a reason tooltip.
/// `options` is `(value, label, disabled, tooltip)`.
fn select_input_opts(
    label: &'static str,
    sig: RwSignal<String>,
    gs: RwSignal<GroupState>,
    options: Vec<(String, String, bool, String)>,
) -> impl IntoView {
    view! {
        <div class="config-field">
            <label>{label}</label>
            <select
                on:change=move |ev| {
                    sig.set(event_target_value(&ev));
                    gs.update(|s| { s.dirty = true; s.saved = false; });
                }
            >
                {options.into_iter().map(|(value, opt_label, disabled, tooltip)| {
                    let v = value.clone();
                    view! {
                        <option
                            value=value
                            disabled=disabled
                            title=tooltip
                            selected=move || sig.get() == v
                        >{opt_label}</option>
                    }
                }).collect_view()}
            </select>
        </div>
    }
}

fn group_footer(gs: RwSignal<GroupState>, on_save: impl Fn() + 'static) -> impl IntoView {
    view! {
        <div class="config-group-footer">
            <button
                class="btn-save"
                disabled=move || !gs.get().dirty || gs.get().saving
                on:click=move |_| on_save()
            >
                {move || if gs.get().saving { "Saving…" } else { "Save" }}
            </button>
            {move || gs.get().error.as_ref().map(|e| view! { <span class="save-error">{e.clone()}</span> })}
            {move || if gs.get().saved && !gs.get().dirty {
                Some(view! { <span class="save-ok">"Saved ✓"</span> })
            } else { None }}
            {move || {
                let rr = gs.get().restart_required;
                (!rr.is_empty()).then(|| view! {
                    <span class="muted">{format!("Sidecar restart required: {}", rr.join(", "))}</span>
                })
            }}
        </div>
    }
}

// ── Group: Audio & Trigger ───────────────────────────────────────

#[component]
fn AudioTriggerGroup(
    trigger_rms: f64,
    trigger_cooldown_seconds: f64,
    preprocess_enabled: bool,
    audio_highpass_hz: f64,
    audio_lowpass_hz: f64,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let rms = RwSignal::new(trigger_rms.to_string());
    let cooldown = RwSignal::new(trigger_cooldown_seconds.to_string());
    let preprocess = RwSignal::new(preprocess_enabled);
    let highpass = RwSignal::new(audio_highpass_hz.to_string());
    let lowpass = RwSignal::new(audio_lowpass_hz.to_string());
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "trigger_rms": rms.get().parse::<f64>().unwrap_or(0.015),
            "trigger_cooldown_seconds": cooldown.get().parse::<f64>().unwrap_or(0.8),
            "preprocess_enabled": preprocess.get(),
            "audio_highpass_hz": highpass.get().parse::<f64>().unwrap_or(50.0),
            "audio_lowpass_hz": lowpass.get().parse::<f64>().unwrap_or(0.0),
        });
        gs.update(|s| {
            s.saving = true;
            s.error = None;
        });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    let restart_required = cfg.restart_required.clone();
                    s.config.set(Some(cfg));
                    gs.update(|st| {
                        st.saving = false;
                        st.dirty = false;
                        st.saved = true;
                        st.restart_required = restart_required;
                    });
                }
                Err(e) => {
                    gs.update(|st| {
                        st.saving = false;
                        st.error = Some(e);
                    });
                }
            }
        });
    };

    view! {
        <div class="config-group">
            <div class="config-group-header">
                "Audio & Trigger"
                <span class="dirty-dot" class:visible=dirty></span>
            </div>
            <div class="config-fields">
                {num_input("Trigger RMS", rms, gs, 0.0001, 1.0, 0.001)}
                {num_input("Cooldown (s)", cooldown, gs, 0.0, 60.0, 0.1)}
                {bool_input("Preprocess Enabled", preprocess, gs)}
                {num_input("Highpass (Hz)", highpass, gs, 0.0, 20000.0, 10.0)}
                {num_input("Lowpass (Hz)", lowpass, gs, 0.0, 20000.0, 10.0)}
            </div>
            {group_footer(gs, save)}
        </div>
    }
}

// ── Group: Localization ──────────────────────────────────────────

#[component]
fn LocalizationGroup(
    localization_window_seconds: f64,
    localization_algorithm: String,
    localization_strategy: String,
    localization_band_min_hz: f64,
    localization_band_max_hz: f64,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let window = RwSignal::new(localization_window_seconds.to_string());
    let algo = RwSignal::new(localization_algorithm);
    let strat = RwSignal::new(localization_strategy);
    let band_min = RwSignal::new(localization_band_min_hz.to_string());
    let band_max = RwSignal::new(localization_band_max_hz.to_string());
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "localization_window_seconds": window.get().parse::<f64>().unwrap_or(0.08),
            "localization_algorithm": algo.get(),
            "localization_strategy": strat.get(),
            "localization_band_min_hz": band_min.get().parse::<f64>().unwrap_or(0.0),
            "localization_band_max_hz": band_max.get().parse::<f64>().unwrap_or(0.0),
        });
        gs.update(|s| {
            s.saving = true;
            s.error = None;
        });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    let restart_required = cfg.restart_required.clone();
                    s.config.set(Some(cfg));
                    gs.update(|st| {
                        st.saving = false;
                        st.dirty = false;
                        st.saved = true;
                        st.restart_required = restart_required;
                    });
                }
                Err(e) => {
                    gs.update(|st| {
                        st.saving = false;
                        st.error = Some(e);
                    });
                }
            }
        });
    };

    view! {
        <div class="config-group">
            <div class="config-group-header">
                "Localization"
                <span class="dirty-dot" class:visible=dirty></span>
            </div>
            <div class="config-fields">
                {num_input("Window (s)", window, gs, 0.01, 2.0, 0.01)}
                {select_input("Algorithm", algo, gs, &["gcc_phat", "srp_phat", "music", "esprit"])}
                {select_input("Strategy", strat, gs, &["fixed", "geometry_aware", "cascade"])}
                {num_input("Band Min (Hz)", band_min, gs, 0.0, 20000.0, 10.0)}
                {num_input("Band Max (Hz)", band_max, gs, 0.0, 20000.0, 10.0)}
            </div>
            {group_footer(gs, save)}
        </div>
    }
}

// ── Group: Classification ────────────────────────────────────────

#[component]
fn ClassificationGroup(
    classification_audio_source: String,
    min_localization_confidence: f64,
    birdnet_enabled: bool,
    drone_head_enabled: bool,
    drone_head_min_confidence: f64,
    stt_enabled: bool,
    stt_trigger_min_confidence: f64,
    transcript_retention_seconds: f64,
    retention_yamnet_audio_seconds: f64,
    retention_birdnet_audio_seconds: f64,
    retention_drone_audio_seconds: f64,
    retention_alert_audio_seconds: f64,
    retention_detection_metadata_seconds: f64,
    omni_scan_enabled: bool,
    omni_scan_interval_seconds: f64,
    omni_scan_window_seconds: f64,
    birdnet_chunked_dispatch_enabled: bool,
    birdnet_trigger_min_confidence: f64,
    birdnet_geo_min_confidence: f64,
    yamnet_min_confidence: f64,
    detection_min_confidence: f64,
    beamformer_type: String,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let audio_source = RwSignal::new(classification_audio_source);
    let min_loc_conf = RwSignal::new(min_localization_confidence.to_string());
    let birdnet = RwSignal::new(birdnet_enabled);
    let drone_head = RwSignal::new(drone_head_enabled);
    let drone_confidence = RwSignal::new(drone_head_min_confidence.to_string());
    let stt = RwSignal::new(stt_enabled);
    let stt_confidence = RwSignal::new(stt_trigger_min_confidence.to_string());
    let transcript_retention_days = RwSignal::new((transcript_retention_seconds / 86_400.0).to_string());
    let yamnet_retention_days = RwSignal::new((retention_yamnet_audio_seconds / 86_400.0).to_string());
    let birdnet_retention_days = RwSignal::new((retention_birdnet_audio_seconds / 86_400.0).to_string());
    let drone_retention_days = RwSignal::new((retention_drone_audio_seconds / 86_400.0).to_string());
    let alert_retention_days = RwSignal::new((retention_alert_audio_seconds / 86_400.0).to_string());
    let metadata_retention_days = RwSignal::new((retention_detection_metadata_seconds / 86_400.0).to_string());
    let omni_scan = RwSignal::new(omni_scan_enabled);
    let omni_interval = RwSignal::new(omni_scan_interval_seconds.to_string());
    let omni_window = RwSignal::new(omni_scan_window_seconds.to_string());
    let chunked = RwSignal::new(birdnet_chunked_dispatch_enabled);
    let birdnet_trigger = RwSignal::new(birdnet_trigger_min_confidence.to_string());
    let birdnet_geo = RwSignal::new(birdnet_geo_min_confidence.to_string());
    let yamnet = RwSignal::new(yamnet_min_confidence.to_string());
    let detection = RwSignal::new(detection_min_confidence.to_string());
    let beamform = RwSignal::new(beamformer_type);
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "classification_audio_source": audio_source.get(),
            "min_localization_confidence": min_loc_conf.get().parse::<f64>().unwrap_or(0.20),
            "birdnet_enabled": birdnet.get(),
            "drone_head_enabled": drone_head.get(),
            "drone_head_min_confidence": drone_confidence.get().parse::<f64>().unwrap_or(0.50),
            "stt_enabled": stt.get(),
            "stt_trigger_min_confidence": stt_confidence.get().parse::<f64>().unwrap_or(0.50),
            "transcript_retention_seconds": transcript_retention_days.get().parse::<f64>().unwrap_or(7.0) * 86_400.0,
            "retention_yamnet_audio_seconds": yamnet_retention_days.get().parse::<f64>().unwrap_or(3.0) * 86_400.0,
            "retention_birdnet_audio_seconds": birdnet_retention_days.get().parse::<f64>().unwrap_or(30.0) * 86_400.0,
            "retention_drone_audio_seconds": drone_retention_days.get().parse::<f64>().unwrap_or(30.0) * 86_400.0,
            "retention_alert_audio_seconds": alert_retention_days.get().parse::<f64>().unwrap_or(30.0) * 86_400.0,
            "retention_detection_metadata_seconds": metadata_retention_days.get().parse::<f64>().unwrap_or(730.0) * 86_400.0,
            "omni_scan_enabled": omni_scan.get(),
            "omni_scan_interval_seconds": omni_interval.get().parse::<f64>().unwrap_or(30.0),
            "omni_scan_window_seconds": omni_window.get().parse::<f64>().unwrap_or(15.0),
            "birdnet_chunked_dispatch_enabled": chunked.get(),
            "birdnet_trigger_min_confidence": birdnet_trigger.get().parse::<f64>().unwrap_or(0.40),
            "birdnet_geo_min_confidence": birdnet_geo.get().parse::<f64>().unwrap_or(0.03),
            "yamnet_min_confidence": yamnet.get().parse::<f64>().unwrap_or(0.25),
            "detection_min_confidence": detection.get().parse::<f64>().unwrap_or(0.05),
            "beamformer_type": beamform.get(),
        });
        gs.update(|s| {
            s.saving = true;
            s.error = None;
        });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    let restart_required = cfg.restart_required.clone();
                    s.config.set(Some(cfg));
                    gs.update(|st| {
                        st.saving = false;
                        st.dirty = false;
                        st.saved = true;
                        st.restart_required = restart_required;
                    });
                }
                Err(e) => {
                    gs.update(|st| {
                        st.saving = false;
                        st.error = Some(e);
                    });
                }
            }
        });
    };

    view! {
        <div class="config-group">
            <div class="config-group-header">
                "Classifier Services"
                <span class="dirty-dot" class:visible=dirty></span>
            </div>
            <div class="config-fields">
                <div class="config-routing-note">
                    <strong>"YAMNet is always active."</strong>
                    " Use the service switches below for common controls. Advanced per-context routing is managed through the classifier-routing API and requires restart."
                </div>
                <div class="config-subsection">"Detection models"</div>
                {bool_input("Run BirdNET", birdnet, gs)}
                {bool_input("Run drone detector", drone_head, gs)}
                {num_input("Drone alert confidence", drone_confidence, gs, 0.0, 1.0, 0.01)}
                <div class="config-subsection">"Speech capture"</div>
                {bool_input("Transcribe YAMNet speech", stt, gs)}
                {num_input("Speech trigger confidence", stt_confidence, gs, 0.0, 1.0, 0.01)}
                {num_input("Transcript retention (days)", transcript_retention_days, gs, 1.0, 365.0, 1.0)}
                <div class="config-subsection">"Audio retention"</div>
                <div class="retention-settings" aria-label="Audio and detection retention">
                    {retention_input("YAMNet detections", "Detection audio only", yamnet_retention_days, gs)}
                    {retention_input("BirdNET detections", "Detection audio only", birdnet_retention_days, gs)}
                    {retention_input("Positive drone detections", "Negative drone results are never written", drone_retention_days, gs)}
                    {retention_input("Alerted audio", "Overrides classifier and label retention", alert_retention_days, gs)}
                    {retention_input("Detection metadata", "Minimal what / where / when history", metadata_retention_days, gs)}
                </div>
                <div class="retention-protection-note">
                    <strong>"Always kept: "</strong>
                    "IAMF and calibration recordings remain until explicitly deleted or included in a full purge. Label-specific overrides are managed in the retention policy."
                </div>
                <div class="config-subsection">"Continuous omni scan"</div>
                {bool_input("Scan omni nodes continuously", omni_scan, gs)}
                {num_input("Scan interval (s)", omni_interval, gs, 1.0, 3600.0, 1.0)}
                {num_input("Audio window (s)", omni_window, gs, 0.5, 300.0, 0.5)}
                <div class="config-subsection">"Audio and model thresholds"</div>
                {select_input_opts("Audio Source", audio_source, gs, vec![
                    ("beamformed".to_string(), "beamformed".to_string(), false, String::new()),
                    ("omni".to_string(), "omni".to_string(), false, String::new()),
                    ("nearest_node_omni".to_string(), "nearest_node_omni".to_string(), false, String::new()),
                ])}
                {num_input("Min Localization Confidence", min_loc_conf, gs, 0.0, 1.0, 0.01)}
                {bool_input("BirdNET Chunked Dispatch", chunked, gs)}
                {num_input("BirdNET Trigger Confidence", birdnet_trigger, gs, 0.0, 1.0, 0.01)}
                {num_input("BirdNET Geo Confidence", birdnet_geo, gs, 0.0, 1.0, 0.01)}
                {num_input("YAMNet Min Confidence", yamnet, gs, 0.0, 1.0, 0.01)}
                {num_input("Detection Min Confidence", detection, gs, 0.0, 1.0, 0.01)}
                {select_input("Beamformer", beamform, gs, &["delay_and_sum", "freq_domain_das", "mvdr", "superdirective", "gevd"])}
            </div>
            {group_footer(gs, save)}
        </div>
    }
}

// ── Group: Tracking & Fusion ─────────────────────────────────────

#[component]
fn TrackingFusionGroup(
    tracking_filter: String,
    fusion_worker_count: u32,
    coordinate_mode: String,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let filter = RwSignal::new(tracking_filter);
    let workers = RwSignal::new(fusion_worker_count.to_string());
    let coord = RwSignal::new(coordinate_mode);
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "tracking_filter": filter.get(),
            "fusion_worker_count": workers.get().parse::<u64>().unwrap_or(1),
            "coordinate_mode": coord.get(),
        });
        gs.update(|s| {
            s.saving = true;
            s.error = None;
        });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    let restart_required = cfg.restart_required.clone();
                    s.config.set(Some(cfg));
                    gs.update(|st| {
                        st.saving = false;
                        st.dirty = false;
                        st.saved = true;
                        st.restart_required = restart_required;
                    });
                }
                Err(e) => {
                    gs.update(|st| {
                        st.saving = false;
                        st.error = Some(e);
                    });
                }
            }
        });
    };

    view! {
        <div class="config-group">
            <div class="config-group-header">
                "Tracking & Fusion"
                <span class="dirty-dot" class:visible=dirty></span>
            </div>
            <div class="config-fields">
                {select_input("Tracking Filter", filter, gs, &["linear", "kalman"])}
                {num_input("Fusion Workers", workers, gs, 1.0, 16.0, 1.0)}
                {select_input("Coordinate Mode", coord, gs, &["flat", "geodetic"])}
            </div>
            {group_footer(gs, save)}
        </div>
    }
}
