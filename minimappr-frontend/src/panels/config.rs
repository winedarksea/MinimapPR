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
}

#[component]
pub fn ConfigPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    view! {
        <div class="tab-pane">
            <div class="config-notice">
                "Settings are in-process only — changes revert on server restart."
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
                        />
                        <ClassificationGroup
                            classifier_backend=c.classifier_backend.clone()
                            yamnet_min_confidence=c.yamnet_min_confidence
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
                prop:value=move || sig.get()
                on:change=move |ev| {
                    sig.set(event_target_value(&ev));
                    gs.update(|s| { s.dirty = true; s.saved = false; });
                }
            >
                {options.iter().map(|&opt| view! {
                    <option value=opt>{opt}</option>
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
    let rms       = RwSignal::new(trigger_rms.to_string());
    let cooldown  = RwSignal::new(trigger_cooldown_seconds.to_string());
    let preprocess = RwSignal::new(preprocess_enabled);
    let highpass  = RwSignal::new(audio_highpass_hz.to_string());
    let lowpass   = RwSignal::new(audio_lowpass_hz.to_string());
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
        gs.update(|s| { s.saving = true; s.error = None; });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    s.config.set(Some(cfg));
                    gs.update(|st| { st.saving = false; st.dirty = false; st.saved = true; });
                }
                Err(e) => {
                    gs.update(|st| { st.saving = false; st.error = Some(e); });
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
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let window = RwSignal::new(localization_window_seconds.to_string());
    let algo   = RwSignal::new(localization_algorithm);
    let strat  = RwSignal::new(localization_strategy);
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "localization_window_seconds": window.get().parse::<f64>().unwrap_or(0.08),
            "localization_algorithm": algo.get(),
            "localization_strategy": strat.get(),
        });
        gs.update(|s| { s.saving = true; s.error = None; });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    s.config.set(Some(cfg));
                    gs.update(|st| { st.saving = false; st.dirty = false; st.saved = true; });
                }
                Err(e) => {
                    gs.update(|st| { st.saving = false; st.error = Some(e); });
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
            </div>
            {group_footer(gs, save)}
        </div>
    }
}

// ── Group: Classification ────────────────────────────────────────

#[component]
fn ClassificationGroup(
    classifier_backend: String,
    yamnet_min_confidence: f64,
    beamformer_type: String,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let backend  = RwSignal::new(classifier_backend);
    let yamnet   = RwSignal::new(yamnet_min_confidence.to_string());
    let beamform = RwSignal::new(beamformer_type);
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "classifier_backend": backend.get(),
            "yamnet_min_confidence": yamnet.get().parse::<f64>().unwrap_or(0.25),
            "beamformer_type": beamform.get(),
        });
        gs.update(|s| { s.saving = true; s.error = None; });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    s.config.set(Some(cfg));
                    gs.update(|st| { st.saving = false; st.dirty = false; st.saved = true; });
                }
                Err(e) => {
                    gs.update(|st| { st.saving = false; st.error = Some(e); });
                }
            }
        });
    };

    view! {
        <div class="config-group">
            <div class="config-group-header">
                "Classification"
                <span class="dirty-dot" class:visible=dirty></span>
            </div>
            <div class="config-fields">
                {select_input("Backend", backend, gs, &["yamnet", "birdnet", "heuristic"])}
                {num_input("YAMNet Min Confidence", yamnet, gs, 0.0, 1.0, 0.01)}
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
    let filter  = RwSignal::new(tracking_filter);
    let workers = RwSignal::new(fusion_worker_count.to_string());
    let coord   = RwSignal::new(coordinate_mode);
    let gs = RwSignal::new(GroupState::default());

    let dirty = move || gs.get().dirty;

    let save = move || {
        let body = json!({
            "tracking_filter": filter.get(),
            "fusion_worker_count": workers.get().parse::<u64>().unwrap_or(1),
            "coordinate_mode": coord.get(),
        });
        gs.update(|s| { s.saving = true; s.error = None; });
        let s = state.clone();
        spawn_local(async move {
            match patch_config(body).await {
                Ok(cfg) => {
                    s.config.set(Some(cfg));
                    gs.update(|st| { st.saving = false; st.dirty = false; st.saved = true; });
                }
                Err(e) => {
                    gs.update(|st| { st.saving = false; st.error = Some(e); });
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
