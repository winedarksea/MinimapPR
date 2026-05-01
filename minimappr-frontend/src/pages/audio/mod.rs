pub mod recording;
pub mod recordings_library;

use crate::audio::detection_analysis::DetectionAudioAnalysisView;
use leptos::prelude::*;
use leptos_router::components::{Outlet, A};
use leptos_router::hooks::{use_location, use_params_map};

// ── Audio section layout with sub-nav ────────────────────────────

#[component]
pub fn AudioLayout() -> impl IntoView {
    let loc = use_location();

    // Record is active on /audio/record.
    // Analysis is active on /audio/analysis (landing) OR /audio/d/:id (detail).
    let is_record_active =
        move || loc.pathname.get().starts_with("/audio/record");
    let is_analysis_active = move || {
        let p = loc.pathname.get();
        p.starts_with("/audio/analysis") || p.starts_with("/audio/d")
    };

    view! {
        <div class="app-page">
            <nav class="subnav" aria-label="Audio sections">
                <A
                    href="/audio/record"
                    attr:class=move || if is_record_active() { "active" } else { "" }
                >
                    "Record"
                </A>
                <A
                    href="/audio/analysis"
                    attr:class=move || if is_analysis_active() { "active" } else { "" }
                >
                    "Analysis"
                </A>
            </nav>
            <div class="page-content">
                <Outlet />
            </div>
        </div>
    }
}

// ── Detection audio analysis page (kept at /audio/analysis and /audio/d/:id) ──

#[component]
pub fn AudioAnalysisPage() -> impl IntoView {
    let params = use_params_map();
    let id_sig: Signal<Option<String>> = Signal::derive(move || params.read().get("id"));

    view! {
        <div class="audio-page">
            {move || match id_sig.get() {
                None => view! {
                    <div class="page-stub">
                        <h2>"Audio analysis"</h2>
                        <p class="muted">
                            "Select a detection to inspect waveform, spectrogram, and classifier context."
                        </p>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
                            <A href="/cop"><span class="btn-sm">"Open COP detections"</span></A>
                            <A href="/analysis/labels">
                                <span class="btn-sm">"Open Analysis labels"</span>
                            </A>
                        </div>
                    </div>
                }.into_any(),
                Some(id) => view! {
                    <div style="display:flex;flex-direction:column;gap:8px;height:100%">
                        <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
                            <A href="/cop">
                                <span class="btn-sm">"← COP detections"</span>
                            </A>
                            <A href="/audio/record">
                                <span class="btn-sm">"← Recording"</span>
                            </A>
                        </div>
                        <DetectionAudioAnalysisView
                            detection_id=id
                            show_expand_link=false
                            container_class="audio-page".to_string()
                            instance_prefix="page".to_string()
                        />
                    </div>
                }.into_any(),
            }}
        </div>
    }
}
