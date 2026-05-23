use crate::audio::detection_analysis::DetectionAudioAnalysisView;
use crate::state::AppState;
use leptos::prelude::*;

#[component]
pub fn AudioAnalysisDrawer() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let is_open = state.audio_drawer_open;
    let selected_detection_id = state.audio_drawer_detection_id;
    let selected_track_id = state.audio_drawer_track_id;

    let close_drawer = move || {
        is_open.set(false);
    };

    view! {
        {move || {
            if !is_open.get() {
                return ().into_any();
            }

            let selected_id = selected_detection_id.get();
            let selected_track = selected_track_id.get();
            view! {
                <div class="audio-drawer-root" role="presentation">
                    <div class="audio-drawer-overlay" on:click=move |_| close_drawer()></div>
                    <aside
                        class="audio-drawer-panel"
                        role="dialog"
                        aria-label="Audio analysis drawer"
                        tabindex="0"
                        autofocus=true
                        on:keydown=move |ev: web_sys::KeyboardEvent| {
                            if ev.key() == "Escape" {
                                close_drawer();
                            }
                        }
                    >
                        <div class="audio-drawer-topbar">
                            <h3>"Audio analysis"</h3>
                            <button class="btn-sm" on:click=move |_| close_drawer()>"Close"</button>
                        </div>
                        {if let Some(id) = selected_id {
                            view! {
                                <DetectionAudioAnalysisView
                                    detection_id=id
                                    show_expand_link=true
                                    container_class="audio-drawer-content".to_string()
                                    instance_prefix="drawer".to_string()
                                />
                            }.into_any()
                        } else if let Some(track_id) = selected_track {
                            let audio_url = format!(
                                "/api/v1/tracks/{}/audio",
                                js_sys::encode_uri_component(&track_id),
                            );
                            let download_url = format!("{audio_url}?download=true");
                            view! {
                                <DetectionAudioAnalysisView
                                    detection_id=track_id
                                    show_expand_link=false
                                    container_class="audio-drawer-content".to_string()
                                    instance_prefix="drawer".to_string()
                                    audio_url_override=audio_url
                                    download_url_override=download_url
                                    load_detection_metadata=false
                                    analysis_title="Track".to_string()
                                />
                            }.into_any()
                        } else {
                            view! {
                                <div class="audio-drawer-empty muted">"No detection selected."</div>
                            }.into_any()
                        }}
                    </aside>
                </div>
            }.into_any()
        }}
    }
}
