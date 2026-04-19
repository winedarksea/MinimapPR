use crate::audio::detection_analysis::DetectionAudioAnalysisView;
use leptos::prelude::*;
use leptos_router::components::A;
use leptos_router::hooks::use_params_map;

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
                        <p class="muted">"Select a detection to inspect waveform, spectrogram, and classifier context."</p>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
                            <A href="/cop"><span class="btn-sm">"Open COP detections"</span></A>
                            <A href="/analysis/labels"><span class="btn-sm">"Open Analysis labels"</span></A>
                        </div>
                        <p class="muted" style="margin-top:8px">"Live stream mode will land in a later pass."</p>
                    </div>
                }.into_any(),
                Some(id) => {
                    view! {
                        <div style="display:flex;flex-direction:column;gap:8px;height:100%">
                            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
                                <A href="/cop">
                                    <span class="btn-sm">"← COP detections"</span>
                                </A>
                                <A href="/audio">
                                    <span class="btn-sm">"← Audio"</span>
                                </A>
                            </div>
                            <DetectionAudioAnalysisView
                                detection_id=id
                                show_expand_link=false
                                container_class="audio-page".to_string()
                                instance_prefix="page".to_string()
                            />
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
