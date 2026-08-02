use crate::map::bindings as lfi;
use crate::pages::analysis::AnalysisFilters;
use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Bin {
    lat: f64,
    lon: f64,
    weight: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct HeatmapResponse {
    window: String,
    bin_deg: f64,
    now_ns: i64,
    bins: Vec<Bin>,
}

#[component]
pub fn GeoHeatmapView() -> impl IntoView {
    let window = RwSignal::new("1h".to_string());
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let count = RwSignal::new(0usize);
    let classifier = use_context::<AnalysisFilters>()
        .expect("AnalysisFilters context provided by AnalysisLayout")
        .classifier;

    // Init the MapLibre map then invalidate size so flex layout dimensions settle.
    Effect::new(move |init_done: Option<bool>| {
        if init_done.is_none() {
            lfi::init(44.987, -93.258, 13);
            // Delay longer than the 100ms in init() to catch post-flex-layout sizing.
            spawn_local(async move {
                gloo_timers::future::TimeoutFuture::new(250).await;
                lfi::invalidate_map_size();
            });
        }
        true
    });

    let reload = {
        move |w: String, cls: Option<String>| {
            error.set(None);
            let mut url = format!("/api/v1/analytics/heatmap?window={w}&bin=0.0001&max_bins=5000");
            if let Some(c) = cls {
                url.push_str(&format!("&classifier={}", js_sys::encode_uri_component(&c)));
            }
            spawn_local(async move {
                match Request::get(&url).send().await {
                    Ok(resp) if resp.ok() => match resp.json::<HeatmapResponse>().await {
                        Ok(d) => {
                            lfi::clear_heatmap();
                            if !d.bins.is_empty() {
                                let max_weight =
                                    d.bins.iter().map(|b| b.weight).max().unwrap_or(1).max(1);
                                let points = js_sys::Array::new();
                                let fit_points = js_sys::Array::new();
                                for b in &d.bins {
                                    let triple = js_sys::Array::new();
                                    triple.push(&wasm_bindgen::JsValue::from_f64(b.lat));
                                    triple.push(&wasm_bindgen::JsValue::from_f64(b.lon));
                                    triple.push(&wasm_bindgen::JsValue::from_f64(b.weight as f64));
                                    points.push(&triple);
                                    let pair = js_sys::Array::new();
                                    pair.push(&wasm_bindgen::JsValue::from_f64(b.lat));
                                    pair.push(&wasm_bindgen::JsValue::from_f64(b.lon));
                                    fit_points.push(&pair);
                                }
                                lfi::set_heatmap_points(&points.into(), max_weight as f64);
                                lfi::fit_bounds_latlons(&fit_points.into());
                            }
                            lfi::invalidate_map_size();
                            count.set(d.bins.len());
                        }
                        Err(e) => error.set(Some(format!("parse: {e}"))),
                    },
                    Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                    Err(e) => error.set(Some(e.to_string())),
                }
            });
        }
    };

    Effect::new(move |_| {
        let w = window.get();
        let cls = classifier.get();
        reload(w, cls);
    });

    // Clean up heatmap when view unmounts to avoid leftover layers on COP map.
    on_cleanup(|| {
        lfi::clear_heatmap();
    });

    view! {
        <div class="heatmap-view">
            <div class="daily-toolbar">
                <WindowButton window=window label="1h" />
                <WindowButton window=window label="24h" />
                <WindowButton window=window label="7d" />
                <WindowButton window=window label="30d" />
                <WindowButton window=window label="365d" />
                <span class="daily-window-label">
                    {move || format!("{} bins", count.get())}
                </span>
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>
            <div class="leaflet-container-wrap" style="flex:1;min-height:400px;position:relative">
                <div id="mmp-map"></div>
                {move || if count.get() == 0 && error.get().is_none() {
                    Some(view! {
                        <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;z-index:500">
                            <span class="muted" style="background:var(--md-sys-color-surface-container);padding:8px 16px;border-radius:8px;font-size:13px">
                                "No geolocated detections in this window"
                            </span>
                        </div>
                    })
                } else {
                    None
                }}
            </div>
        </div>
    }
}

#[component]
fn WindowButton(window: RwSignal<String>, label: &'static str) -> impl IntoView {
    let is_active = move || window.get() == label;
    view! {
        <button
            class="btn-sm"
            class:active=is_active
            on:click=move |_| window.set(label.to_string())
        >{label}</button>
    }
}
