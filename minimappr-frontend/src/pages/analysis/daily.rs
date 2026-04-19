use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct DailyResponse {
    mode: String,
    tz: String,
    hours: usize,
    #[serde(default)]
    start: String,
    #[serde(default)]
    end: String,
    bucket_starts: Vec<String>,
    labels: Vec<String>,
    counts: Vec<Vec<u32>>,
    label_totals: Vec<u32>,
    bucket_totals: Vec<u32>,
}

/// Rolling-24h detection matrix. X = 24 hour buckets ending at the current hour,
/// Y = detection labels. Rendered as a CSS Grid for accessibility + trivial layout.
#[component]
pub fn DailyDetectionsView() -> impl IntoView {
    // Offset in days from "now rolling 24h" — 0 = today rolling, 1 = yesterday's 24h, etc.
    let day_offset = RwSignal::new(0i64);
    let data: RwSignal<Option<DailyResponse>> = RwSignal::new(None);
    let loading = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    // Use browser's detected tz (falls back to UTC if Intl is unavailable).
    let tz = detect_tz();

    let fetch_data = {
        let tz = tz.clone();
        move |offset: i64| {
            loading.set(true);
            error.set(None);
            let end_iso = iso_hour_offset(-offset * 24);
            let url = format!(
                "/api/v1/analytics/daily?end={}&hours=24&tz={}",
                js_sys::encode_uri_component(&end_iso),
                js_sys::encode_uri_component(&tz),
            );
            spawn_local(async move {
                match Request::get(&url).send().await {
                    Ok(resp) if resp.ok() => match resp.json::<DailyResponse>().await {
                        Ok(d) => {
                            data.set(Some(d));
                            error.set(None);
                        }
                        Err(e) => error.set(Some(format!("parse: {e}"))),
                    },
                    Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                    Err(e) => error.set(Some(e.to_string())),
                }
                loading.set(false);
            });
        }
    };

    // Initial fetch + refetch when offset changes.
    Effect::new(move |_| {
        let offset = day_offset.get();
        fetch_data(offset);
    });

    let on_prev = move |_| day_offset.update(|v| *v += 1);
    let on_today = move |_| day_offset.set(0);
    let on_next = move |_| {
        day_offset.update(|v| {
            if *v > 0 {
                *v -= 1
            }
        })
    };

    view! {
        <div class="daily-view">
            <div class="daily-toolbar">
                <button class="btn-sm" on:click=on_prev title="Shift window back 24h">
                    "◀ Prev day"
                </button>
                <button
                    class="btn-sm"
                    on:click=on_today
                    disabled=move || day_offset.get() == 0
                    title="Rolling 24 hours ending now"
                >
                    "Today (rolling 24h)"
                </button>
                <button
                    class="btn-sm"
                    on:click=on_next
                    disabled=move || day_offset.get() == 0
                    title="Shift window forward 24h"
                >
                    "Next day ▶"
                </button>
                <span class="daily-window-label">
                    {move || match data.get() {
                        Some(d) => format!("{} · tz {}", d.start, d.tz),
                        None    => "—".into(),
                    }}
                </span>
                <span class="daily-loading" class:visible=move || loading.get()>"loading…"</span>
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>

            {move || {
                let Some(d) = data.get() else {
                    return view! { <div class="empty-state">"No data yet"</div> }.into_any();
                };
                if d.labels.is_empty() {
                    return view! {
                        <div class="empty-state">"No detections in this 24h window"</div>
                    }.into_any();
                }
                view! { <DailyMatrix data=d /> }.into_any()
            }}
        </div>
    }
}

#[component]
fn DailyMatrix(data: DailyResponse) -> impl IntoView {
    let num_buckets = data.hours;
    let max_bucket_total = data.bucket_totals.iter().copied().max().unwrap_or(1).max(1);
    let max_cell = data
        .counts
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let hour_labels: Vec<String> = data
        .bucket_starts
        .iter()
        .map(|iso| extract_hour_label(iso))
        .collect();

    let grid_cols = format!("repeat({}, minmax(0, 1fr))", num_buckets);
    let counts = data.counts.clone();
    let labels = data.labels.clone();
    let label_totals = data.label_totals.clone();
    let bucket_totals = data.bucket_totals.clone();

    view! {
        <div class="daily-card card">
            // Histogram row — aligned to the same grid as the matrix.
            <div class="daily-grid-wrap">
                <div class="daily-grid-row daily-histogram" style:grid-template-columns=grid_cols.clone()>
                    {bucket_totals.iter().enumerate().map(|(i, total)| {
                        let t = *total;
                        let frac = (t as f64) / (max_bucket_total as f64);
                        let h = (frac * 56.0).max(if t > 0 { 2.0 } else { 0.0 });
                        let title = format!("{}: {} detections", hour_labels[i], t);
                        view! {
                            <div class="hist-cell" title=title>
                                <div class="hist-bar" style:height=format!("{:.0}px", h)></div>
                            </div>
                        }
                    }).collect_view()}
                </div>

                // Hour axis row.
                <div class="daily-grid-row daily-hour-axis" style:grid-template-columns=grid_cols.clone()>
                    {hour_labels.iter().map(|h| view! {
                        <span class="hour-tick">{h.clone()}</span>
                    }).collect_view()}
                </div>

                // Matrix rows (one per label).
                {labels.iter().enumerate().map(|(row, label)| {
                    let label = label.clone();
                    let lbl_total = label_totals.get(row).copied().unwrap_or(0);
                    let row_data = counts.get(row).cloned().unwrap_or_default();
                    let cols_fmt = format!("120px {}", grid_cols);
                    view! {
                        <div class="daily-grid-row daily-matrix-row" style:grid-template-columns=cols_fmt>
                            <div class="daily-label-cell">
                                <span class="daily-label">{label.clone()}</span>
                                <span class="daily-label-total mono">{lbl_total}</span>
                            </div>
                            {row_data.iter().enumerate().map(|(col, count)| {
                                let c = *count;
                                let intensity = if max_cell > 0 { (c as f64 / max_cell as f64).clamp(0.0, 1.0) } else { 0.0 };
                                let alpha = if c == 0 { 0.0 } else { (0.18 + intensity * 0.82).min(1.0) };
                                let hour = hour_labels.get(col).cloned().unwrap_or_default();
                                let lbl = label.clone();
                                let aria = format!("label {lbl} hour {hour} count {c}");
                                let title = format!("{lbl} · {hour} · {c}");
                                view! {
                                    <button
                                        class="daily-cell"
                                        class:zero=move || c == 0
                                        aria-label=aria
                                        title=title
                                        style:--cell-alpha=format!("{:.3}", alpha)
                                    >
                                        {if c > 0 { Some(view! { <span class="cell-count mono">{c}</span> }) } else { None }}
                                    </button>
                                }
                            }).collect_view()}
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

/// Hour label from ISO timestamp like "2026-04-18T09:00:00+00:00" → "09".
fn extract_hour_label(iso: &str) -> String {
    iso.split('T')
        .nth(1)
        .and_then(|t| t.get(..2))
        .unwrap_or("--")
        .to_string()
}

/// Return ISO-8601 UTC timestamp of "now + offset_hours", e.g. "2026-04-18T12:34:56Z".
fn iso_hour_offset(offset_hours: i64) -> String {
    let now_ms = js_sys::Date::now();
    let ms = now_ms + (offset_hours as f64) * 3_600_000.0;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    d.to_iso_string().as_string().unwrap_or_default()
}

/// Detect browser IANA timezone via Intl; fall back to UTC.
fn detect_tz() -> String {
    let win = web_sys::window();
    let intl = win
        .as_ref()
        .and_then(|w| js_sys::Reflect::get(w, &"Intl".into()).ok());
    if let Some(intl) = intl {
        let dtf = js_sys::Reflect::get(&intl, &"DateTimeFormat".into()).ok();
        if let Some(dtf_ctor) = dtf {
            if let Ok(fn_v) = dtf_ctor.dyn_into::<js_sys::Function>() {
                if let Ok(instance) = js_sys::Reflect::construct(&fn_v, &js_sys::Array::new()) {
                    if let Ok(opts_fn_v) =
                        js_sys::Reflect::get(&instance, &"resolvedOptions".into())
                    {
                        if let Ok(opts_fn) = opts_fn_v.dyn_into::<js_sys::Function>() {
                            if let Ok(opts) = opts_fn.call0(&instance) {
                                if let Ok(v) = js_sys::Reflect::get(&opts, &"timeZone".into()) {
                                    if let Some(s) = v.as_string() {
                                        return s;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    "UTC".into()
}

use wasm_bindgen::JsCast;
