use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;
use leptos_router::hooks::use_params_map;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LabelSummary {
    label: String,
    count: u32,
    first_seen_ns: Option<i64>,
    last_seen_ns: Option<i64>,
    node_count: Option<u32>,
    category: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LabelsResponse {
    window: String,
    now_ns: i64,
    labels: Vec<LabelSummary>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
struct LabelDetailResponse {
    label: String,
    #[serde(default)]
    total: u32,
    first_seen_ns: Option<i64>,
    last_seen_ns: Option<i64>,
    #[serde(default)]
    hour_histogram: Vec<u32>,
    #[serde(default)]
    dow_histogram: Vec<u32>,
    #[serde(default)]
    month_histogram: Vec<u32>,
    #[serde(default)]
    recent: Vec<crate::state::Detection>,
}

#[component]
pub fn LabelSummaryView() -> impl IntoView {
    let data: RwSignal<Option<LabelsResponse>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let window = RwSignal::new("30d".to_string());

    let reload = move |w: String| {
        let url = format!("/api/v1/analytics/labels?window={}", w);
        spawn_local(async move {
            match Request::get(&url).send().await {
                Ok(resp) if resp.ok() => match resp.json::<LabelsResponse>().await {
                    Ok(d) => data.set(Some(d)),
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    };

    Effect::new(move |_| {
        let w = window.get();
        reload(w);
    });

    view! {
        <div class="labels-view">
            <div class="daily-toolbar">
                <WindowButton window=window label="24h" />
                <WindowButton window=window label="7d" />
                <WindowButton window=window label="30d" />
                <WindowButton window=window label="365d" />
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>

            {move || {
                let Some(d) = data.get() else {
                    return view! { <div class="empty-state">"Loading…"</div> }.into_any();
                };
                if d.labels.is_empty() {
                    return view! { <div class="empty-state">"No detections in window"</div> }.into_any();
                }
                let max_count = d.labels.iter().map(|l| l.count).max().unwrap_or(1).max(1);
                view! {
                    <table class="data-table labels-table">
                        <thead>
                            <tr>
                                <th>"Label"</th>
                                <th>"Category"</th>
                                <th style="text-align:right">"Count"</th>
                                <th style="width:160px">"Share"</th>
                                <th>"Nodes"</th>
                                <th>"Last seen"</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {d.labels.into_iter().map(|row| {
                                let frac = (row.count as f64) / (max_count as f64);
                                let width = format!("{}%", (frac * 100.0).round() as u32);
                                let last_age = row.last_seen_ns.map(format_age_ns).unwrap_or_else(|| "—".into());
                                let href = format!("/analysis/labels/{}", row.label);
                                view! {
                                    <tr>
                                        <td><strong>{row.label.clone()}</strong></td>
                                        <td>{row.category.clone().unwrap_or_else(|| "—".into())}</td>
                                        <td class="mono" style="text-align:right">{row.count}</td>
                                        <td>
                                            <span class="tqi-bar" style:width=width></span>
                                        </td>
                                        <td class="mono">{row.node_count.unwrap_or(0)}</td>
                                        <td class="mono">{last_age}</td>
                                        <td>
                                            <A href=href>
                                                <span class="btn-sm">"Detail →"</span>
                                            </A>
                                        </td>
                                    </tr>
                                }
                            }).collect_view()}
                        </tbody>
                    </table>
                }.into_any()
            }}
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

#[component]
pub fn LabelDetailView() -> impl IntoView {
    let params = use_params_map();
    let label = move || params.read().get("id").unwrap_or_default();
    let data: RwSignal<Option<LabelDetailResponse>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    Effect::new(move |_| {
        let lbl = label();
        if lbl.is_empty() { return; }
        let url = format!(
            "/api/v1/analytics/labels/{}?window=30d&recent_limit=50",
            js_sys::encode_uri_component(&lbl),
        );
        spawn_local(async move {
            match Request::get(&url).send().await {
                Ok(resp) if resp.ok() => match resp.json::<LabelDetailResponse>().await {
                    Ok(d) => data.set(Some(d)),
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    });

    view! {
        <div class="labels-view">
            <div class="daily-toolbar">
                <A href="/analysis/labels"><span class="btn-sm">"← All labels"</span></A>
                <span class="daily-window-label">{label}</span>
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>
            {move || {
                let Some(d) = data.get() else {
                    return view! { <div class="empty-state">"Loading…"</div> }.into_any();
                };
                if d.total == 0 {
                    return view! { <div class="empty-state">"No detections for this label in the last 30 days"</div> }.into_any();
                }
                view! { <LabelDetailBody data=d /> }.into_any()
            }}
        </div>
    }
}

#[component]
fn LabelDetailBody(data: LabelDetailResponse) -> impl IntoView {
    let last_seen = data.last_seen_ns.map(format_age_ns).unwrap_or_else(|| "—".into());
    let first_seen = data.first_seen_ns.map(format_age_ns).unwrap_or_else(|| "—".into());
    let hour_max = *data.hour_histogram.iter().max().unwrap_or(&1).max(&1);
    let dow_max  = *data.dow_histogram.iter().max().unwrap_or(&1).max(&1);
    let month_max = *data.month_histogram.iter().max().unwrap_or(&1).max(&1);
    let dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

    view! {
        <div class="label-detail-grid">
            <div class="card">
                <div class="panel-header">"Overview"</div>
                <div style="padding:12px;display:grid;grid-template-columns:auto 1fr;gap:4px 12px">
                    <div class="node-detail-label">"Total"</div>
                    <div class="mono">{data.total}</div>
                    <div class="node-detail-label">"First seen"</div>
                    <div class="mono">{first_seen}</div>
                    <div class="node-detail-label">"Last seen"</div>
                    <div class="mono">{last_seen}</div>
                </div>
            </div>
            <div class="card">
                <div class="panel-header">"Hour of day"</div>
                <div class="hist-row">
                    {data.hour_histogram.iter().enumerate().map(|(i, &c)| {
                        let frac = (c as f64) / (hour_max as f64);
                        let h = (frac * 48.0).max(if c > 0 { 2.0 } else { 0.0 });
                        let title = format!("{:02}:00 — {} detections", i, c);
                        view! {
                            <div class="hist-slot" title=title>
                                <div class="hist-bar" style:height=format!("{:.0}px", h)></div>
                                <span class="hist-tick">{format!("{:02}", i)}</span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </div>
            <div class="card">
                <div class="panel-header">"Day of week"</div>
                <div class="hist-row">
                    {data.dow_histogram.iter().enumerate().map(|(i, &c)| {
                        let frac = (c as f64) / (dow_max as f64);
                        let h = (frac * 48.0).max(if c > 0 { 2.0 } else { 0.0 });
                        let name = dow_names[i % 7];
                        view! {
                            <div class="hist-slot" title=format!("{name}: {c}")>
                                <div class="hist-bar" style:height=format!("{:.0}px", h)></div>
                                <span class="hist-tick">{name}</span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </div>
            <div class="card">
                <div class="panel-header">"Month of year"</div>
                <div class="hist-row">
                    {data.month_histogram.iter().enumerate().map(|(i, &c)| {
                        let frac = (c as f64) / (month_max as f64);
                        let h = (frac * 48.0).max(if c > 0 { 2.0 } else { 0.0 });
                        let name = month_names[i % 12];
                        view! {
                            <div class="hist-slot" title=format!("{name}: {c}")>
                                <div class="hist-bar" style:height=format!("{:.0}px", h)></div>
                                <span class="hist-tick">{name}</span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </div>
            <div class="card" style="grid-column:1 / -1">
                <div class="panel-header">{format!("Recent detections ({})", data.recent.len())}</div>
                <RecentDetList detections=data.recent />
            </div>
        </div>
    }
}

#[component]
fn RecentDetList(detections: Vec<crate::state::Detection>) -> impl IntoView {
    view! {
        <table class="data-table">
            <thead>
                <tr>
                    <th>"Time"</th>
                    <th>"Node"</th>
                    <th>"Conf"</th>
                    <th>"Audio"</th>
                </tr>
            </thead>
            <tbody>
                {detections.into_iter().map(|d| {
                    let age = d.received_ns.map(format_age_ns).unwrap_or_else(|| "—".into());
                    let node = d.node_id.clone().unwrap_or_else(|| "—".into());
                    let conf = d.label_confidence.or(d.confidence)
                        .map(|c| format!("{:.0}%", c * 100.0))
                        .unwrap_or_else(|| "—".into());
                    let has_audio = d.has_audio.unwrap_or(false) || d.snippet_path.is_some();
                    let eid = d.event_id.clone();
                    view! {
                        <tr>
                            <td class="mono">{age}</td>
                            <td>{node}</td>
                            <td><span class="conf-pill">{conf}</span></td>
                            <td>
                                {if has_audio {
                                    view! {
                                        <button class="play-btn" on:click=move |_| play_detection(&eid)>"▶"</button>
                                    }.into_any()
                                } else {
                                    view! { <span style="color:var(--text-muted)">"-"</span> }.into_any()
                                }}
                            </td>
                        </tr>
                    }
                }).collect_view()}
            </tbody>
        </table>
    }
}

fn play_detection(event_id: &str) {
    use wasm_bindgen::JsCast;
    let url = format!(
        "/api/v1/detections/{}/audio",
        js_sys::encode_uri_component(event_id)
    );
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(el) = doc.get_element_by_id("audio-player") {
            if let Ok(audio) = el.dyn_into::<web_sys::HtmlAudioElement>() {
                audio.set_src(&url);
                let _ = audio.play();
            }
        }
    }
}

fn format_age_ns(ts_ns: i64) -> String {
    let now_ms = js_sys::Date::now();
    let sample_ms = ts_ns as f64 / 1_000_000.0;
    let age_s = ((now_ms - sample_ms) / 1000.0).max(0.0);
    if age_s < 60.0 {
        format!("{:.0}s ago", age_s)
    } else if age_s < 3600.0 {
        format!("{:.0}m ago", age_s / 60.0)
    } else if age_s < 86400.0 {
        format!("{:.1}h ago", age_s / 3600.0)
    } else {
        format!("{:.1}d ago", age_s / 86400.0)
    }
}
