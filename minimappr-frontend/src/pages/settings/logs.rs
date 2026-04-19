use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LogRecord {
    seq: u64,
    ts_ns: i64,
    level: String,
    level_no: i32,
    logger: String,
    message: String,
    #[serde(default)]
    exc: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct LogResponse {
    records: Vec<LogRecord>,
    #[serde(default)]
    capacity: u64,
}

fn fmt_ts(ns: i64) -> String {
    // ISO-ish local-ish: use JS Date via js-sys for real local time.
    let ms = (ns as f64) / 1e6;
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    let hh = d.get_hours();
    let mm = d.get_minutes();
    let ss = d.get_seconds();
    let msv = d.get_milliseconds();
    format!("{:02}:{:02}:{:02}.{:03}", hh, mm, ss, msv)
}

fn level_class(level_no: i32) -> &'static str {
    // Python logging: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    if level_no >= 50 {
        "log-critical"
    } else if level_no >= 40 {
        "log-error"
    } else if level_no >= 30 {
        "log-warn"
    } else if level_no >= 20 {
        "log-info"
    } else {
        "log-debug"
    }
}

#[component]
pub fn ServerLogsView() -> impl IntoView {
    let level = RwSignal::new("INFO".to_string());
    let filter = RwSignal::new(String::new());
    let paused = RwSignal::new(false);
    let records: RwSignal<Vec<LogRecord>> = RwSignal::new(Vec::new());
    let since: RwSignal<u64> = RwSignal::new(0);
    let capacity: RwSignal<u64> = RwSignal::new(0);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    let reload_all = move || {
        let lvl = level.get();
        let url = format!("/api/v1/system/logs?limit=500&level={lvl}");
        spawn_local(async move {
            match Request::get(&url).send().await {
                Ok(resp) if resp.ok() => match resp.json::<LogResponse>().await {
                    Ok(d) => {
                        if let Some(last) = d.records.last() {
                            since.set(last.seq);
                        }
                        capacity.set(d.capacity);
                        records.set(d.records);
                        error.set(None);
                    }
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    };

    let tail = move || {
        if paused.get() {
            return;
        }
        let lvl = level.get();
        let s = since.get();
        let url = format!("/api/v1/system/logs?limit=500&level={lvl}&since_seq={s}");
        spawn_local(async move {
            if let Ok(resp) = Request::get(&url).send().await {
                if resp.ok() {
                    if let Ok(d) = resp.json::<LogResponse>().await {
                        if !d.records.is_empty() {
                            if let Some(last) = d.records.last() {
                                since.set(last.seq);
                            }
                            records.update(|r| {
                                r.extend(d.records);
                                let len = r.len();
                                if len > 2000 {
                                    r.drain(0..len - 2000);
                                }
                            });
                        }
                    }
                }
            }
        });
    };

    // Initial load + on level change.
    Effect::new(move |_| {
        let _ = level.get();
        reload_all();
    });

    Effect::new(move |_| {
        spawn_local(async move {
            let mut iv = IntervalStream::new(1_500);
            while iv.next().await.is_some() {
                tail();
            }
        });
    });

    let filtered = move || {
        let f = filter.get().to_lowercase();
        let rs = records.get();
        if f.is_empty() {
            rs
        } else {
            rs.into_iter()
                .filter(|r| {
                    r.message.to_lowercase().contains(&f) || r.logger.to_lowercase().contains(&f)
                })
                .collect()
        }
    };

    view! {
        <div class="page-stub">
            <div class="daily-toolbar">
                <h2 style="margin:0">"Server logs"</h2>
                <select
                    class="btn-sm"
                    on:change=move |ev| {
                        let v = event_target_value(&ev);
                        level.set(v);
                    }
                    prop:value=move || level.get()
                >
                    <option value="DEBUG">"DEBUG"</option>
                    <option value="INFO">"INFO"</option>
                    <option value="WARNING">"WARNING"</option>
                    <option value="ERROR">"ERROR"</option>
                    <option value="CRITICAL">"CRITICAL"</option>
                </select>
                <input
                    class="btn-sm"
                    type="text"
                    placeholder="filter…"
                    prop:value=move || filter.get()
                    on:input=move |ev| filter.set(event_target_value(&ev))
                />
                <button
                    class="btn-sm"
                    class:active=move || paused.get()
                    on:click=move |_| paused.update(|p| *p = !*p)
                >
                    {move || if paused.get() { "Resume" } else { "Pause" }}
                </button>
                <button class="btn-sm" on:click=move |_| reload_all()>"Reload"</button>
                <span class="muted">
                    {move || format!("{} shown · buf {}", filtered().len(), capacity.get())}
                </span>
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>
            <div class="log-list">
                {move || filtered().into_iter().rev().map(|r| {
                    let cls = level_class(r.level_no);
                    let ts = fmt_ts(r.ts_ns);
                    let exc_block = r.exc.clone().map(|e| view! {
                        <pre class="log-exc">{e}</pre>
                    });
                    view! {
                        <div class=format!("log-row {cls}")>
                            <span class="log-ts">{ts}</span>
                            <span class="log-level">{r.level.clone()}</span>
                            <span class="log-logger">{r.logger.clone()}</span>
                            <span class="log-msg">{r.message.clone()}</span>
                            {exc_block}
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
