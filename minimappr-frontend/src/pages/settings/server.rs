use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PyInfo {
    #[serde(default)]
    version: String,
    #[serde(default)]
    implementation: String,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PlatformInfo {
    #[serde(default)]
    system: String,
    #[serde(default)]
    release: String,
    #[serde(default)]
    machine: String,
    #[serde(default)]
    node: String,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct ProcInfo {
    #[serde(default)]
    pid: u64,
    #[serde(default)]
    threads: u64,
    #[serde(default)]
    cpu_user_s: Option<f64>,
    #[serde(default)]
    cpu_system_s: Option<f64>,
    #[serde(default)]
    max_rss_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct LoadInfo {
    #[serde(rename = "1m", default)]
    one: f64,
    #[serde(rename = "5m", default)]
    five: f64,
    #[serde(rename = "15m", default)]
    fifteen: f64,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct DiskEntry {
    #[serde(default)]
    path: String,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    used: u64,
    #[serde(default)]
    free: u64,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct Diagnostics {
    #[serde(default)]
    now_ns: i64,
    #[serde(default)]
    uptime_ns: Option<i64>,
    #[serde(default)]
    python: PyInfo,
    #[serde(default)]
    platform: PlatformInfo,
    #[serde(default)]
    process: ProcInfo,
    #[serde(default)]
    load: Option<LoadInfo>,
    #[serde(default)]
    cpu_count: Option<u32>,
    #[serde(default)]
    disk: std::collections::BTreeMap<String, Option<DiskEntry>>,
    #[serde(default)]
    pipeline: Option<PipelineDiagnostics>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PipelineDiagnostics {
    #[serde(default)]
    queue: QueueInfo,
    #[serde(default)]
    workers: WorkerInfo,
    #[serde(default)]
    realtime: RealtimeInfo,
    #[serde(default)]
    drop_on_backpressure: bool,
    #[serde(default)]
    metrics: PipelineMetrics,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct QueueInfo {
    #[serde(default)]
    localization_depth: u64,
    #[serde(default)]
    classification_depth: u64,
    #[serde(default)]
    rules_depth: u64,
    #[serde(default)]
    localization_max: u64,
    #[serde(default)]
    classification_max: u64,
    #[serde(default)]
    rules_max: u64,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct WorkerInfo {
    #[serde(default)]
    configured: u64,
    #[serde(default)]
    localization_running: u64,
    #[serde(default)]
    classification_running: u64,
    #[serde(default)]
    rules_running: u64,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct RealtimeInfo {
    #[serde(default)]
    pipeline_seconds_behind_realtime: Option<f64>,
    #[serde(default)]
    last_completed_lag_seconds: Option<f64>,
    #[serde(default)]
    max_completed_lag_seconds: Option<f64>,
    #[serde(default)]
    stages: std::collections::BTreeMap<String, StageRealtimeInfo>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct StageRealtimeInfo {
    #[serde(default)]
    queued_items: u64,
    #[serde(default)]
    inflight_items: u64,
    #[serde(default)]
    seconds_behind_realtime: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
struct PipelineMetrics {
    #[serde(default)]
    triggers_enqueued: u64,
    #[serde(default)]
    triggers_dropped_queue_full: u64,
    #[serde(default)]
    stage_drops_backpressure: u64,
    #[serde(default)]
    classification_reuse_hits: u64,
    #[serde(default)]
    detections_emitted: u64,
}

fn fmt_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", UNITS[i])
}

fn fmt_uptime(ns: i64) -> String {
    let s = (ns as f64 / 1e9) as i64;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    if h > 0 {
        format!("{h}h {m}m {sec}s")
    } else if m > 0 {
        format!("{m}m {sec}s")
    } else {
        format!("{sec}s")
    }
}

fn fmt_seconds(value: Option<f64>) -> String {
    match value {
        Some(seconds) if seconds >= 60.0 => format!("{:.1} min", seconds / 60.0),
        Some(seconds) => format!("{seconds:.2} s"),
        None => "—".into(),
    }
}

#[component]
pub fn ServerDiagnosticsView() -> impl IntoView {
    let data: RwSignal<Option<Diagnostics>> = RwSignal::new(None);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    let fetch = move || {
        spawn_local(async move {
            match Request::get("/api/v1/system/diagnostics").send().await {
                Ok(resp) if resp.ok() => match resp.json::<Diagnostics>().await {
                    Ok(d) => {
                        data.set(Some(d));
                        error.set(None);
                    }
                    Err(e) => error.set(Some(format!("parse: {e}"))),
                },
                Ok(resp) => error.set(Some(format!("HTTP {}", resp.status()))),
                Err(e) => error.set(Some(e.to_string())),
            }
        });
    };

    Effect::new(move |_| fetch());

    // Refresh every 3s while mounted.
    Effect::new(move |_| {
        spawn_local(async move {
            let mut iv = IntervalStream::new(3_000);
            while iv.next().await.is_some() {
                fetch();
            }
        });
    });

    view! {
        <div class="page-stub">
            <div class="daily-toolbar">
                <h2 style="margin:0">"Server diagnostics"</h2>
                <span class="daily-error">{move || error.get().unwrap_or_default()}</span>
            </div>

            {move || {
                let d = data.get();
                match d {
                    None => view! { <div class="muted">"Loading…"</div> }.into_any(),
                    Some(d) => {
                        let uptime = d.uptime_ns.map(fmt_uptime).unwrap_or_else(|| "—".into());
                        let cpu_count = d.cpu_count.map(|c| c.to_string()).unwrap_or_else(|| "—".into());
                        let (la1, la5, la15) = match &d.load {
                            Some(l) => (format!("{:.2}", l.one), format!("{:.2}", l.five), format!("{:.2}", l.fifteen)),
                            None => ("—".into(), "—".into(), "—".into()),
                        };
                        let rss = d.process.max_rss_bytes.map(fmt_bytes).unwrap_or_else(|| "—".into());
                        let cpu_user = d.process.cpu_user_s.map(|v| format!("{v:.1}s")).unwrap_or_else(|| "—".into());
                        let cpu_sys = d.process.cpu_system_s.map(|v| format!("{v:.1}s")).unwrap_or_else(|| "—".into());
                        let disks = d.disk.clone();
                        let pipeline = d.pipeline.clone().unwrap_or_default();
                        let localization_lag = pipeline
                            .realtime
                            .stages
                            .get("localization")
                            .and_then(|stage| stage.seconds_behind_realtime);
                        let classification_lag = pipeline
                            .realtime
                            .stages
                            .get("classification")
                            .and_then(|stage| stage.seconds_behind_realtime);
                        let rules_lag = pipeline
                            .realtime
                            .stages
                            .get("rules")
                            .and_then(|stage| stage.seconds_behind_realtime);

                        view! {
                            <div class="diag-grid">
                                <DiagCard title="Runtime">
                                    <DiagRow k="Uptime".into() v=uptime />
                                    <DiagRow k="Python".into() v=format!("{} ({})", d.python.version, d.python.implementation) />
                                    <DiagRow k="Platform".into() v=format!("{} {} ({})", d.platform.system, d.platform.release, d.platform.machine) />
                                    <DiagRow k="Host".into() v=d.platform.node.clone() />
                                </DiagCard>

                                <DiagCard title="Process">
                                    <DiagRow k="PID".into() v=d.process.pid.to_string() />
                                    <DiagRow k="Threads".into() v=d.process.threads.to_string() />
                                    <DiagRow k="CPU user".into() v=cpu_user />
                                    <DiagRow k="CPU system".into() v=cpu_sys />
                                    <DiagRow k="Max RSS".into() v=rss />
                                </DiagCard>

                                <DiagCard title="Load">
                                    <DiagRow k="CPU count".into() v=cpu_count />
                                    <DiagRow k="Load 1m".into() v=la1 />
                                    <DiagRow k="Load 5m".into() v=la5 />
                                    <DiagRow k="Load 15m".into() v=la15 />
                                </DiagCard>

                                <DiagCard title="Pipeline">
                                    <DiagRow k="Behind real time".into() v=fmt_seconds(pipeline.realtime.pipeline_seconds_behind_realtime) />
                                    <DiagRow k="Last detection lag".into() v=fmt_seconds(pipeline.realtime.last_completed_lag_seconds) />
                                    <DiagRow k="Worst detection lag".into() v=fmt_seconds(pipeline.realtime.max_completed_lag_seconds) />
                                    <DiagRow
                                        k="Queue depths".into()
                                        v=format!(
                                            "L {}/{}  C {}/{}  R {}/{}",
                                            pipeline.queue.localization_depth,
                                            pipeline.queue.localization_max,
                                            pipeline.queue.classification_depth,
                                            pipeline.queue.classification_max,
                                            pipeline.queue.rules_depth,
                                            pipeline.queue.rules_max,
                                        )
                                    />
                                    <DiagRow
                                        k="Workers".into()
                                        v=format!(
                                            "cfg {}  live L:{} C:{} R:{}",
                                            pipeline.workers.configured,
                                            pipeline.workers.localization_running,
                                            pipeline.workers.classification_running,
                                            pipeline.workers.rules_running,
                                        )
                                    />
                                    <DiagRow k="Localization lag".into() v=fmt_seconds(localization_lag) />
                                    <DiagRow k="Classification lag".into() v=fmt_seconds(classification_lag) />
                                    <DiagRow k="Rules lag".into() v=fmt_seconds(rules_lag) />
                                    <DiagRow
                                        k="Backpressure".into()
                                        v=if pipeline.drop_on_backpressure { "drop when full".into() } else { "block / replay".into() }
                                    />
                                    <DiagRow
                                        k="Dropped triggers".into()
                                        v=pipeline.metrics.triggers_dropped_queue_full.to_string()
                                    />
                                    <DiagRow
                                        k="Stage drops".into()
                                        v=pipeline.metrics.stage_drops_backpressure.to_string()
                                    />
                                </DiagCard>

                                <DiagCard title="Disk">
                                    {disks.into_iter().map(|(name, entry)| {
                                        match entry {
                                            Some(e) => {
                                                let pct = if e.total > 0 { (e.used as f64 / e.total as f64) * 100.0 } else { 0.0 };
                                                view! {
                                                    <div class="diag-disk">
                                                        <div class="diag-disk-head">
                                                            <strong>{name}</strong>
                                                            <span class="muted">{e.path.clone()}</span>
                                                        </div>
                                                        <div class="diag-disk-bar">
                                                            <div class="diag-disk-fill" style=format!("width:{:.1}%", pct)></div>
                                                        </div>
                                                        <div class="diag-disk-meta muted">
                                                            {format!("{} used / {} ({:.1}%) — {} free", fmt_bytes(e.used), fmt_bytes(e.total), pct, fmt_bytes(e.free))}
                                                        </div>
                                                    </div>
                                                }.into_any()
                                            }
                                            None => view! { <DiagRow k=name.clone() v="unavailable".into() /> }.into_any(),
                                        }
                                    }).collect_view()}
                                </DiagCard>
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}

#[component]
fn DiagCard(title: &'static str, children: Children) -> impl IntoView {
    view! {
        <div class="diag-card">
            <h3>{title}</h3>
            <div class="diag-card-body">{children()}</div>
        </div>
    }
}

#[component]
fn DiagRow(k: String, v: String) -> impl IntoView {
    view! {
        <div class="diag-row">
            <span class="muted">{k}</span>
            <span class="diag-val">{v}</span>
        </div>
    }
}
