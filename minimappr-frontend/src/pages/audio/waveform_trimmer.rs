//! Waveform trimmer for calibration ground-truth events.
//!
//! Field captures typically run: start recording → wait → trigger the test
//! event → wait → stop. Rather than typing nanosecond timestamps, the
//! operator sees the node's raw audio as a waveform and drags two handles to
//! bracket the event; the handles write directly into the same start_ns/
//! end_ns signals the manual entry fields use.

use crate::recording::api;
use crate::recording::CalibrationManifest;
use leptos::prelude::*;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

const CANVAS_HEIGHT: f64 = 100.0;
const PEAK_BUCKETS: usize = 800;

#[derive(Clone)]
struct WaveformData {
    peaks: Vec<(f32, f32)>,
    duration_ns: f64,
    audio_start_ns: f64,
}

#[component]
pub fn WaveformTrimmer(
    session_id: String,
    manifest: RwSignal<Option<CalibrationManifest>>,
    view_node_id: RwSignal<String>,
    start_ns: RwSignal<String>,
    end_ns: RwSignal<String>,
) -> impl IntoView {
    let canvas_ref: NodeRef<leptos::html::Canvas> = NodeRef::new();
    let audio_ref: NodeRef<leptos::html::Audio> = NodeRef::new();
    let waveform: RwSignal<Option<WaveformData>> = RwSignal::new(None);
    let loading = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let playing = RwSignal::new(false);
    let playhead_ns: RwSignal<Option<f64>> = RwSignal::new(None);

    // (Re)load + decode audio whenever the viewed node changes.
    {
        let session_id = session_id.clone();
        Effect::new(move |_| {
            let node_id = view_node_id.get();
            if node_id.is_empty() {
                return;
            }
            let session_id = session_id.clone();
            let manifest_snapshot = manifest.get_untracked();
            waveform.set(None);
            playhead_ns.set(None);
            error.set(None);
            loading.set(true);
            spawn_local(async move {
                match load_waveform(&session_id, &node_id, manifest_snapshot).await {
                    Ok(data) => waveform.set(Some(data)),
                    Err(e) => error.set(Some(e)),
                }
                loading.set(false);
            });
        });
    }

    // Redraw on any relevant signal change.
    Effect::new(move |_| {
        let Some(data) = waveform.get() else {
            return;
        };
        let start_v = start_ns.get().trim().parse::<f64>().ok();
        let end_v = end_ns.get().trim().parse::<f64>().ok();
        let head = playhead_ns.get();
        if let Some(canvas) = canvas_ref.get() {
            draw_waveform(&canvas, &data, start_v, end_v, head);
        }
    });

    let on_mousedown = move |event: web_sys::MouseEvent| {
        let Some(data) = waveform.get_untracked() else {
            return;
        };
        let Some(canvas) = canvas_ref.get_untracked() else {
            return;
        };
        if data.duration_ns <= 0.0 {
            return;
        }
        let rect = canvas.get_bounding_client_rect();
        let width = rect.width();
        if width <= 0.0 {
            return;
        }
        let x = event.client_x() as f64 - rect.left();
        let ns_at_x = data.audio_start_ns + (x / width).clamp(0.0, 1.0) * data.duration_ns;

        let default_start = data.audio_start_ns;
        let default_end = data.audio_start_ns + data.duration_ns;
        let start_v = start_ns
            .get_untracked()
            .trim()
            .parse::<f64>()
            .unwrap_or(default_start);
        let end_v = end_ns
            .get_untracked()
            .trim()
            .parse::<f64>()
            .unwrap_or(default_end);
        let dragging_start = (ns_at_x - start_v).abs() <= (ns_at_x - end_v).abs();

        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        let audio_start_ns = data.audio_start_ns;
        let duration_ns = data.duration_ns;
        let canvas_for_move = canvas.clone();
        let on_move = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(
            move |move_event: web_sys::MouseEvent| {
                let rect = canvas_for_move.get_bounding_client_rect();
                let width = rect.width();
                if width <= 0.0 {
                    return;
                }
                let x = move_event.client_x() as f64 - rect.left();
                let ns = audio_start_ns + (x / width).clamp(0.0, 1.0) * duration_ns;
                if dragging_start {
                    let end_v = end_ns
                        .get_untracked()
                        .trim()
                        .parse::<f64>()
                        .unwrap_or(audio_start_ns + duration_ns);
                    start_ns.set(format!("{}", ns.min(end_v) as i64));
                } else {
                    let start_v = start_ns
                        .get_untracked()
                        .trim()
                        .parse::<f64>()
                        .unwrap_or(audio_start_ns);
                    end_ns.set(format!("{}", ns.max(start_v) as i64));
                }
            },
        );
        let on_move_ref = on_move.as_ref().unchecked_ref();
        let _ = document.add_event_listener_with_callback("mousemove", on_move_ref);

        let cleanup_document = document.clone();
        // `on_move` moves into this closure and is dropped once it fires
        // (after detaching the listener), instead of being leaked forever.
        let on_up = Closure::<dyn FnMut(web_sys::MouseEvent)>::once(move |_| {
            let _ = cleanup_document
                .remove_event_listener_with_callback("mousemove", on_move.as_ref().unchecked_ref());
        });
        let _ =
            document.add_event_listener_with_callback("mouseup", on_up.as_ref().unchecked_ref());
        on_up.forget();
    };

    let toggle_play = move |_| {
        let Some(audio) = audio_ref.get_untracked() else {
            return;
        };
        if playing.get_untracked() {
            let _ = audio.pause();
        } else {
            let _ = audio.play();
        }
    };

    let audio_src = {
        let session_id = session_id.clone();
        move || {
            let node_id = view_node_id.get();
            if node_id.is_empty() {
                String::new()
            } else {
                api::calibration_audio_url(&session_id, &node_id)
            }
        }
    };

    view! {
        <div class="waveform-trimmer">
            <div class="waveform-toolbar">
                <select class="form-input waveform-node-select"
                    prop:value=move || view_node_id.get()
                    on:change=move |ev| view_node_id.set(event_target_value(&ev))>
                    {move || manifest.get().map(|m| m.nodes).unwrap_or_default().into_iter().map(|n| {
                        let id = n.node_id.clone();
                        let label = id.clone();
                        view! { <option value=id>{label}</option> }
                    }).collect::<Vec<_>>()}
                </select>
                <button type="button" class="btn-sm"
                    on:click=toggle_play
                    disabled=move || waveform.get().is_none()>
                    {move || if playing.get() { "Pause" } else { "Play" }}
                </button>
                <span class="muted waveform-hint">
                    "Drag the green (start) / red (end) markers to bracket the event"
                </span>
                {move || loading.get().then(|| view! { <span class="muted">"Loading audio…"</span> })}
            </div>
            <canvas
                node_ref=canvas_ref
                class="waveform-canvas"
                on:mousedown=on_mousedown
            ></canvas>
            <audio
                node_ref=audio_ref
                src=audio_src
                on:play=move |_| playing.set(true)
                on:pause=move |_| playing.set(false)
                on:ended=move |_| playing.set(false)
                on:timeupdate=move |_| {
                    let audio_el = audio_ref.get_untracked();
                    let data = waveform.get_untracked();
                    if let (Some(audio_el), Some(data)) = (audio_el, data) {
                        let current: f64 = audio_el.current_time();
                        playhead_ns.set(Some(data.audio_start_ns + current * 1e9));
                    }
                }
            ></audio>
            {move || error.get().map(|e| view! { <p class="row-error">{e}</p> })}
        </div>
    }
}

async fn load_waveform(
    session_id: &str,
    node_id: &str,
    manifest: Option<CalibrationManifest>,
) -> Result<WaveformData, String> {
    let audio_start_ns = manifest
        .and_then(|m| m.nodes.into_iter().find(|n| n.node_id == node_id))
        .map(|n| n.audio_start_time_ns)
        .unwrap_or(0.0);

    let bytes = api::fetch_calibration_audio_bytes(session_id, node_id).await?;
    let array = js_sys::Uint8Array::from(bytes.as_slice());
    let array_buffer = array.buffer();

    let ctx = web_sys::AudioContext::new().map_err(|e| format!("{e:?}"))?;
    let promise = ctx
        .decode_audio_data(&array_buffer)
        .map_err(|e| format!("{e:?}"))?;
    let js_value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|e| format!("failed to decode audio: {e:?}"))?;
    let audio_buffer: web_sys::AudioBuffer = js_value
        .dyn_into()
        .map_err(|_| "unexpected decode result".to_string())?;

    let sample_rate = audio_buffer.sample_rate() as f64;
    let n_channels = audio_buffer.number_of_channels();
    let length = audio_buffer.length() as usize;

    let mut mono = vec![0f32; length];
    for ch in 0..n_channels {
        let mut data = vec![0f32; length];
        audio_buffer
            .copy_from_channel(&mut data, ch as i32)
            .map_err(|e| format!("{e:?}"))?;
        for (m, v) in mono.iter_mut().zip(data.iter()) {
            *m += v;
        }
    }
    if n_channels > 0 {
        for v in mono.iter_mut() {
            *v /= n_channels as f32;
        }
    }

    let buckets = PEAK_BUCKETS.min(length.max(1));
    let chunk = ((length as f64) / (buckets as f64)).ceil().max(1.0) as usize;
    let mut peaks = Vec::with_capacity(buckets);
    let mut start = 0usize;
    while start < length {
        let end = (start + chunk).min(length);
        let slice = &mono[start..end];
        let mut min_v = 0f32;
        let mut max_v = 0f32;
        for &v in slice {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        peaks.push((min_v, max_v));
        start = end;
    }

    let duration_ns = if sample_rate > 0.0 {
        (length as f64 / sample_rate) * 1_000_000_000.0
    } else {
        0.0
    };

    Ok(WaveformData {
        peaks,
        duration_ns,
        audio_start_ns,
    })
}

fn draw_waveform(
    canvas: &web_sys::HtmlCanvasElement,
    data: &WaveformData,
    start_ns: Option<f64>,
    end_ns: Option<f64>,
    playhead_ns: Option<f64>,
) {
    let rect = canvas.get_bounding_client_rect();
    let width = rect.width().max(1.0);
    let height = CANVAS_HEIGHT;
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);

    let Ok(Some(ctx_obj)) = canvas.get_context("2d") else {
        return;
    };
    let Ok(ctx) = ctx_obj.dyn_into::<web_sys::CanvasRenderingContext2d>() else {
        return;
    };

    ctx.clear_rect(0.0, 0.0, width, height);
    ctx.set_fill_style_str("#0d1117");
    ctx.fill_rect(0.0, 0.0, width, height);

    let ns_to_x = |ns: f64| -> f64 {
        if data.duration_ns <= 0.0 {
            return 0.0;
        }
        ((ns - data.audio_start_ns) / data.duration_ns).clamp(0.0, 1.0) * width
    };

    if let (Some(s), Some(e)) = (start_ns, end_ns) {
        let x0 = ns_to_x(s.min(e));
        let x1 = ns_to_x(s.max(e));
        ctx.set_fill_style_str("rgba(88, 166, 255, 0.18)");
        ctx.fill_rect(x0, 0.0, (x1 - x0).max(0.0), height);
    }

    let mid = height / 2.0;
    let n = data.peaks.len().max(1) as f64;
    ctx.set_stroke_style_str("#58a6ff");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    for (i, (min_v, max_v)) in data.peaks.iter().enumerate() {
        let x = (i as f64 / n) * width;
        let y_min = mid - (*max_v as f64) * mid;
        let y_max = mid - (*min_v as f64) * mid;
        ctx.move_to(x, y_min);
        ctx.line_to(x, y_max);
    }
    ctx.stroke();

    if let Some(s) = start_ns {
        draw_marker(&ctx, ns_to_x(s), height, "#3fb950");
    }
    if let Some(e) = end_ns {
        draw_marker(&ctx, ns_to_x(e), height, "#f85149");
    }
    if let Some(p) = playhead_ns {
        draw_marker(&ctx, ns_to_x(p), height, "#e6edf3");
    }
}

fn draw_marker(ctx: &web_sys::CanvasRenderingContext2d, x: f64, height: f64, color: &str) {
    ctx.set_stroke_style_str(color);
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x, 0.0);
    ctx.line_to(x, height);
    ctx.stroke();
}
