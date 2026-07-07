use leptos::prelude::*;

#[component]
pub fn RmsMeter(samples: Vec<f64>, label: &'static str) -> impl IntoView {
    let peak = samples.iter().copied().fold(0.0_f64, f64::max);
    let level = peak.clamp(0.0, 1.0);
    let width = format!("{:.1}%", level * 100.0);
    let tone = if peak >= 0.5 {
        "danger"
    } else if peak >= 0.08 {
        "ok"
    } else if samples.is_empty() {
        "empty"
    } else {
        "quiet"
    };

    view! {
        <div class=format!("rms-meter rms-meter-{tone}") title=format!("{label}: peak {peak:.5}")>
            <span class="rms-meter-label">{label}</span>
            <span class="rms-meter-track">
                <span class="rms-meter-fill" style:width=width></span>
            </span>
            <span class="rms-meter-value">{format!("{peak:.4}")}</span>
        </div>
    }
}
