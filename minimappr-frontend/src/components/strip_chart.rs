use leptos::prelude::*;

fn sparkline_path(samples: &[f64], width: f64, height: f64) -> String {
    if samples.len() < 2 {
        return String::new();
    }
    let max = samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-9);
    let points = samples
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = index as f64 / (samples.len() - 1).max(1) as f64 * width;
            let y = height - (value / max) * height;
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>();
    format!("M {}", points.join(" L "))
}

#[component]
pub fn StripChart(
    samples: Vec<f64>,
    #[prop(default = 120.0)] width: f64,
    #[prop(default = 24.0)] height: f64,
    #[prop(default = "var(--md-sys-color-primary)")] color: &'static str,
) -> impl IntoView {
    let path = sparkline_path(&samples, width, height);
    let view_box = format!("0 0 {width:.0} {height:.0}");

    view! {
        <svg
            class="strip-chart"
            viewBox=view_box
            width=width.to_string()
            height=height.to_string()
            aria-hidden="true"
        >
            <path d=path stroke=color stroke-width="1.5" fill="none" vector-effect="non-scaling-stroke" />
        </svg>
    }
}
