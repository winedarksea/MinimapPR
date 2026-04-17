use crate::state::AppState;
use leptos::prelude::*;

#[component]
pub fn TracksPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let tracks = state.tracks;

    view! {
        <div class="tab-pane">
            {move || {
                let ts = tracks.get();
                if ts.is_empty() {
                    return view! { <div class="empty-state">"No active tracks"</div> }.into_any();
                }
                view! {
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>"ID"</th>
                                <th>"Label"</th>
                                <th>"Conf"</th>
                                <th>"TQI"</th>
                                <th>"Sensors"</th>
                                <th>"Pos (m)"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {ts.into_iter().map(|t| {
                                let id   = t.track_id[..8.min(t.track_id.len())].to_string();
                                let label= t.label.clone().unwrap_or_else(|| "—".to_string());
                                let conf = t.confidence.map(|c| format!("{:.0}%", c * 100.0)).unwrap_or_else(|| "—".to_string());
                                let tqi_val = t.tqi.unwrap_or(0.0);
                                let tqi_w = format!("{}px", (tqi_val * 60.0) as u32);
                                let sensors = t.sensor_count.map(|s| s.to_string()).unwrap_or_else(|| "—".to_string());
                                let pos = t.position_m.as_ref().map(|p| {
                                    match p.as_slice() {
                                        [x, y, z] => format!("{x:.1},{y:.1},{z:.1}"),
                                        [x, y]    => format!("{x:.1},{y:.1}"),
                                        _          => "—".to_string(),
                                    }
                                }).unwrap_or_else(|| "—".to_string());

                                view! {
                                    <tr>
                                        <td><code>{id}</code></td>
                                        <td>{label}</td>
                                        <td><span class="conf-pill">{conf}</span></td>
                                        <td>
                                            <span class="tqi-bar" style:width=tqi_w></span>
                                        </td>
                                        <td>{sensors}</td>
                                        <td style="font-size:0.7rem">{pos}</td>
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
