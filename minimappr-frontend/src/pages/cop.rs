use crate::map::LeafletMapPanel;
use crate::panels::{
    alerts::AlertsPane, detections::DetectionsPane, node_status::NodeStatusPanel,
    tracks::TracksPane,
};
use leptos::prelude::*;

/// COP (Common Operating Picture) — three-column real-time view.
/// Config lives under /settings now, so the right column has Tracks / Detections / Alerts only.
#[component]
pub fn CopPage() -> impl IntoView {
    view! {
        <main class="main-grid">
            <NodeStatusPanel />
            <LeafletMapPanel />
            <CopTabs />
        </main>
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CopTab {
    Tracks,
    Detections,
    Alerts,
}

#[component]
fn CopTabs() -> impl IntoView {
    use crate::state::AppState;
    let state = use_context::<AppState>().expect("AppState");
    let tracks = state.tracks;
    let dets = state.detections;
    let alerts = state.alerts;

    let active = RwSignal::new(CopTab::Tracks);

    view! {
        <div class="panel tab-column">
            <div class="tab-strip" role="tablist">
                <TabButton active=active tab=CopTab::Tracks     label="Tracks"
                    count=Signal::derive(move || tracks.get().len()) />
                <TabButton active=active tab=CopTab::Detections label="Detections"
                    count=Signal::derive(move || dets.get().len()) />
                <TabButton active=active tab=CopTab::Alerts     label="Alerts"
                    count=Signal::derive(move || alerts.get().len()) />
            </div>
            <div class="tab-content">
                {move || match active.get() {
                    CopTab::Tracks     => view! { <TracksPane /> }.into_any(),
                    CopTab::Detections => view! { <DetectionsPane /> }.into_any(),
                    CopTab::Alerts     => view! { <AlertsPane /> }.into_any(),
                }}
            </div>
        </div>
    }
}

#[component]
fn TabButton(
    active: RwSignal<CopTab>,
    tab: CopTab,
    label: &'static str,
    count: Signal<usize>,
) -> impl IntoView {
    view! {
        <button
            class="tab-btn"
            role="tab"
            class:active=move || active.get() == tab
            on:click=move |_| active.set(tab)
        >
            {label}
            <span class="tab-badge" class:zero=move || count.get() == 0>
                {move || { let n = count.get(); if n > 0 { n.to_string() } else { String::new() } }}
            </span>
        </button>
    }
}
