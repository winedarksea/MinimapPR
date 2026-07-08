pub mod context_menu;
pub mod dock;
pub mod layout;
pub mod registry;
pub mod status_ribbon;

use crate::components::drawer_shell::DrawerShell;
use crate::inspector::EntityInspector;
use crate::map::bindings::pan_to;
use crate::map::MapPanel;
use crate::panels::{
    alerts::AlertsPane, detections::DetectionsPane, node_status::NodeStatusPanel,
    rf_panel::RfPanel, seismic_panel::SeismicPanel, speech_panel::SpeechPanel, tracks::TracksPane,
};
use crate::state::{AppState, CopItemKind, CopSelection, ZoneSpec};
use layout::{clamp_dock_width, WorkspaceLayout};
use leptos::prelude::*;
use registry::DrawerId;

#[component]
pub fn MapWorkspace() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let persisted_layout = WorkspaceLayout::load();

    let nodes_open = RwSignal::new(persisted_layout.nodes_open);
    let tracks_open = RwSignal::new(persisted_layout.tracks_open);
    let detections_open = RwSignal::new(persisted_layout.detections_open);
    let alerts_open = RwSignal::new(persisted_layout.alerts_open);
    let zones_open = RwSignal::new(persisted_layout.zones_open);
    let audio_open = RwSignal::new(persisted_layout.audio_open);
    let rf_open = RwSignal::new(persisted_layout.rf_open);
    let seismic_open = RwSignal::new(persisted_layout.seismic_open);
    let speech_open = RwSignal::new(persisted_layout.speech_open);
    let left_dock_width_px = RwSignal::new(clamp_dock_width(persisted_layout.left_dock_width_px));
    let right_dock_width_px = RwSignal::new(clamp_dock_width(persisted_layout.right_dock_width_px));

    Effect::new(move |_| {
        WorkspaceLayout {
            nodes_open: nodes_open.get(),
            tracks_open: tracks_open.get(),
            detections_open: detections_open.get(),
            alerts_open: alerts_open.get(),
            zones_open: zones_open.get(),
            audio_open: audio_open.get(),
            rf_open: rf_open.get(),
            seismic_open: seismic_open.get(),
            speech_open: speech_open.get(),
            left_dock_width_px: left_dock_width_px.get(),
            right_dock_width_px: right_dock_width_px.get(),
        }
        .save();
    });

    let node_count = Signal::derive(move || state.nodes.get().len());
    let track_count = Signal::derive(move || state.tracks.get().len());
    let detection_count = Signal::derive(move || state.detections.get().len());
    let alert_count = Signal::derive(move || state.alerts.get().len());
    let zone_count = Signal::derive(move || state.zones.get().len());
    let rf_node_count = Signal::derive(move || {
        state
            .nodes
            .get()
            .into_iter()
            .filter(|node| node.has_capability("sdr") || node.has_capability("radar_24g"))
            .count()
    });
    let seismic_node_count = Signal::derive(move || {
        state
            .nodes
            .get()
            .into_iter()
            .filter(|node| node.has_capability("seismic"))
            .count()
    });
    let speech_node_count = Signal::derive(move || {
        state
            .nodes
            .get()
            .into_iter()
            .filter(|node| node.has_capability("speech"))
            .count()
    });
    let rf_badge_count = Signal::derive(move || {
        state
            .modality
            .rf_spectra
            .get()
            .into_iter()
            .map(|frame| frame.emitters.len())
            .sum::<usize>()
    });
    let seismic_badge_count = Signal::derive(move || state.modality.seismic_traces.get().len());
    let speech_badge_count = Signal::derive(move || state.modality.speech_lines.get().len());
    let audio_badge = Signal::derive(move || {
        usize::from(
            state.audio_drawer_detection_id.get().is_some()
                || state.audio_drawer_track_id.get().is_some(),
        )
    });

    view! {
        <main class="cop-workspace">
            <div class="cop-workspace-map">
                <MapPanel />
            </div>

            <status_ribbon::StatusRibbon />

            <dock::WorkspaceDock side="left" width_px=left_dock_width_px>
                <DrawerShell title=DrawerId::Nodes.title() icon="hub" open=nodes_open badge=node_count>
                    <NodeStatusPanel />
                </DrawerShell>
            </dock::WorkspaceDock>

            <dock::WorkspaceDock side="right" width_px=right_dock_width_px>
                <DrawerShell title=DrawerId::Tracks.title() icon="near_me" open=tracks_open badge=track_count>
                    <TracksPane />
                </DrawerShell>
                <DrawerShell title=DrawerId::Detections.title() icon="graphic_eq" open=detections_open badge=detection_count>
                    <DetectionsPane />
                </DrawerShell>
                <DrawerShell title=DrawerId::Alerts.title() icon="notification_important" open=alerts_open badge=alert_count>
                    <AlertsPane />
                </DrawerShell>
                <DrawerShell title=DrawerId::Zones.title() icon="polyline" open=zones_open badge=zone_count>
                    <ZonesPane />
                </DrawerShell>
                <DrawerShell title=DrawerId::Audio.title() icon="equalizer" open=audio_open badge=audio_badge>
                    <div class="workspace-audio-drawer">
                        <p class="muted">
                            "Audio review opens as a focused slide-over so playback and analysis stay stable while the map remains live."
                        </p>
                        <button
                            class="btn-sm"
                            on:click=move |_| state.audio_drawer_open.set(true)
                        >
                            "Open audio analysis"
                        </button>
                    </div>
                </DrawerShell>
                {move || (rf_node_count.get() > 0).then(|| view! {
                    <DrawerShell title=DrawerId::Rf.title() icon="settings_input_antenna" open=rf_open badge=rf_badge_count>
                        <RfPanel />
                    </DrawerShell>
                })}
                {move || (seismic_node_count.get() > 0).then(|| view! {
                    <DrawerShell title=DrawerId::Seismic.title() icon="monitor_heart" open=seismic_open badge=seismic_badge_count>
                        <SeismicPanel />
                    </DrawerShell>
                })}
                {move || (speech_node_count.get() > 0).then(|| view! {
                    <DrawerShell title=DrawerId::Speech.title() icon="record_voice_over" open=speech_open badge=speech_badge_count>
                        <SpeechPanel />
                    </DrawerShell>
                })}
            </dock::WorkspaceDock>

            <EntityInspector right_dock_width_px />
            <context_menu::MapContextMenu />
            <context_menu::ZoneDraftDialog />
        </main>
    }
}

#[component]
fn ZonesPane() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let zones = state.zones;
    let occupancy = state.zone_occupancy;
    let selected_cop_item = state.selected_cop_item;

    view! {
        <div class="tab-pane">
            {move || {
                let zone_list = zones.get();
                if zone_list.is_empty() {
                    return view! { <div class="empty-state">"No zones configured"</div> }.into_any();
                }

                view! {
                    <ul class="compact-list">
                        {zone_list.into_iter().map(|zone| {
                            let zone_id = zone.id.clone();
                            let click_id = zone_id.clone();
                            let occupied = occupancy
                                .get()
                                .iter()
                                .find(|state| state.zone_id == zone_id)
                                .map(|state| state.occupied)
                                .unwrap_or(false);
                            let row_class = move || {
                                let selected = selected_cop_item.get();
                                let is_selected = selected
                                    .as_ref()
                                    .is_some_and(|item| item.kind == CopItemKind::Zone && item.id == zone_id);
                                match (occupied, is_selected) {
                                    (true, true) => "compact-row cop-row-selected zone-row-occupied",
                                    (true, false) => "compact-row zone-row-occupied",
                                    (false, true) => "compact-row cop-row-selected",
                                    (false, false) => "compact-row",
                                }
                            };
                            let center = zone_center(&zone);
                            view! {
                                <li>
                                    <button
                                        type="button"
                                        class=row_class
                                        on:click=move |_| {
                                            selected_cop_item.set(Some(CopSelection::pinned(
                                                CopItemKind::Zone,
                                                click_id.clone(),
                                            )));
                                            if let Some((lat, lon)) = center {
                                                pan_to(lat, lon);
                                            }
                                        }
                                    >
                                        <span class="row-status-dot age-fresh"></span>
                                        <span class="row-label">{zone.name}</span>
                                        <span class="row-summary-meta">
                                            <span class="conf-pill">{zone.zone_type}</span>
                                            <span class="conf-pill">{if occupied { "occupied" } else { "clear" }}</span>
                                        </span>
                                    </button>
                                </li>
                            }
                        }).collect_view()}
                    </ul>
                }.into_any()
            }}
        </div>
    }
}

fn zone_center(zone: &ZoneSpec) -> Option<(f64, f64)> {
    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;
    let mut count = 0.0;
    for point in &zone.polygon_geo {
        if point.len() >= 2 {
            lat_sum += point[0];
            lon_sum += point[1];
            count += 1.0;
        }
    }
    (count > 0.0).then_some((lat_sum / count, lon_sum / count))
}
