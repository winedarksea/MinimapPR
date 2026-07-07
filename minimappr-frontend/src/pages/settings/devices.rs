use crate::components::form_row::FormRow;
use crate::devices::registry::save_devices;
use crate::devices::schema::{DeviceKind, DeviceRecord};
use crate::state::AppState;
use js_sys::Date;
use leptos::prelude::*;

fn parse_optional_f64(value: &str) -> Result<Option<f64>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<f64>()
        .map(Some)
        .map_err(|_| format!("'{trimmed}' is not a number"))
}

fn now_ns() -> i64 {
    (Date::now() * 1_000_000.0) as i64
}

fn upsert_device(devices: &mut Vec<DeviceRecord>, next: DeviceRecord) {
    if let Some(existing) = devices.iter_mut().find(|device| device.id == next.id) {
        *existing = next;
    } else {
        devices.push(next);
        devices.sort_by(|left, right| {
            left.kind
                .as_str()
                .cmp(right.kind.as_str())
                .then(left.id.cmp(&right.id))
        });
    }
}

#[component]
pub fn DevicesView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let devices = state.devices;
    let selected_kind = RwSignal::new(DeviceKind::Radar24G.as_str().to_string());
    let device_id = RwSignal::new(String::new());
    let label = RwSignal::new(String::new());
    let lat = RwSignal::new(String::new());
    let lon = RwSignal::new(String::new());
    let alt_m = RwSignal::new(String::new());
    let yaw_deg = RwSignal::new(String::new());
    let pitch_deg = RwSignal::new(String::new());
    let import_json = RwSignal::new(String::new());
    let message = RwSignal::new(None::<String>);
    let error = RwSignal::new(None::<String>);

    let save_current = move |_| {
        error.set(None);
        message.set(None);

        let id = device_id.get_untracked().trim().to_string();
        if id.is_empty() {
            error.set(Some("Device id is required".to_string()));
            return;
        }

        let kind = DeviceKind::parse(&selected_kind.get_untracked());
        let parsed_lat = match parse_optional_f64(&lat.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(format!("Latitude {reason}")));
                return;
            }
        };
        let parsed_lon = match parse_optional_f64(&lon.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(format!("Longitude {reason}")));
                return;
            }
        };
        if parsed_lat.is_some() != parsed_lon.is_some() {
            error.set(Some(
                "Latitude and longitude must be provided together".to_string(),
            ));
            return;
        }
        let parsed_alt = match parse_optional_f64(&alt_m.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(format!("Altitude {reason}")));
                return;
            }
        };
        let parsed_yaw = match parse_optional_f64(&yaw_deg.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(format!("Yaw {reason}")));
                return;
            }
        };
        let parsed_pitch = match parse_optional_f64(&pitch_deg.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(format!("Pitch {reason}")));
                return;
            }
        };
        if kind == DeviceKind::Radar24G && (parsed_lat.is_none() || parsed_yaw.is_none()) {
            error.set(Some(
                "24G radar requires a geo position and yaw orientation".to_string(),
            ));
            return;
        }

        let next = DeviceRecord {
            id,
            kind,
            label: label.get_untracked().trim().to_string(),
            lat: parsed_lat,
            lon: parsed_lon,
            alt_m: parsed_alt,
            yaw_deg: parsed_yaw,
            pitch_deg: parsed_pitch,
            created_ns: now_ns(),
        };

        devices.update(|items| {
            upsert_device(items, next);
            save_devices(items);
        });
        device_id.set(String::new());
        label.set(String::new());
        message.set(Some("Device saved locally".to_string()));
    };

    let export_devices = move |_| {
        error.set(None);
        match serde_json::to_string_pretty(&devices.get_untracked()) {
            Ok(serialized) => {
                import_json.set(serialized);
                message.set(Some(
                    "Export copied into the import/export field".to_string(),
                ));
            }
            Err(reason) => error.set(Some(reason.to_string())),
        }
    };

    let import_devices = move |_| {
        error.set(None);
        message.set(None);
        match serde_json::from_str::<Vec<DeviceRecord>>(&import_json.get_untracked()) {
            Ok(mut imported) => {
                imported.sort_by(|left, right| {
                    left.kind
                        .as_str()
                        .cmp(right.kind.as_str())
                        .then(left.id.cmp(&right.id))
                });
                save_devices(&imported);
                devices.set(imported);
                message.set(Some("Imported local device registry".to_string()));
            }
            Err(reason) => error.set(Some(format!("Import JSON failed: {reason}"))),
        }
    };

    view! {
        <div class="page-stub devices-page">
            <section class="card overlay-upload-card">
                <div class="settings-card-head">
                    <div>
                        <h2>"Devices"</h2>
                        <p class="muted">
                            "Local registration for backend-pending modalities. COP drawers and mock surfaces stay hidden until a matching device exists."
                        </p>
                    </div>
                    <span class="tone-badge neutral">"local"</span>
                </div>

                <div class="settings-field-rows">
                    <FormRow label="Kind">
                        <select
                            class="settings-field-input"
                            prop:value=move || selected_kind.get()
                            on:change=move |event| selected_kind.set(event_target_value(&event))
                        >
                            {DeviceKind::ALL.into_iter().map(|kind| view! {
                                <option value=kind.as_str()>{kind.label()}</option>
                            }).collect_view()}
                        </select>
                    </FormRow>
                    <FormRow label="Device id">
                        <input
                            type="text"
                            class="settings-field-input"
                            prop:value=move || device_id.get()
                            on:input=move |event| device_id.set(event_target_value(&event))
                            placeholder="rf-north-1"
                        />
                    </FormRow>
                    <FormRow label="Label">
                        <input
                            type="text"
                            class="settings-field-input"
                            prop:value=move || label.get()
                            on:input=move |event| label.set(event_target_value(&event))
                            placeholder="North ridge RF"
                        />
                    </FormRow>
                    <FormRow label="Latitude">
                        <input type="number" step="0.000001" class="settings-field-input" prop:value=move || lat.get() on:input=move |event| lat.set(event_target_value(&event)) />
                    </FormRow>
                    <FormRow label="Longitude">
                        <input type="number" step="0.000001" class="settings-field-input" prop:value=move || lon.get() on:input=move |event| lon.set(event_target_value(&event)) />
                    </FormRow>
                    <FormRow label="Altitude m">
                        <input type="number" step="0.1" class="settings-field-input" prop:value=move || alt_m.get() on:input=move |event| alt_m.set(event_target_value(&event)) />
                    </FormRow>
                    <FormRow label="Yaw deg">
                        <input type="number" step="0.1" class="settings-field-input" prop:value=move || yaw_deg.get() on:input=move |event| yaw_deg.set(event_target_value(&event)) />
                    </FormRow>
                    <FormRow label="Pitch deg">
                        <input type="number" step="0.1" class="settings-field-input" prop:value=move || pitch_deg.get() on:input=move |event| pitch_deg.set(event_target_value(&event)) />
                    </FormRow>
                </div>

                <div class="overlay-card-actions">
                    <button class="btn-sm btn-primary" on:click=save_current>"Save device"</button>
                    <button class="btn-sm" on:click=export_devices>"Export JSON"</button>
                </div>
                {move || message.get().map(|text| view! { <span class="review-status-ok">{text}</span> })}
                {move || error.get().map(|text| view! { <span class="daily-error">{text}</span> })}
            </section>

            <section class="card overlay-upload-card">
                <div class="settings-card-head">
                    <h2>"Registered Devices"</h2>
                    <span class="tone-badge info">{move || devices.get().len().to_string()}</span>
                </div>
                {move || {
                    let current = devices.get();
                    if current.is_empty() {
                        return view! { <div class="empty-state">"No future-modality devices registered"</div> }.into_any();
                    }
                    view! {
                        <div class="device-list">
                            {current.into_iter().map(|device| {
                                let remove_id = device.id.clone();
                                let position = match (device.lat, device.lon) {
                                    (Some(lat), Some(lon)) => format!("{lat:.5}, {lon:.5}"),
                                    _ => "-".to_string(),
                                };
                                view! {
                                    <article class="device-card">
                                        <div class="device-card-main">
                                            <span class="tone-badge neutral">{device.kind.drawer_label()}</span>
                                            <strong>{device.display_label()}</strong>
                                            <code>{device.id.clone()}</code>
                                        </div>
                                        <div class="device-card-meta">
                                            <span>{position}</span>
                                            <span>{device.orientation_label()}</span>
                                            <span class="mock-watermark">"MOCK - awaiting backend"</span>
                                        </div>
                                        <button
                                            class="btn-sm btn-danger"
                                            on:click=move |_| {
                                                devices.update(|items| {
                                                    items.retain(|item| item.id != remove_id);
                                                    save_devices(items);
                                                });
                                            }
                                        >
                                            "Delete"
                                        </button>
                                    </article>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }}
            </section>

            <section class="card overlay-upload-card">
                <div class="settings-card-head">
                    <h2>"Import / Export"</h2>
                    <span class="tone-badge neutral">"mmp.devices.v1"</span>
                </div>
                <textarea
                    class="settings-field-textarea devices-json-editor"
                    prop:value=move || import_json.get()
                    on:input=move |event| import_json.set(event_target_value(&event))
                    placeholder="Paste exported devices JSON"
                ></textarea>
                <div class="overlay-card-actions">
                    <button class="btn-sm" on:click=import_devices>"Import JSON"</button>
                </div>
            </section>
        </div>
    }
}
