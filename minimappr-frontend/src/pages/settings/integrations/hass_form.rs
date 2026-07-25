//! Home Assistant MQTT bridge configuration form.
//!
//! Reads come from the nested `hass` block of the config snapshot; writes go out
//! as flat `hass_*` keys via `PATCH /api/v1/config` (matching the `federation`
//! pattern on the backend).

use crate::api::patch_config;
use crate::components::form_row::FormRow;
use crate::state::AppState;
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

/// The backend redacts stored secrets to this in GET responses. Echoing it back
/// on save is guarded here *and* server-side, so neither layer alone can destroy
/// a secret the operator did not retype.
const REDACTED: &str = "***";

fn parse_port(value: &str) -> Result<u32, String> {
    let parsed = value
        .trim()
        .parse::<u32>()
        .map_err(|_| "MQTT port must be a number".to_string())?;
    if (1..=65_535).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("MQTT port must be in [1, 65535]".to_string())
    }
}

fn parse_u32(label: &str, value: &str, min: u32, max: u32) -> Result<u32, String> {
    let parsed = value
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("{label} must be a whole number"))?;
    if (min..=max).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("{label} must be in [{min}, {max}]"))
    }
}

fn parse_f64(label: &str, value: &str, min: f64) -> Result<f64, String> {
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{label} must be a number"))?;
    if parsed >= min {
        Ok(parsed)
    } else {
        Err(format!("{label} must be >= {min}"))
    }
}

fn is_topic_level(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && !trimmed.contains(['+', '#', '/'])
}

#[component]
pub fn HassForm() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let enabled = RwSignal::new(false);
    let base_url = RwSignal::new(String::new());
    let token = RwSignal::new(String::new());
    let mqtt_host = RwSignal::new(String::new());
    let mqtt_port = RwSignal::new("1883".to_string());
    let mqtt_username = RwSignal::new(String::new());
    let mqtt_password = RwSignal::new(String::new());
    let mqtt_client_id = RwSignal::new("minimappr".to_string());
    let mqtt_tls_enabled = RwSignal::new(false);
    let discovery_prefix = RwSignal::new("homeassistant".to_string());
    let base_topic = RwSignal::new("minimappr".to_string());
    let device_id = RwSignal::new("minimappr".to_string());
    let device_name = RwSignal::new("MinimapPR".to_string());
    let publish_interval = RwSignal::new("5".to_string());
    let off_delay = RwSignal::new("30".to_string());
    let track_slot_count = RwSignal::new("8".to_string());
    let publish_zone_occupancy = RwSignal::new(true);
    let publish_zone_spl = RwSignal::new(true);
    let publish_detection_classes = RwSignal::new(true);
    let publish_node_status = RwSignal::new(true);
    let publish_system_health = RwSignal::new(true);
    let publish_events = RwSignal::new(true);
    let publish_track_slots = RwSignal::new(false);
    let save_pending = RwSignal::new(false);
    let message = RwSignal::new(None::<String>);
    let error = RwSignal::new(None::<String>);

    Effect::new(move |_| {
        if let Some(config) = state.config.get() {
            let hass = config.hass;
            enabled.set(hass.enabled);
            base_url.set(hass.base_url);
            token.set(hass.token);
            mqtt_host.set(hass.mqtt_host);
            mqtt_port.set(hass.mqtt_port.to_string());
            mqtt_username.set(hass.mqtt_username);
            mqtt_password.set(hass.mqtt_password);
            mqtt_client_id.set(hass.mqtt_client_id);
            mqtt_tls_enabled.set(hass.mqtt_tls_enabled);
            discovery_prefix.set(hass.discovery_prefix);
            base_topic.set(hass.base_topic);
            device_id.set(hass.device_id);
            device_name.set(hass.device_name);
            publish_interval.set(hass.publish_interval_seconds.to_string());
            off_delay.set(hass.detection_off_delay_seconds.to_string());
            track_slot_count.set(hass.track_slot_count.to_string());
            publish_zone_occupancy.set(hass.publish_zone_occupancy);
            publish_zone_spl.set(hass.publish_zone_spl);
            publish_detection_classes.set(hass.publish_detection_classes);
            publish_node_status.set(hass.publish_node_status);
            publish_system_health.set(hass.publish_system_health);
            publish_events.set(hass.publish_events);
            publish_track_slots.set(hass.publish_track_slots);
        }
    });

    let save = move |_| {
        if save_pending.get_untracked() {
            return;
        }
        error.set(None);
        message.set(None);

        let host = mqtt_host.get_untracked().trim().to_string();
        if enabled.get_untracked() && host.is_empty() {
            error.set(Some(
                "Enabling the bridge requires an MQTT broker host".to_string(),
            ));
            return;
        }
        for (label, value) in [
            ("Discovery prefix", discovery_prefix.get_untracked()),
            ("Base topic", base_topic.get_untracked()),
            ("Device id", device_id.get_untracked()),
        ] {
            if !is_topic_level(&value) {
                error.set(Some(format!(
                    "{label} must be a single non-empty topic level (no +, #, or /)"
                )));
                return;
            }
        }

        let parsed = (|| -> Result<(u32, f64, u32, u32), String> {
            Ok((
                parse_port(&mqtt_port.get_untracked())?,
                parse_f64("Publish interval", &publish_interval.get_untracked(), 1.0)?,
                parse_u32("Detection off delay", &off_delay.get_untracked(), 1, 86_400)?,
                parse_u32("Track slot count", &track_slot_count.get_untracked(), 0, 64)?,
            ))
        })();
        let (parsed_port, parsed_interval, parsed_off_delay, parsed_slots) = match parsed {
            Ok(values) => values,
            Err(reason) => {
                error.set(Some(reason));
                return;
            }
        };

        let mut body = serde_json::Map::new();
        for (key, value) in [
            ("hass_enabled", enabled.get_untracked()),
            ("hass_mqtt_tls_enabled", mqtt_tls_enabled.get_untracked()),
            (
                "hass_publish_zone_occupancy",
                publish_zone_occupancy.get_untracked(),
            ),
            ("hass_publish_zone_spl", publish_zone_spl.get_untracked()),
            (
                "hass_publish_detection_classes",
                publish_detection_classes.get_untracked(),
            ),
            ("hass_publish_node_status", publish_node_status.get_untracked()),
            (
                "hass_publish_system_health",
                publish_system_health.get_untracked(),
            ),
            ("hass_publish_events", publish_events.get_untracked()),
            ("hass_publish_track_slots", publish_track_slots.get_untracked()),
        ] {
            body.insert(key.to_string(), serde_json::Value::Bool(value));
        }

        for (key, value) in [
            ("hass_base_url", base_url.get_untracked().trim().to_string()),
            ("hass_mqtt_host", host),
            (
                "hass_mqtt_username",
                mqtt_username.get_untracked().trim().to_string(),
            ),
            (
                "hass_mqtt_client_id",
                mqtt_client_id.get_untracked().trim().to_string(),
            ),
            (
                "hass_discovery_prefix",
                discovery_prefix.get_untracked().trim().to_string(),
            ),
            (
                "hass_base_topic",
                base_topic.get_untracked().trim().to_string(),
            ),
            ("hass_device_id", device_id.get_untracked().trim().to_string()),
            (
                "hass_device_name",
                device_name.get_untracked().trim().to_string(),
            ),
        ] {
            body.insert(key.to_string(), serde_json::Value::String(value));
        }

        body.insert(
            "hass_mqtt_port".to_string(),
            serde_json::Value::from(parsed_port),
        );
        body.insert(
            "hass_publish_interval_seconds".to_string(),
            serde_json::Value::from(parsed_interval),
        );
        body.insert(
            "hass_detection_off_delay_seconds".to_string(),
            serde_json::Value::from(parsed_off_delay),
        );
        body.insert(
            "hass_track_slot_count".to_string(),
            serde_json::Value::from(parsed_slots),
        );

        // Both secrets: only send when the operator actually typed a new value.
        for (key, signal) in [("hass_token", token), ("hass_mqtt_password", mqtt_password)] {
            let value = signal.get_untracked();
            if value != REDACTED {
                body.insert(key.to_string(), serde_json::Value::String(value));
            }
        }

        let state = state.clone();
        save_pending.set(true);
        spawn_local(async move {
            match patch_config(serde_json::Value::Object(body)).await {
                Ok(snapshot) => {
                    state.config.set(Some(snapshot));
                    message.set(Some("Integration settings saved".to_string()));
                }
                Err(reason) => error.set(Some(reason)),
            }
            save_pending.set(false);
        });
    };

    view! {
        <section class="card integration-card">
            <div class="settings-card-head">
                <div>
                    <h2>"Home Assistant"</h2>
                    <p class="muted">
                        "Publishes zone occupancy, sound levels, node health, and detection/alert events to Home Assistant over MQTT. Discovery is automatic and retained, so there is nothing to configure on the Home Assistant side."
                    </p>
                </div>
            </div>

            <div class="settings-field-rows">
                <label class="integration-enabled-toggle">
                    <input
                        type="checkbox"
                        class="settings-field-input"
                        prop:checked=move || enabled.get()
                        on:change=move |event| enabled.set(event_target_checked(&event))
                    />
                    <span>"Enable Home Assistant bridge"</span>
                </label>
                <FormRow label="MQTT host">
                    <input
                        type="text"
                        class="settings-field-input"
                        placeholder="mqtt.local"
                        prop:value=move || mqtt_host.get()
                        on:input=move |event| mqtt_host.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="MQTT port">
                    <input
                        type="number"
                        class="settings-field-input"
                        min="1"
                        max="65535"
                        prop:value=move || mqtt_port.get()
                        on:input=move |event| mqtt_port.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="MQTT username">
                    <input
                        type="text"
                        class="settings-field-input"
                        placeholder="Leave blank for an anonymous broker"
                        prop:value=move || mqtt_username.get()
                        on:input=move |event| mqtt_username.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="MQTT password">
                    <input
                        type="password"
                        class="settings-field-input"
                        placeholder="Stored server-side; redacted after save"
                        prop:value=move || mqtt_password.get()
                        on:input=move |event| mqtt_password.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="MQTT client id">
                    <input
                        type="text"
                        class="settings-field-input"
                        prop:value=move || mqtt_client_id.get()
                        on:input=move |event| mqtt_client_id.set(event_target_value(&event))
                    />
                </FormRow>
                <label class="integration-enabled-toggle">
                    <input
                        type="checkbox"
                        class="settings-field-input"
                        prop:checked=move || mqtt_tls_enabled.get()
                        on:change=move |event| mqtt_tls_enabled.set(event_target_checked(&event))
                    />
                    <span>"Connect over TLS"</span>
                </label>
                <FormRow label="Discovery prefix">
                    <input
                        type="text"
                        class="settings-field-input"
                        placeholder="homeassistant"
                        prop:value=move || discovery_prefix.get()
                        on:input=move |event| discovery_prefix.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Base topic">
                    <input
                        type="text"
                        class="settings-field-input"
                        placeholder="minimappr"
                        prop:value=move || base_topic.get()
                        on:input=move |event| base_topic.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Device id">
                    <input
                        type="text"
                        class="settings-field-input"
                        placeholder="minimappr"
                        prop:value=move || device_id.get()
                        on:input=move |event| device_id.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Device name">
                    <input
                        type="text"
                        class="settings-field-input"
                        prop:value=move || device_name.get()
                        on:input=move |event| device_name.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Publish interval (s)">
                    <input
                        type="number"
                        class="settings-field-input"
                        min="1"
                        step="0.5"
                        prop:value=move || publish_interval.get()
                        on:input=move |event| publish_interval.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Detection off delay (s)">
                    <input
                        type="number"
                        class="settings-field-input"
                        min="1"
                        prop:value=move || off_delay.get()
                        on:input=move |event| off_delay.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Home Assistant base URL">
                    <input
                        type="url"
                        class="settings-field-input"
                        placeholder="http://homeassistant.local:8123"
                        prop:value=move || base_url.get()
                        on:input=move |event| base_url.set(event_target_value(&event))
                    />
                </FormRow>
                <FormRow label="Long-lived access token">
                    <input
                        type="password"
                        class="settings-field-input"
                        placeholder="Stored server-side; redacted after save"
                        prop:value=move || token.get()
                        on:input=move |event| token.set(event_target_value(&event))
                    />
                </FormRow>
                <p class="muted">
                    "The base URL and access token are used by the not-yet-implemented inbound enrichment client. The MQTT bridge does not read them."
                </p>
            </div>

            <h3>"Published entities"</h3>
            <div class="settings-field-rows">
                <EntityToggle signal=publish_zone_occupancy label="Zone occupancy (one binary_sensor per zone)" />
                <EntityToggle signal=publish_zone_spl label="Zone sound level (one sensor per zone)" />
                <EntityToggle signal=publish_detection_classes label="Detection classes (impulse binary_sensor)" />
                <EntityToggle signal=publish_node_status label="Node connectivity (diagnostic binary_sensor)" />
                <EntityToggle signal=publish_system_health label="System health and active track count" />
                <EntityToggle signal=publish_events label="Detection and alert event entities" />
                <EntityToggle signal=publish_track_slots label="Track slots (device_tracker pool)" />
                {move || {
                    publish_track_slots.get().then(|| view! {
                        <FormRow label="Track slot count">
                            <input
                                type="number"
                                class="settings-field-input"
                                min="0"
                                max="64"
                                prop:value=move || track_slot_count.get()
                                on:input=move |event| track_slot_count.set(event_target_value(&event))
                            />
                        </FormRow>
                    })
                }}
                {move || {
                    publish_track_slots.get().then(|| view! {
                        <p class="muted">
                            "Home Assistant keeps every entity id it has ever seen, so tracks map onto a fixed pool of slots rather than one entity per track. Slots fill highest-quality-first and stay with a track for its lifetime."
                        </p>
                    })
                }}
            </div>

            <div class="overlay-card-actions">
                <button class="btn-sm btn-primary" disabled=move || save_pending.get() on:click=save>
                    {move || if save_pending.get() { "Saving" } else { "Save integration" }}
                </button>
            </div>
            {move || message.get().map(|text| view! { <span class="review-status-ok">{text}</span> })}
            {move || error.get().map(|text| view! { <span class="daily-error">{text}</span> })}
        </section>
    }
}

#[component]
fn EntityToggle(signal: RwSignal<bool>, label: &'static str) -> impl IntoView {
    view! {
        <label class="integration-enabled-toggle">
            <input
                type="checkbox"
                class="settings-field-input"
                prop:checked=move || signal.get()
                on:change=move |event| signal.set(event_target_checked(&event))
            />
            <span>{label}</span>
        </label>
    }
}
