use crate::api::patch_config;
use crate::state::AppState;
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

fn parse_mqtt_port(value: &str) -> Result<u32, String> {
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

#[component]
pub fn IntegrationsView() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let enabled = RwSignal::new(false);
    let base_url = RwSignal::new(String::new());
    let token = RwSignal::new(String::new());
    let mqtt_host = RwSignal::new(String::new());
    let mqtt_port = RwSignal::new("1883".to_string());
    let save_pending = RwSignal::new(false);
    let message = RwSignal::new(None::<String>);
    let error = RwSignal::new(None::<String>);

    Effect::new(move |_| {
        if let Some(config) = state.config.get() {
            enabled.set(config.hass.enabled);
            base_url.set(config.hass.base_url);
            token.set(config.hass.token);
            mqtt_host.set(config.hass.mqtt_host);
            mqtt_port.set(config.hass.mqtt_port.to_string());
        }
    });

    let save = move |_| {
        if save_pending.get_untracked() {
            return;
        }
        error.set(None);
        message.set(None);
        let parsed_port = match parse_mqtt_port(&mqtt_port.get_untracked()) {
            Ok(value) => value,
            Err(reason) => {
                error.set(Some(reason));
                return;
            }
        };
        let mut body = serde_json::Map::new();
        body.insert(
            "hass_enabled".to_string(),
            serde_json::Value::Bool(enabled.get_untracked()),
        );
        body.insert(
            "hass_base_url".to_string(),
            serde_json::Value::String(base_url.get_untracked().trim().to_string()),
        );
        body.insert(
            "hass_mqtt_host".to_string(),
            serde_json::Value::String(mqtt_host.get_untracked().trim().to_string()),
        );
        body.insert(
            "hass_mqtt_port".to_string(),
            serde_json::Value::from(parsed_port),
        );
        let token_value = token.get_untracked();
        if token_value != "***" {
            body.insert(
                "hass_token".to_string(),
                serde_json::Value::String(token_value),
            );
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
        <div class="settings-page integrations-page">
            <section class="card integration-card">
                <div class="card-head">
                    <div>
                        <h2>"Home Assistant"</h2>
                        <p class="muted">
                            "Placeholder configuration for the MQTT/webhook bridge. The backend stores these values now; live entity publishing remains backend pending."
                        </p>
                    </div>
                    <span class="tone-badge neutral">"backend pending"</span>
                </div>

                <div class="settings-form-grid integrations-grid">
                    <label class="integration-enabled-toggle">
                        <input
                            type="checkbox"
                            prop:checked=move || enabled.get()
                            on:change=move |event| enabled.set(event_target_checked(&event))
                        />
                        <span>"Enable Home Assistant bridge"</span>
                    </label>
                    <label>"Base URL"
                        <input
                            type="url"
                            placeholder="http://homeassistant.local:8123"
                            prop:value=move || base_url.get()
                            on:input=move |event| base_url.set(event_target_value(&event))
                        />
                    </label>
                    <label>"Long-lived access token"
                        <input
                            type="password"
                            placeholder="Stored server-side; redacted after save"
                            prop:value=move || token.get()
                            on:input=move |event| token.set(event_target_value(&event))
                        />
                    </label>
                    <label>"MQTT host"
                        <input
                            type="text"
                            placeholder="mqtt.local"
                            prop:value=move || mqtt_host.get()
                            on:input=move |event| mqtt_host.set(event_target_value(&event))
                        />
                    </label>
                    <label>"MQTT port"
                        <input
                            type="number"
                            min="1"
                            max="65535"
                            prop:value=move || mqtt_port.get()
                            on:input=move |event| mqtt_port.set(event_target_value(&event))
                        />
                    </label>
                </div>

                <div class="overlay-card-actions">
                    <button
                        class="btn-sm btn-primary"
                        disabled=move || save_pending.get()
                        on:click=save
                    >
                        {move || if save_pending.get() { "Saving" } else { "Save integration" }}
                    </button>
                    <button class="btn-sm" disabled=true title="Backend connection test pending">
                        "Test connection"
                    </button>
                </div>
                {move || message.get().map(|text| view! { <span class="review-status-ok">{text}</span> })}
                {move || error.get().map(|text| view! { <span class="daily-error">{text}</span> })}
            </section>

            <section class="card integration-card">
                <div class="card-head">
                    <h2>"Planned Bridge Outputs"</h2>
                    <span class="mock-watermark">"MOCK - awaiting backend"</span>
                </div>
                <div class="integration-output-grid">
                    <div><strong>"Semantic sensors"</strong><span>"Zone occupancy, SPL, node health, and active track counts."</span></div>
                    <div><strong>"Priority alerts"</strong><span>"Rule-triggered notifications with detection and track context."</span></div>
                    <div><strong>"Device bus"</strong><span>"Future bidirectional state sync for physical devices and effectors."</span></div>
                </div>
            </section>
        </div>
    }
}
