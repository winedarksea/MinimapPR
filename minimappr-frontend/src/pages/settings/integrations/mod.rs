//! Settings → Integrations. One page, two cards: the Home Assistant MQTT bridge
//! configuration form and its live connection/counter panel.
//!
//! Split into a directory module because the single-file version outgrew a
//! comfortable read once the bridge became real (AGENTS §1.1).

mod hass_form;
mod hass_status;

use hass_form::HassForm;
use hass_status::HassStatusCard;
use leptos::prelude::*;

#[component]
pub fn IntegrationsView() -> impl IntoView {
    view! {
        <div class="page-stub integrations-page">
            <HassForm />
            <HassStatusCard />
        </div>
    }
}
