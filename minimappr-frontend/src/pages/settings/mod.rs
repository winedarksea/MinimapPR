pub mod config;
pub mod integrations;
pub mod logs;
pub mod nodes;
pub mod overlays;
pub mod rules;
pub mod server;

use crate::state::AppState;
use leptos::prelude::*;
use leptos_router::components::{Outlet, A};
use leptos_router::hooks::use_location;

#[component]
pub fn SettingsLayout() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    view! {
        <div class="app-page">
            <nav class="subnav" aria-label="Settings sections">
                <SubNavLink href="/settings/config"   label="Config" />
                <SubNavLink href="/settings/nodes"    label="Nodes" />
                <SubNavLink href="/settings/rules"    label="Rules" />
                <SubNavLink href="/settings/overlays" label="Overlays" />
                <SubNavLink href="/settings/integrations" label="Integrations" />
                <SubNavLink href="/settings/server"   label="Server" />
                {move || (!nodes.get().is_empty()).then(|| view! {
                    <span class="tone-badge tone-blue" style="margin-left:-0.25rem">
                        {nodes.get().len().to_string()}
                    </span>
                })}
                <SubNavLink href="/settings/logs"     label="Logs" />
            </nav>
            <div class="page-content">
                <Outlet />
            </div>
        </div>
    }
}

#[component]
fn SubNavLink(href: &'static str, label: &'static str) -> impl IntoView {
    let loc = use_location();
    let href_owned = href.to_string();
    let is_active = move || loc.pathname.get().starts_with(href_owned.as_str());
    view! {
        <A href=href attr:class=move || if is_active() { "active" } else { "" }>
            {label}
        </A>
    }
}
