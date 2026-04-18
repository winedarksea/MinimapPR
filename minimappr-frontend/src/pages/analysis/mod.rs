pub mod daily;
pub mod labels;
pub mod heatmap;

use leptos::prelude::*;
use leptos_router::components::{A, Outlet};
use leptos_router::hooks::use_location;

#[component]
pub fn AnalysisLayout() -> impl IntoView {
    view! {
        <div class="app-page">
            <nav class="subnav" aria-label="Analysis sections">
                <SubNavLink href="/analysis/daily"   label="Daily" />
                <SubNavLink href="/analysis/labels"  label="Labels" />
                <SubNavLink href="/analysis/heatmap" label="Heatmap" />
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
