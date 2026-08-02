pub mod daily;
pub mod heatmap;
pub mod labels;

use gloo_net::http::Request;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::{Outlet, A};
use leptos_router::hooks::use_location;
use serde::Deserialize;

/// Shared analysis-page filter state, provided via context so Daily/Labels/Heatmap
/// all read+write the same selection as the user moves between subnav tabs.
#[derive(Clone, Copy)]
pub struct AnalysisFilters {
    /// `None` = all classifiers.
    pub classifier: RwSignal<Option<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct ClassifierCount {
    name: String,
    #[allow(dead_code)]
    count: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct ClassifiersResponse {
    classifiers: Vec<ClassifierCount>,
}

#[component]
pub fn AnalysisLayout() -> impl IntoView {
    let classifier = RwSignal::new(None::<String>);
    provide_context(AnalysisFilters { classifier });

    let classifiers: RwSignal<Vec<ClassifierCount>> = RwSignal::new(Vec::new());
    Effect::new(move |_| {
        spawn_local(async move {
            if let Ok(resp) = Request::get("/api/v1/analytics/classifiers").send().await {
                if resp.ok() {
                    if let Ok(d) = resp.json::<ClassifiersResponse>().await {
                        classifiers.set(d.classifiers);
                    }
                }
            }
        });
    });

    view! {
        <div class="app-page">
            <nav class="subnav" aria-label="Analysis sections">
                <SubNavLink href="/analysis/daily"   label="Daily" />
                <SubNavLink href="/analysis/labels"  label="Labels" />
                <SubNavLink href="/analysis/heatmap" label="Heatmap" />
            </nav>
            <div class="classifier-filter-row" role="group" aria-label="Filter by classifier">
                <span class="classifier-filter-label">"Classifier"</span>
                <ClassifierChip classifier=classifier value=None label="All".to_string() />
                {move || classifiers.get().into_iter().map(|c| {
                    view! {
                        <ClassifierChip classifier=classifier value=Some(c.name.clone()) label=c.name.clone() />
                    }
                }).collect_view()}
            </div>
            <div class="page-content">
                <Outlet />
            </div>
        </div>
    }
}

#[component]
fn ClassifierChip(
    classifier: RwSignal<Option<String>>,
    value: Option<String>,
    label: String,
) -> impl IntoView {
    let is_active = {
        let value = value.clone();
        move || classifier.get() == value
    };
    let is_active_aria = {
        let value = value.clone();
        move || (classifier.get() == value).to_string()
    };
    view! {
        <button
            type="button"
            class="btn-sm classifier-chip"
            class:active=is_active
            aria-pressed=is_active_aria
            on:click=move |_| classifier.set(value.clone())
        >{label}</button>
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
