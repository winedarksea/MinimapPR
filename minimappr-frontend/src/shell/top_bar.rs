use crate::prefs;
use crate::state::{AppState, WsStatus};
use leptos::prelude::*;
use leptos_router::components::A;
use leptos_router::hooks::use_location;

#[component]
pub fn TopBar() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let ws_status = state.ws_status;
    let theme = state.theme;

    let pill_class = move || match ws_status.get() {
        WsStatus::Connected => "ws-pill connected",
        WsStatus::Reconnecting => "ws-pill reconnecting",
        WsStatus::Disconnected => "ws-pill disconnected",
    };
    let pill_text = move || match ws_status.get() {
        WsStatus::Connected => "Live",
        WsStatus::Reconnecting => "Reconnect…",
        WsStatus::Disconnected => "Offline",
    };

    // Periodic clock tick — MD3 top bar shows UTC + local, tabular-nums mono.
    let now_ms = RwSignal::new(js_sys::Date::now());
    leptos::task::spawn_local(async move {
        let mut stream = gloo_timers::future::IntervalStream::new(1_000);
        use futures::StreamExt;
        while stream.next().await.is_some() {
            now_ms.set(js_sys::Date::now());
        }
    });

    let clock_utc = move || format_hms_utc(now_ms.get());
    let clock_local = move || format_hms_local(now_ms.get());

    let on_theme_click = move |_| {
        let next = prefs::toggle_theme();
        theme.set(next);
    };
    let theme_icon = move || {
        if theme.get() == "light" {
            "dark_mode"
        } else {
            "light_mode"
        }
    };
    let theme_title = move || {
        if theme.get() == "light" {
            "Switch to dark theme"
        } else {
            "Switch to light theme"
        }
    };
    view! {
        <header class="topbar">
            <span class="topbar-brand">
                "MinimapPR"
                <span class="topbar-brand-tag">"COP"</span>
            </span>

            <nav class="topbar-nav" aria-label="Primary">
                <NavLink href="/cop"      label="COP"      icon="radar" />
                <NavLink href="/analysis" label="Analysis" icon="analytics" />
                <NavLink href="/audio"    label="Audio"    icon="graphic_eq" />
                <NavLink href="/settings" label="Settings" icon="settings" />
            </nav>

            <span class="topbar-spacer"></span>

            <div class="topbar-tools">
                <span class=pill_class>{pill_text}</span>
                <div class="topbar-clock" aria-label="Clock">
                    <div class="utc">{clock_utc}" UTC"</div>
                    <div class="local">{clock_local}" local"</div>
                </div>
                <button
                    class="topbar-iconbtn"
                    title=theme_title
                    aria-label=theme_title
                    on:click=on_theme_click
                >
                    <span class="material-symbols-rounded">{theme_icon}</span>
                </button>
            </div>
        </header>
    }
}

#[component]
fn NavLink(href: &'static str, label: &'static str, icon: &'static str) -> impl IntoView {
    let loc = use_location();
    let href_owned = href.to_string();
    let is_active = move || {
        let path = loc.pathname.get();
        // Active if the route prefix matches — handles nested routes.
        if href_owned == "/cop" {
            path == "/" || path.starts_with(href_owned.as_str())
        } else {
            path.starts_with(href_owned.as_str())
        }
    };
    view! {
        <A
            href=href
            attr:class=move || if is_active() { "active" } else { "" }
        >
            <span class="material-symbols-rounded">{icon}</span>
            {label}
        </A>
    }
}

fn format_hms_utc(ms: f64) -> String {
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    format!(
        "{:02}:{:02}:{:02}",
        d.get_utc_hours(),
        d.get_utc_minutes(),
        d.get_utc_seconds()
    )
}

fn format_hms_local(ms: f64) -> String {
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
    format!(
        "{:02}:{:02}:{:02}",
        d.get_hours(),
        d.get_minutes(),
        d.get_seconds()
    )
}
