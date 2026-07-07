use leptos::prelude::*;

#[component]
pub fn DrawerShell(
    title: &'static str,
    icon: &'static str,
    open: RwSignal<bool>,
    badge: Signal<usize>,
    children: Children,
) -> impl IntoView {
    view! {
        <section class=move || {
            if open.get() {
                "workspace-drawer"
            } else {
                "workspace-drawer workspace-drawer-collapsed"
            }
        }>
            <button
                type="button"
                class="workspace-drawer-header"
                aria-expanded=move || open.get().to_string()
                on:click=move |_| open.update(|is_open| *is_open = !*is_open)
            >
                <span class="material-symbols-rounded workspace-drawer-icon" aria-hidden="true">{icon}</span>
                <span class="workspace-drawer-title">{title}</span>
                <span class="workspace-drawer-badge" class:zero=move || badge.get() == 0>
                    {move || badge.get().to_string()}
                </span>
                <span class="material-symbols-rounded workspace-drawer-toggle" aria-hidden="true">
                    {move || if open.get() { "expand_less" } else { "expand_more" }}
                </span>
            </button>
            <div class="workspace-drawer-body">
                {children()}
            </div>
        </section>
    }
}
