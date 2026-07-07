use leptos::prelude::*;

#[component]
pub fn WorkspaceDock(side: &'static str, children: Children) -> impl IntoView {
    let class_name = format!("workspace-dock workspace-dock-{side}");
    view! {
        <aside class=class_name>
            {children()}
        </aside>
    }
}
