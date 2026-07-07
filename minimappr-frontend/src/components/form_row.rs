use leptos::prelude::*;

#[component]
pub fn FormRow(label: &'static str, children: Children) -> impl IntoView {
    view! {
        <div class="settings-field-row">
            <label>{label}</label>
            {children()}
        </div>
    }
}
