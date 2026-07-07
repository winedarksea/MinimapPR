use super::layout::clamp_dock_width;
use leptos::prelude::*;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::JsCast;

#[component]
pub fn WorkspaceDock(
    side: &'static str,
    width_px: RwSignal<f64>,
    children: Children,
) -> impl IntoView {
    let class_name = format!("workspace-dock workspace-dock-{side}");
    let style = move || {
        format!(
            "width: min({:.0}px, calc(100vw - 2 * var(--mmp-density-gap-sm)));",
            clamp_dock_width(width_px.get())
        )
    };
    let side_for_drag = side.to_string();
    let start_resize = move |event: web_sys::MouseEvent| {
        event.prevent_default();
        let Some(document) = web_sys::window().and_then(|window| window.document()) else {
            return;
        };
        let start_x = event.client_x() as f64;
        let start_width = width_px.get_untracked();
        let drag_side = side_for_drag.clone();

        let move_width = width_px;
        let on_move = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(
            move |move_event: web_sys::MouseEvent| {
                let delta = move_event.client_x() as f64 - start_x;
                let next_width = if drag_side == "left" {
                    start_width + delta
                } else {
                    start_width - delta
                };
                move_width.set(clamp_dock_width(next_width));
            },
        );
        let on_move_ref = on_move.as_ref().unchecked_ref();
        let _ = document.add_event_listener_with_callback("mousemove", on_move_ref);

        let cleanup_document = document.clone();
        let on_move_value = std::rc::Rc::new(on_move);
        let on_move_for_cleanup = on_move_value.clone();
        let on_up = Closure::<dyn FnMut(web_sys::MouseEvent)>::once(move |_| {
            let _ = cleanup_document.remove_event_listener_with_callback(
                "mousemove",
                on_move_for_cleanup.as_ref().as_ref().unchecked_ref(),
            );
        });
        let _ =
            document.add_event_listener_with_callback("mouseup", on_up.as_ref().unchecked_ref());
        on_up.forget();
        std::mem::forget(on_move_value);
    };

    view! {
        <aside class=class_name style=style>
            <button
                type="button"
                class="workspace-dock-resize"
                aria-label=move || format!("Resize {side} dock")
                on:mousedown=start_resize
            ></button>
            {children()}
        </aside>
    }
}
