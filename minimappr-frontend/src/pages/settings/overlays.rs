use crate::api::{
    delete_overlay, list_overlays, patch_overlay, upload_overlay, MapOverlay, MapOverlayUpdate,
};
use crate::components::form_row::FormRow;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;

fn parse_bounds_json(value: &str) -> Result<Vec<Vec<f64>>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let parsed = serde_json::from_str::<Vec<Vec<f64>>>(trimmed)
        .map_err(|error| format!("Bounds JSON is invalid: {error}"))?;
    if parsed.len() != 4 {
        return Err("Bounds must contain four [lat, lon] corners".to_string());
    }
    for corner in &parsed {
        if corner.len() != 2 {
            return Err("Each bounds corner must be [lat, lon]".to_string());
        }
        if !(-90.0..=90.0).contains(&corner[0]) || !(-180.0..=180.0).contains(&corner[1]) {
            return Err("Bounds contain invalid latitude or longitude".to_string());
        }
    }
    Ok(parsed)
}

fn format_bounds(bounds: &[Vec<f64>]) -> String {
    if bounds.is_empty() {
        String::new()
    } else {
        serde_json::to_string(bounds).unwrap_or_default()
    }
}

#[component]
pub fn OverlaysView() -> impl IntoView {
    let overlays: RwSignal<Vec<MapOverlay>> = RwSignal::new(Vec::new());
    let loading = RwSignal::new(true);
    let uploading = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let notice: RwSignal<Option<String>> = RwSignal::new(None);

    let upload_name = RwSignal::new(String::new());
    let upload_kind = RwSignal::new("image".to_string());
    let upload_opacity = RwSignal::new("0.75".to_string());
    let upload_storey = RwSignal::new(String::new());
    let upload_bounds = RwSignal::new(String::new());
    let upload_file: RwSignal<Option<web_sys::File>> = RwSignal::new(None);

    let reload = move || {
        loading.set(true);
        error.set(None);
        spawn_local(async move {
            match list_overlays().await {
                Ok(items) => overlays.set(items),
                Err(message) => error.set(Some(message)),
            }
            loading.set(false);
        });
    };
    reload();

    let on_upload = move |_| {
        error.set(None);
        notice.set(None);
        let Some(file) = upload_file.get_untracked() else {
            error.set(Some("Choose an overlay file first".to_string()));
            return;
        };
        let name = upload_name.get_untracked();
        if name.trim().is_empty() {
            error.set(Some("Overlay name is required".to_string()));
            return;
        }
        if let Err(message) = parse_bounds_json(&upload_bounds.get_untracked()) {
            error.set(Some(message));
            return;
        }
        let opacity = upload_opacity
            .get_untracked()
            .trim()
            .parse::<f64>()
            .unwrap_or(0.75)
            .clamp(0.0, 1.0);
        uploading.set(true);
        let kind = upload_kind.get_untracked();
        let bounds = upload_bounds.get_untracked();
        let storey = upload_storey.get_untracked();
        spawn_local(async move {
            match upload_overlay(file, name.trim(), &kind, &bounds, opacity, &storey).await {
                Ok(overlay) => {
                    overlays.update(|items| items.insert(0, overlay));
                    upload_name.set(String::new());
                    upload_bounds.set(String::new());
                    upload_storey.set(String::new());
                    upload_file.set(None);
                    notice.set(Some("Overlay uploaded".to_string()));
                }
                Err(message) => error.set(Some(message)),
            }
            uploading.set(false);
        });
    };

    view! {
        <div class="page-stub overlays-page">
            <div class="rules-header">
                <div>
                    <h2 style="margin:0">"Map Overlays"</h2>
                    <p class="muted" style="font-size:0.85rem">
                        "Import floorplans, site rasters, SVGs, or GeoJSON for COP map context."
                    </p>
                </div>
                <button class="btn-sm" disabled=move || loading.get() on:click=move |_| reload()>
                    "Reload"
                </button>
            </div>

            <section class="diag-card overlay-upload-card">
                <div class="pipeline-section-label">"Upload overlay"</div>
                <div class="settings-field-rows">
                    <FormRow label="Name">
                        <input
                            type="text"
                            class="settings-field-input"
                            placeholder="Barn floorplan"
                            prop:value=move || upload_name.get()
                            on:input=move |event| upload_name.set(event_target_value(&event))
                        />
                    </FormRow>
                    <FormRow label="Kind">
                        <select
                            class="settings-field-input"
                            prop:value=move || upload_kind.get()
                            on:change=move |event| upload_kind.set(event_target_value(&event))
                        >
                            <option value="image">"Image"</option>
                            <option value="svg">"SVG"</option>
                            <option value="geojson">"GeoJSON"</option>
                        </select>
                    </FormRow>
                    <FormRow label="Opacity">
                        <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.05"
                            class="settings-field-input"
                            prop:value=move || upload_opacity.get()
                            on:input=move |event| upload_opacity.set(event_target_value(&event))
                        />
                    </FormRow>
                    <FormRow label="Storey">
                        <input
                            type="text"
                            class="settings-field-input"
                            placeholder="L1"
                            prop:value=move || upload_storey.get()
                            on:input=move |event| upload_storey.set(event_target_value(&event))
                        />
                    </FormRow>
                    <FormRow label="File">
                        <input
                            type="file"
                            class="settings-field-input overlay-file-input"
                            accept=".png,.jpg,.jpeg,.webp,.svg,.json,.geojson"
                            on:change=move |event| {
                                let input = event
                                    .target()
                                    .and_then(|target| target.dyn_into::<HtmlInputElement>().ok());
                                let file = input.and_then(|input| input.files()).and_then(|files| files.get(0));
                                upload_file.set(file);
                            }
                        />
                    </FormRow>
                </div>
                <label class="overlay-bounds-label">"Bounds JSON"
                    <textarea
                        class="settings-field-textarea overlay-bounds-editor"
                        placeholder="[[44.0,-93.0],[44.0,-92.9],[43.9,-92.9],[43.9,-93.0]]"
                        prop:value=move || upload_bounds.get()
                        on:input=move |event| upload_bounds.set(event_target_value(&event))
                    />
                </label>
                <div class="pipeline-save-row">
                    <button class="btn-primary" disabled=move || uploading.get() on:click=on_upload>
                        {move || if uploading.get() { "Uploading..." } else { "Upload overlay" }}
                    </button>
                    {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
                    {move || notice.get().map(|message| view! { <span class="tone-badge ok">{message}</span> })}
                </div>
            </section>

            {move || if loading.get() {
                view! { <div class="empty-state"><p>"Loading overlays..."</p></div> }.into_any()
            } else if overlays.get().is_empty() {
                view! { <div class="empty-state"><p>"No overlays imported yet."</p></div> }.into_any()
            } else {
                view! {
                    <div class="overlay-list">
                        {move || overlays.get().into_iter().map(|overlay| {
                            view! { <OverlayCard overlay overlays /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn OverlayCard(overlay: MapOverlay, overlays: RwSignal<Vec<MapOverlay>>) -> impl IntoView {
    let overlay_id = overlay.id.clone();
    let opacity_text = RwSignal::new(overlay.opacity.to_string());
    let bounds_text = RwSignal::new(format_bounds(&overlay.bounds));
    let saving = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);

    let patch_enabled = {
        let overlay_id = overlay_id.clone();
        move |enabled: bool| {
            saving.set(true);
            error.set(None);
            let id = overlay_id.clone();
            spawn_local(async move {
                match patch_overlay(
                    &id,
                    MapOverlayUpdate {
                        enabled: Some(enabled),
                        ..MapOverlayUpdate::default()
                    },
                )
                .await
                {
                    Ok(updated) => overlays.update(|items| {
                        if let Some(existing) = items.iter_mut().find(|item| item.id == updated.id)
                        {
                            *existing = updated;
                        }
                    }),
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    let save_metadata = {
        let overlay_id = overlay_id.clone();
        move |_| {
            error.set(None);
            let opacity = opacity_text
                .get_untracked()
                .trim()
                .parse::<f64>()
                .unwrap_or(0.75)
                .clamp(0.0, 1.0);
            let bounds = match parse_bounds_json(&bounds_text.get_untracked()) {
                Ok(value) => value,
                Err(message) => {
                    error.set(Some(message));
                    return;
                }
            };
            saving.set(true);
            let id = overlay_id.clone();
            spawn_local(async move {
                match patch_overlay(
                    &id,
                    MapOverlayUpdate {
                        opacity: Some(opacity),
                        bounds: Some(bounds),
                        ..MapOverlayUpdate::default()
                    },
                )
                .await
                {
                    Ok(updated) => overlays.update(|items| {
                        if let Some(existing) = items.iter_mut().find(|item| item.id == updated.id)
                        {
                            *existing = updated;
                        }
                    }),
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    let remove_overlay = {
        let overlay_id = overlay_id.clone();
        move |_| {
            saving.set(true);
            error.set(None);
            let id = overlay_id.clone();
            spawn_local(async move {
                match delete_overlay(&id).await {
                    Ok(()) => overlays.update(|items| items.retain(|item| item.id != id)),
                    Err(message) => error.set(Some(message)),
                }
                saving.set(false);
            });
        }
    };

    view! {
        <section class="diag-card overlay-card">
            <div class="overlay-card-header">
                <div>
                    <div class="pipeline-section-label">{overlay.name.clone()}</div>
                    <div class="overlay-meta-row">
                        <span class="tone-badge neutral">{overlay.kind.clone()}</span>
                        <span class="muted">{overlay.mime.clone()}</span>
                        {overlay.storey.clone().map(|storey| view! {
                            <span class="tone-badge info">{storey}</span>
                        })}
                    </div>
                </div>
                <div class="overlay-card-actions">
                    <a class="btn-sm" href=overlay.content_url.clone() target="_blank">"Open"</a>
                    <button class="btn-sm btn-sm--danger" disabled=move || saving.get() on:click=remove_overlay>
                        "Delete"
                    </button>
                </div>
            </div>

            <div class="settings-field-rows">
                <label class="rule-enabled-toggle overlay-enabled-toggle">
                    <input
                        type="checkbox"
                        class="settings-field-input"
                        prop:checked=overlay.enabled
                        on:change=move |event| patch_enabled(event_target_checked(&event))
                    />
                    <span>"Enabled"</span>
                </label>
                <FormRow label="Opacity">
                    <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        class="settings-field-input"
                        prop:value=move || opacity_text.get()
                        on:input=move |event| opacity_text.set(event_target_value(&event))
                    />
                </FormRow>
                <label class="overlay-bounds-label">"Bounds"
                    <textarea
                        class="settings-field-textarea overlay-bounds-editor overlay-bounds-editor--compact"
                        prop:value=move || bounds_text.get()
                        on:input=move |event| bounds_text.set(event_target_value(&event))
                    />
                </label>
            </div>
            <div class="pipeline-save-row">
                <button class="btn-sm" disabled=move || saving.get() on:click=save_metadata>
                    "Save geometry"
                </button>
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
            </div>
        </section>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bounds_accepts_empty_or_four_corners() {
        assert!(parse_bounds_json("").unwrap().is_empty());
        assert_eq!(
            parse_bounds_json("[[44,-93],[44,-92.9],[43.9,-92.9],[43.9,-93]]")
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn parse_bounds_rejects_bad_shapes() {
        assert!(parse_bounds_json("[[44,-93]]").is_err());
        assert!(parse_bounds_json("[[144,-93],[44,-92.9],[43.9,-92.9],[43.9,-93]]").is_err());
    }
}
