use crate::api::{get_rules, put_rules, RuleAction, RuleCondition, RuleModel};
use crate::components::form_row::FormRow;
use leptos::prelude::*;
use leptos::task::spawn_local;

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn join_csv(values: &[String]) -> String {
    values.join(", ")
}

fn pretty_rules_payload(rules: &[RuleModel]) -> String {
    serde_json::to_string_pretty(&serde_json::json!({ "rules": rules }))
        .unwrap_or_else(|_| "{\"rules\":[]}".to_string())
}

fn pretty_payload(payload: &serde_json::Value) -> String {
    serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string())
}

fn next_rule_id(existing_rules: &[RuleModel]) -> String {
    let mut index = existing_rules.len() + 1;
    loop {
        let candidate = format!("rule_{index}");
        if !existing_rules.iter().any(|rule| rule.id == candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn parse_raw_rules_config(raw_json: &str) -> Result<Vec<RuleModel>, String> {
    let value = serde_json::from_str::<serde_json::Value>(raw_json)
        .map_err(|error| format!("Rules JSON is invalid: {error}"))?;
    let rules_value = if value.is_array() {
        value
    } else {
        value.get("rules").cloned().ok_or_else(|| {
            "Rules JSON must be an array or an object with a rules array".to_string()
        })?
    };
    let rules = serde_json::from_value::<Vec<RuleModel>>(rules_value)
        .map_err(|error| format!("Rules JSON does not match the rules schema: {error}"))?;
    validate_and_normalize_rules(&rules)
}

fn validate_and_normalize_rules(input_rules: &[RuleModel]) -> Result<Vec<RuleModel>, String> {
    let mut normalized = Vec::with_capacity(input_rules.len());
    for (index, input_rule) in input_rules.iter().enumerate() {
        let mut rule = input_rule.clone();
        rule.id = rule.id.trim().to_string();
        if rule.id.is_empty() {
            return Err(format!("Rule {} needs an id", index + 1));
        }
        rule.scope = rule.scope.trim().to_lowercase();
        if rule.scope != "detection" && rule.scope != "track" {
            return Err(format!("Rule {} scope must be detection or track", rule.id));
        }
        if rule.cooldown_seconds < 0.0 {
            return Err(format!("Rule {} cooldown cannot be negative", rule.id));
        }
        if let Some(min_confidence) = rule.when.min_confidence {
            if !(0.0..=1.0).contains(&min_confidence) {
                return Err(format!(
                    "Rule {} minimum confidence must be between 0 and 1",
                    rule.id
                ));
            }
        }
        if rule.actions.is_empty() {
            return Err(format!("Rule {} needs at least one action", rule.id));
        }
        for action in &mut rule.actions {
            action.action_type = action.action_type.trim().to_lowercase();
            if action.action_type.is_empty() {
                action.action_type = "alert".to_string();
            }
            action.destination = action.destination.trim().to_lowercase();
            if action.destination.is_empty() {
                action.destination = "cop".to_string();
            }
            action.priority = action.priority.trim().to_lowercase();
            if action.priority.is_empty() {
                action.priority = "normal".to_string();
            }
            if !action.payload.is_object() {
                return Err(format!(
                    "Rule {} action payload must be a JSON object",
                    rule.id
                ));
            }
        }
        normalized.push(rule);
    }
    Ok(normalized)
}

#[component]
pub fn RulesView() -> impl IntoView {
    let rules = RwSignal::new(Vec::<RuleModel>::new());
    let source = RwSignal::new("loading".to_string());
    let path = RwSignal::new(String::new());
    let raw_payload = RwSignal::new("{\"rules\":[]}".to_string());
    let loading = RwSignal::new(true);
    let saving = RwSignal::new(false);
    let error: RwSignal<Option<String>> = RwSignal::new(None);
    let notice: RwSignal<Option<String>> = RwSignal::new(None);

    let load_rules = move || {
        loading.set(true);
        error.set(None);
        notice.set(None);
        spawn_local(async move {
            match get_rules().await {
                Ok(response) => {
                    raw_payload.set(pretty_rules_payload(&response.rules));
                    path.set(response.path);
                    source.set(response.source);
                    rules.set(response.rules);
                }
                Err(message) => error.set(Some(message)),
            }
            loading.set(false);
        });
    };

    load_rules();

    let save_structured = move |_| {
        error.set(None);
        notice.set(None);
        let next_rules = match rules.with(|items| validate_and_normalize_rules(items)) {
            Ok(items) => items,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        saving.set(true);
        spawn_local(async move {
            match put_rules(next_rules).await {
                Ok(response) => {
                    raw_payload.set(pretty_rules_payload(&response.rules));
                    path.set(response.path);
                    source.set(response.source);
                    rules.set(response.rules);
                    notice.set(Some("Rules saved".to_string()));
                }
                Err(message) => error.set(Some(message)),
            }
            saving.set(false);
        });
    };

    let save_raw = move |_| {
        error.set(None);
        notice.set(None);
        let next_rules = match parse_raw_rules_config(&raw_payload.get_untracked()) {
            Ok(items) => items,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        saving.set(true);
        spawn_local(async move {
            match put_rules(next_rules).await {
                Ok(response) => {
                    raw_payload.set(pretty_rules_payload(&response.rules));
                    path.set(response.path);
                    source.set(response.source);
                    rules.set(response.rules);
                    notice.set(Some("Rules saved from raw JSON".to_string()));
                }
                Err(message) => error.set(Some(message)),
            }
            saving.set(false);
        });
    };

    let add_rule = move |_| {
        rules.update(|items| {
            let mut rule = RuleModel {
                id: next_rule_id(items),
                ..RuleModel::default()
            };
            rule.when = RuleCondition {
                min_confidence: Some(0.5),
                ..RuleCondition::default()
            };
            items.push(rule);
        });
    };

    let refresh_raw_from_form = move |_| {
        raw_payload.set(rules.with(|items| pretty_rules_payload(items)));
    };

    view! {
        <div class="page-stub rules-page">
            <div class="rules-header">
                <div>
                    <h2 style="margin:0">"Rules"</h2>
                    <p class="muted" style="font-size:0.85rem">
                        "Configure detection and track automation without touching the audio path."
                    </p>
                </div>
                <div class="rules-meta">
                    <span class="tone-badge neutral">{move || source.get()}</span>
                    <span class="muted rules-path">{move || path.get()}</span>
                </div>
            </div>

            <div class="pipeline-save-row rules-toolbar">
                <button class="btn-primary" disabled=move || loading.get() || saving.get() on:click=save_structured>
                    {move || if saving.get() { "Saving..." } else { "Save rules" }}
                </button>
                <button class="btn-sm" disabled=move || loading.get() || saving.get() on:click=add_rule>
                    "+ Add rule"
                </button>
                <button class="btn-sm" disabled=move || loading.get() || saving.get() on:click=move |_| load_rules()>
                    "Reload"
                </button>
                {move || error.get().map(|message| view! { <span class="daily-error">{message}</span> })}
                {move || notice.get().map(|message| view! { <span class="tone-badge ok">{message}</span> })}
            </div>

            {move || if loading.get() {
                view! { <div class="empty-state"><p>"Loading rules..."</p></div> }.into_any()
            } else if rules.get().is_empty() {
                view! { <div class="empty-state"><p>"No rules configured."</p></div> }.into_any()
            } else {
                view! {
                    <div class="rules-grid">
                        {move || rules.get().into_iter().enumerate().map(|(index, _rule)| {
                            view! { <RuleEditor rules=rules index=index /> }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}

            <details class="diag-card rules-raw-panel">
                <summary>"Raw JSON"</summary>
                <textarea
                    class="settings-field-textarea rules-json-editor"
                    spellcheck="false"
                    prop:value=move || raw_payload.get()
                    on:input=move |event| raw_payload.set(event_target_value(&event))
                />
                <div class="pipeline-save-row">
                    <button class="btn-sm" disabled=move || saving.get() on:click=refresh_raw_from_form>
                        "Refresh from form"
                    </button>
                    <button class="btn-primary" disabled=move || saving.get() on:click=save_raw>
                        "Save raw JSON"
                    </button>
                </div>
            </details>
        </div>
    }
}

#[component]
fn RuleEditor(rules: RwSignal<Vec<RuleModel>>, index: usize) -> impl IntoView {
    let remove_rule = move |_| {
        rules.update(|items| {
            if index < items.len() {
                items.remove(index);
            }
        });
    };
    let add_action = move |_| {
        rules.update(|items| {
            if let Some(rule) = items.get_mut(index) {
                rule.actions.push(RuleAction::default());
            }
        });
    };

    view! {
        <section class="diag-card rule-card">
            <div class="rule-card-header">
                <label class="rule-enabled-toggle">
                    <input
                        type="checkbox"
                        prop:checked=move || {
                            rules.with(|items| items.get(index).map(|rule| rule.enabled).unwrap_or(false))
                        }
                        on:change=move |event| {
                            let checked = event_target_checked(&event);
                            rules.update(|items| {
                                if let Some(rule) = items.get_mut(index) {
                                    rule.enabled = checked;
                                }
                            });
                        }
                    />
                    <span>{move || if rules.with(|items| items.get(index).map(|rule| rule.enabled).unwrap_or(false)) { "Enabled" } else { "Disabled" }}</span>
                </label>
                <button class="btn-sm btn-sm--danger" on:click=remove_rule>"Remove"</button>
            </div>

            <div class="settings-field-rows">
                <FormRow label="Rule ID">
                    <input
                        type="text"
                        class="settings-field-input"
                        prop:value=move || rules.with(|items| items.get(index).map(|rule| rule.id.clone()).unwrap_or_default())
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            rules.update(|items| {
                                if let Some(rule) = items.get_mut(index) {
                                    rule.id = value;
                                }
                            });
                        }
                    />
                </FormRow>
                <FormRow label="Scope">
                    <select
                        class="settings-field-input"
                        on:change=move |event| {
                            let value = event_target_value(&event);
                            rules.update(|items| {
                                if let Some(rule) = items.get_mut(index) {
                                    rule.scope = value;
                                }
                            });
                        }
                    >
                        <option
                            value="detection"
                            selected=move || rules.with(|items| items.get(index).map(|rule| rule.scope == "detection").unwrap_or(true))
                        >
                            "Detection"
                        </option>
                        <option
                            value="track"
                            selected=move || rules.with(|items| items.get(index).map(|rule| rule.scope == "track").unwrap_or(false))
                        >
                            "Track"
                        </option>
                    </select>
                </FormRow>
                <FormRow label="Cooldown seconds">
                    <input
                        type="number"
                        min="0"
                        step="0.1"
                        class="settings-field-input"
                        prop:value=move || rules.with(|items| {
                            items
                                .get(index)
                                .map(|rule| rule.cooldown_seconds.to_string())
                                .unwrap_or_else(|| "0".to_string())
                        })
                        on:input=move |event| {
                            let parsed = event_target_value(&event).parse::<f64>().unwrap_or(0.0).max(0.0);
                            rules.update(|items| {
                                if let Some(rule) = items.get_mut(index) {
                                    rule.cooldown_seconds = parsed;
                                }
                            });
                        }
                    />
                </FormRow>
                <FormRow label="Minimum confidence">
                    <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        class="settings-field-input"
                        prop:value=move || rules.with(|items| {
                            items
                                .get(index)
                                .and_then(|rule| rule.when.min_confidence)
                                .map(|value| value.to_string())
                                .unwrap_or_default()
                        })
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            let parsed = if value.trim().is_empty() {
                                None
                            } else {
                                value.parse::<f64>().ok()
                            };
                            rules.update(|items| {
                                if let Some(rule) = items.get_mut(index) {
                                    rule.when.min_confidence = parsed;
                                }
                            });
                        }
                    />
                </FormRow>
            </div>

            <div class="rules-section-label">"Match criteria"</div>
            <div class="compact-form-grid rules-condition-grid">
                <CsvField rules=rules index=index field="label_categories" label="Label categories" />
                <CsvField rules=rules index=index field="labels" label="Labels" />
                <CsvField rules=rules index=index field="reporting_modalities" label="Reporting modalities" />
                <CsvField rules=rules index=index field="zone_ids" label="Zone IDs" />
                <CsvField rules=rules index=index field="track_statuses" label="Track statuses" />
                <CsvField rules=rules index=index field="source_types" label="Source types" />
            </div>

            <div class="rules-section-row">
                <div class="rules-section-label">"Actions"</div>
                <button class="btn-sm" on:click=add_action>"+ Add action"</button>
            </div>
            <div class="rules-actions">
                {move || {
                    rules
                        .with(|items| items.get(index).map(|rule| rule.actions.len()).unwrap_or(0))
                        .checked_sub(0)
                        .map(|action_count| {
                            (0..action_count)
                                .map(|action_index| view! {
                                    <ActionEditor rules=rules index=index action_index=action_index />
                                })
                                .collect_view()
                        })
                }}
            </div>
        </section>
    }
}

#[component]
fn CsvField(
    rules: RwSignal<Vec<RuleModel>>,
    index: usize,
    field: &'static str,
    label: &'static str,
) -> impl IntoView {
    let read_value = move || {
        rules.with(|items| {
            let Some(rule) = items.get(index) else {
                return String::new();
            };
            match field {
                "label_categories" => join_csv(&rule.when.label_categories),
                "labels" => join_csv(&rule.when.labels),
                "reporting_modalities" => join_csv(&rule.when.reporting_modalities),
                "zone_ids" => join_csv(&rule.when.zone_ids),
                "track_statuses" => join_csv(&rule.when.track_statuses),
                "source_types" => join_csv(&rule.when.source_types),
                _ => String::new(),
            }
        })
    };
    let write_value = move |value: String| {
        let parsed = parse_csv(&value);
        rules.update(|items| {
            if let Some(rule) = items.get_mut(index) {
                match field {
                    "label_categories" => rule.when.label_categories = parsed,
                    "labels" => rule.when.labels = parsed,
                    "reporting_modalities" => rule.when.reporting_modalities = parsed,
                    "zone_ids" => rule.when.zone_ids = parsed,
                    "track_statuses" => rule.when.track_statuses = parsed,
                    "source_types" => rule.when.source_types = parsed,
                    _ => {}
                }
            }
        });
    };

    view! {
        <label>{label}
            <input
                type="text"
                class="settings-field-input"
                prop:value=read_value
                placeholder="comma-separated"
                on:input=move |event| write_value(event_target_value(&event))
            />
        </label>
    }
}

#[component]
fn ActionEditor(
    rules: RwSignal<Vec<RuleModel>>,
    index: usize,
    action_index: usize,
) -> impl IntoView {
    let payload_text = RwSignal::new(rules.with(|items| {
        items
            .get(index)
            .and_then(|rule| rule.actions.get(action_index))
            .map(|action| pretty_payload(&action.payload))
            .unwrap_or_else(|| "{}".to_string())
    }));
    let remove_action = move |_| {
        rules.update(|items| {
            if let Some(rule) = items.get_mut(index) {
                if rule.actions.len() > 1 && action_index < rule.actions.len() {
                    rule.actions.remove(action_index);
                }
            }
        });
    };

    view! {
        <div class="rule-action-card">
            <div class="compact-form-grid rules-action-grid">
                <label>"Type"
                    <input
                        type="text"
                        class="settings-field-input"
                        prop:value=move || {
                            rules.with(|items| {
                                items
                                    .get(index)
                                    .and_then(|rule| rule.actions.get(action_index))
                                    .map(|action| action.action_type.clone())
                                    .unwrap_or_else(|| "alert".to_string())
                            })
                        }
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            rules.update(|items| {
                                if let Some(action) = items
                                    .get_mut(index)
                                    .and_then(|rule| rule.actions.get_mut(action_index))
                                {
                                    action.action_type = value;
                                }
                            });
                        }
                    />
                </label>
                <label>"Destination"
                    <input
                        type="text"
                        class="settings-field-input"
                        prop:value=move || {
                            rules.with(|items| {
                                items
                                    .get(index)
                                    .and_then(|rule| rule.actions.get(action_index))
                                    .map(|action| action.destination.clone())
                                    .unwrap_or_else(|| "cop".to_string())
                            })
                        }
                        on:input=move |event| {
                            let value = event_target_value(&event);
                            rules.update(|items| {
                                if let Some(action) = items
                                    .get_mut(index)
                                    .and_then(|rule| rule.actions.get_mut(action_index))
                                {
                                    action.destination = value;
                                }
                            });
                        }
                    />
                </label>
                <label>"Priority"
                    <select
                        class="settings-field-input"
                        on:change=move |event| {
                            let value = event_target_value(&event);
                            rules.update(|items| {
                                if let Some(action) = items
                                    .get_mut(index)
                                    .and_then(|rule| rule.actions.get_mut(action_index))
                                {
                                    action.priority = value;
                                }
                            });
                        }
                    >
                        <option value="normal" selected=move || priority_matches(rules, index, action_index, "normal")>"Normal"</option>
                        <option value="high" selected=move || priority_matches(rules, index, action_index, "high")>"High"</option>
                        <option value="critical" selected=move || priority_matches(rules, index, action_index, "critical")>"Critical"</option>
                    </select>
                </label>
                <button class="btn-sm btn-sm--danger" on:click=remove_action>"Remove action"</button>
            </div>
            <label class="rules-payload-label">"Payload JSON"
                <textarea
                    class="settings-field-textarea rules-payload-editor"
                    spellcheck="false"
                    prop:value=move || payload_text.get()
                    on:input=move |event| payload_text.set(event_target_value(&event))
                    on:change=move |_| {
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&payload_text.get_untracked()) {
                            if value.is_object() {
                                rules.update(|items| {
                                    if let Some(action) = items
                                        .get_mut(index)
                                        .and_then(|rule| rule.actions.get_mut(action_index))
                                    {
                                        action.payload = value;
                                    }
                                });
                            }
                        }
                    }
                />
            </label>
        </div>
    }
}

fn priority_matches(
    rules: RwSignal<Vec<RuleModel>>,
    index: usize,
    action_index: usize,
    priority: &'static str,
) -> bool {
    rules.with(|items| {
        items
            .get(index)
            .and_then(|rule| rule.actions.get(action_index))
            .map(|action| action.priority == priority)
            .unwrap_or(priority == "normal")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv_discards_empty_items() {
        assert_eq!(parse_csv(" bird, , drone "), vec!["bird", "drone"]);
    }

    #[test]
    fn parse_raw_rules_accepts_array_or_object() {
        let object_rules = parse_raw_rules_config(
            r#"{"rules":[{"id":"security","when":{"min_confidence":0.4},"actions":[{"type":"alert"}]}]}"#,
        )
        .unwrap();
        let array_rules = parse_raw_rules_config(
            r#"[{"id":"security","when":{"label_categories":["security"]},"actions":[{"type":"alert"}]}]"#,
        )
        .unwrap();
        assert_eq!(object_rules[0].id, "security");
        assert_eq!(array_rules[0].actions[0].destination, "cop");
    }

    #[test]
    fn validation_rejects_invalid_confidence() {
        let mut rule = RuleModel {
            id: "bad_confidence".to_string(),
            ..RuleModel::default()
        };
        rule.when.min_confidence = Some(1.1);
        assert!(validate_and_normalize_rules(&[rule]).is_err());
    }
}
