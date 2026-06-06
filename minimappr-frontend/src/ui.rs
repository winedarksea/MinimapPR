use js_sys::Date;

use crate::state::{CopItemKind, CopSelection};

pub fn short_id(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

pub fn cop_sidebar_element_id(kind: CopItemKind, id: &str) -> String {
    let mut escaped_id = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            escaped_id.push(ch);
        } else {
            escaped_id.push('_');
        }
    }
    format!("cop-{}-{escaped_id}", kind.as_js_kind())
}

pub fn is_cop_item_selected(selection: &Option<CopSelection>, kind: CopItemKind, id: &str) -> bool {
    selection
        .as_ref()
        .is_some_and(|selected| selected.kind == kind && selected.id == id)
}

pub fn format_age_seconds(total_seconds: u64) -> String {
    if total_seconds < 60 {
        format!("{total_seconds}s")
    } else if total_seconds < 3_600 {
        format!("{}m {:02}s", total_seconds / 60, total_seconds % 60)
    } else {
        format!(
            "{}h {:02}m",
            total_seconds / 3_600,
            (total_seconds % 3_600) / 60
        )
    }
}

pub fn classify_age_from_ns(
    timestamp_ns: Option<i64>,
    stale_after_seconds: f64,
    lost_after_seconds: f64,
) -> (String, &'static str) {
    let Some(timestamp_ns) = timestamp_ns else {
        return ("—".to_string(), "age-unknown");
    };

    let age_seconds = (Date::now() * 1_000_000.0 - timestamp_ns as f64) / 1_000_000_000.0;
    if age_seconds < 0.0 {
        return ("0s".to_string(), "age-fresh");
    }

    let age_class = if age_seconds < stale_after_seconds {
        "age-fresh"
    } else if age_seconds < lost_after_seconds {
        "age-stale"
    } else {
        "age-lost"
    };
    (format_age_seconds(age_seconds as u64), age_class)
}

pub fn health_chip_class(health: &str) -> &'static str {
    match health {
        "online" => "health-chip online",
        "degraded" => "health-chip degraded",
        "offline" => "health-chip offline",
        _ => "health-chip unknown",
    }
}

pub fn track_status_chip_class(status: Option<&str>) -> &'static str {
    match status {
        Some("active") | Some("confirmed") => "health-chip online",
        Some("coasting") => "health-chip degraded",
        Some("lost") | Some("dropped") => "health-chip offline",
        _ => "health-chip unknown",
    }
}

pub fn track_status_label(status: Option<&str>) -> &'static str {
    match status {
        Some("active") => "ACT",
        Some("confirmed") => "CFM",
        Some("coasting") => "CST",
        Some("lost") => "LST",
        Some("dropped") => "DRP",
        _ => "UNK",
    }
}

pub fn severity_badge_class(severity: &str) -> &'static str {
    match severity {
        "critical" | "high" => "tone-badge danger",
        "medium" | "warn" => "tone-badge warn",
        "low" | "normal" => "tone-badge info",
        _ => "tone-badge neutral",
    }
}

pub fn alert_status_badge_class(status: &str) -> &'static str {
    match status {
        "acknowledged" => "tone-badge ok",
        "dismissed" => "tone-badge neutral",
        "escalated" => "tone-badge danger",
        "sent" => "tone-badge warn",
        _ => "tone-badge neutral",
    }
}
