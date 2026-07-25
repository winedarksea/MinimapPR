# Home Assistant Integration (outbound MQTT bridge)

MinimapPR publishes itself to Home Assistant as a single MQTT device with a set
of well-defined entities: zone occupancy, per-zone sound level, detection-class
impulse sensors, node connectivity diagnostics, system health, and detection /
alert event entities. Discovery is automatic and retained, so **there is nothing
to configure on the Home Assistant side** beyond having the MQTT integration set
up against the same broker.

Direction is **outbound only** in this version. See
[Not implemented](#not-implemented) for what is deliberately absent.

> **Compatibility caveat.** No live Home Assistant instance was available when
> this was built. The discovery payloads were hand-verified against the HA MQTT
> Discovery documentation and are locked down by golden fixtures plus a spec
> lint (`tests/test_hass_discovery_payloads.py`), and message delivery is
> covered by an env-gated mosquitto test. That is not the same as
> spec-verified compatibility. **A manual HA smoke test is the first follow-up.**
> The highest-risk fields are called out in [Risks](#known-risks).

## Architecture

```
detections/tracks ──► LiveEventHub.broadcast() ──sync tee──► _inbound queue
                                                                  │
zones/nodes/health ──► poll every ~5s ────────────────────────────┤
                                                                  ▼
rule actions ──► HassRuleActionHandler.handle() ──put_nowait──► _outbound queue
                                                                  │
                                         coalesce + dedupe + throttle
                                                                  ▼
                                              MqttTransport ──► broker ──► HA
```

Two input paths, because they carry different kinds of state:

* **Tee** (`LiveEventHub.subscribe()`) — detections and alerts, which are
  broadcast events. The subscriber is a bare `put_nowait`; `broadcast()` runs on
  the fusion hot path, so the fan-out adds no await points and all
  interpretation happens later in the publisher task.
* **Poll** — zone occupancy, node health, and track counts, which are *never*
  broadcast and so cannot be tee'd. This is the cost driver of the publish
  interval: `ZoneMatcher.compute_occupancy` is O(zones × tracks) with a
  point-in-polygon test per pair, which is why `hass_publish_interval_seconds`
  has a validated 1.0 s floor.

| File | Responsibility |
|---|---|
| `core/hass/models.py` | `HassBridgeConfig` + `HassBridgeMetrics` value types |
| `core/hass/transport.py` | `MqttTransport` Protocol + message/config value types |
| `core/hass/aiomqtt_transport.py` | `aiomqtt` implementation (optional extra, lazily imported) |
| `core/hass/topics.py` | Topic builders, `slugify`, the rule-action topic guard |
| `core/hass/discovery.py` | Pure discovery-payload builders + `device_class` maps |
| `core/hass/state_mapper.py` | MinimapPR state → `list[MqttPublish]`; owns the retain decision |
| `core/hass/entity_set.py` | Which entities should exist, for the reconciler to diff |
| `core/hass/outbound.py` | The publish queue: coalescing, suppress-unchanged, rate floor |
| `core/hass/track_slots.py` | Sticky N-slot `device_tracker` assignment |
| `core/hass/spl_aggregator.py` | Per-zone rolling SPL window |
| `core/hass/ledger.py` | Persisted record of what was published, for cross-restart removal |
| `core/hass/reconcile.py` | Diffs the desired set against the ledger: create, update, remove |
| `core/hass/bridge.py` | `HassBridge`: lifecycle, queues, coalescing, reconcile |
| `core/hass/rules_handler.py` | `HassRuleActionHandler` (rules `destination: "hass"`) |

## Setup

1. Install the optional MQTT client:

   ```bash
   pip install -e '.[hass]'      # or '.[full]'
   ```

   Without it the bridge reports `transport_available: false` and publishes
   nothing — it does not fail to start.

2. Point MinimapPR at your broker, either in Settings → Integrations or by env:

   ```bash
   export MINIMAPPR_HASS_ENABLED=1
   export MINIMAPPR_HASS_MQTT_HOST=mqtt.local
   ```

   Those two are the only settings that are ever required.

3. Restart MinimapPR. Entities appear in HA within one publish interval.

The bridge runs only when `process_role != "api"`. In a split deployment the
api-only process deliberately does not start it — two processes publishing the
same retained topics would fight over every entity's state.

## Entities and topics

`P` = `hass_discovery_prefix` (`homeassistant`), `B` = `hass_base_topic`
(`minimappr`), `D` = `hass_device_id` (`minimappr`).

| Entity | Discovery topic | State topic | Retained |
|---|---|---|---|
| Availability (LWT + birth) | — | `B/status` (`online`/`offline`) | yes |
| Zone occupancy | `P/binary_sensor/D/zone_occupancy_<z>/config` | `B/zone/<z>/occupancy` + `/attributes` | yes |
| Zone sound level | `P/sensor/D/zone_spl_<z>/config` | `B/zone/<z>/spl_db` | yes |
| Detection class | `P/binary_sensor/D/detection_<c>/config` | `B/detection_class/<c>` | no |
| Node connectivity | `P/binary_sensor/D/node_online_<n>/config` | `B/node/<n>/connectivity` + `/attributes` | yes |
| System health | `P/sensor/D/system_health/config` | `B/system/health` + `B/system/attributes` | yes |
| Active track count | `P/sensor/D/active_track_count/config` | `B/system/active_track_count` | yes |
| Track slot ×N | `P/device_tracker/D/track_slot_<NN>/config` | `B/track/<NN>/state` + `/attributes` | yes |
| Detection event | `P/event/D/detection/config` | `B/event/detection` | no |
| Alert event | `P/event/D/alert/config` | `B/event/alert` | no |
| Rule-authored topics | *(no discovery)* | `B/<whatever the rule names>` | per-rule |

**The retain rule, stated once.** Retained: availability, all discovery configs,
and all *stateful* entities — so an HA restart restores last-known state without
waiting for our next poll. Not retained: `event` entities (a retained event
re-fires on every HA restart) and the impulse detection binary_sensors, which
use `off_delay` (a retained ON restores and then auto-offs, firing every attached
automation for a sound that happened hours ago).

### device_class mapping

| Source | Mapping |
|---|---|
| Zone `alert_zone` | `safety` |
| Zone `exclusion_zone` | `problem` |
| Zone `coverage_zone` / `interest_zone` | `occupancy` |
| Detection `gunshot` / `glass_break` / `scream` / `explosion` | `safety` |
| Detection `smoke_alarm` / `fire_alarm` | `problem` |
| Detection `vehicle` / `drone` | `motion` |
| Any other detection | `sound` |
| Node connectivity | `connectivity` + `entity_category: diagnostic` |

`unique_id` is `minimappr_<device_id>_<object_id>`. Zone and node ids are
slugified deterministically (lowercase, non-`[a-z0-9_]` → `_`, truncated at 48
with a sha1 suffix), because the slug is baked into the `unique_id` that HA's
entity registry persists forever.

### Notable semantics

* **Zone sound level is a new derived concept.** `spl_db` exists elsewhere in
  the system only per-`DetectionEvent`; there is no aggregate zone SPL. The
  bridge keeps a rolling window (default 60 s) per zone and publishes its
  **max**, not its mean — a mean buries a gunshot under 59 s of quiet. A zone
  with no sample in the window publishes HA's unknown sentinel, never `0.0` dB,
  which would be a measurement never taken.
* **Node connectivity is binary.** `online` → ON, `offline` → OFF;
  `degraded` / `bit_fail` → **ON** with `degraded: true` in the attributes. A
  degraded node is still reachable, and reading as unplugged would be wrong.
* **Detection `event_type` is clamped** to the taxonomy category set, because HA
  discards an `event_type` that was not declared in `event_types`.
* **Track slots are a fixed pool**, default 8 and **shipped disabled**. HA's
  entity registry persists every `unique_id` forever, so one entity per track id
  would grow it without bound and leave an orphan per track that ever existed.
  Slots fill highest-TQI-first, ties broken on lowest track id, and are sticky
  for a track's lifetime so an automation watching `track_slot_03` follows one
  target. Attributes carry `source_type: "gps"`, without which HA ignores
  `latitude`/`longitude`. Vacant slots publish `not_home` with `track_id: null`.

## Rule actions (`destination: "hass"`)

A rule with `destination: "hass"` and a `topic` in its payload publishes to
`<base_topic>/<topic>`:

```json
{
  "action_type": "alert",
  "destination": "hass",
  "priority": "high",
  "payload": {"topic": "rooms/kitchen/occupancy", "message": "ON", "retain": true}
}
```

* `topic` is required. A `message` key publishes a bare scalar (for an HA entity
  expecting `"ON"`); otherwise the body is a JSON object carrying the action
  type, priority, any extra payload keys, and detection/track context when
  present. On the transcript dispatch path, where both are `None`, those fields
  are **omitted rather than null-filled**.
* `retain` is opt-in per rule; rule actions are never coalesced or deduped.
* **Topics are guarded.** Wildcards (`+`, `#`), a leading `/`, and empty levels
  are refused outright; everything else is prefixed into `<base_topic>/`. This is
  a real safety control: without it a stored rule could name
  `homeassistant/binary_sensor/.../config` and overwrite every discovery payload
  on the broker.

**`delivered=True` means queued, not broker-acked.** `FusionNode._dispatch_rule_action`
awaits handlers inline on the fusion pipeline, so a broker round-trip in the
handler would stall detection emission. The handler enqueues and returns, which
means the alert row records `"sent"` for a message that could still be dropped
(queue full, broker down past queue capacity). `alert_id` is passed into the
handler, so what remains to close the loop is for the bridge to patch that row
once the publish lands. Out of scope here; tracked in `TODO.md`.

## Delivery behaviour

* **Coalescing** — stateful topics are last-write-wins within a flush cycle, so
  three occupancy changes in one interval cost one publish. Impulses (events,
  alerts, rule actions) bypass coalescing entirely: two identical gunshots one
  second apart are two events.
* **Suppress-unchanged** — a retained payload identical to the last one sent is
  skipped. This is what stops republishing unchanged zone state every interval.
  The cache is cleared on every (re)connect, because a broker restart may have
  lost retained messages and a stale cache would suppress exactly the republish
  that heals it.
* **Rate floor** — `hass_publish_min_interval_seconds` per topic. Deferred
  messages re-inject next cycle, which is lossless precisely because they are
  coalescable.
* **Reconnect** — exponential backoff, ±20 % jitter, 1 s → 60 s. While
  disconnected the loop keeps draining and polling into the outbound queue;
  coalescing means an hour offline costs at most one message per topic.
* **Drop policy is drop-newest** on a full queue, counted in
  `messages_dropped_queue_full` / `live_events_dropped`. Retained state is fully
  re-derived by the next poll, so a dropped state message self-heals within one
  interval; impulses can genuinely be lost, which is why the counters are on the
  status endpoint rather than only in the log.

## Entity add/remove

The desired set is derived each reconcile from `storage.list_zones()` and
`storage.list_nodes()` plus the static entities. Zone and node CRUD routes call
`HassBridge.request_reconcile()` (synchronous — it only sets a flag), and a full
reconcile also runs every `hass_reconcile_interval_seconds` so a missed hook
self-heals within a minute.

Removal publishes an **empty retained payload** to the discovery config topic
(HA's documented delete) and to that entity's state and attribute topics, so the
broker keeps no zombie value for an entity HA has dropped.

`data/hass_discovery_ledger.json` persists the published set across restarts,
written tmp-then-`os.replace`. Without it, a zone deleted while MinimapPR was
stopped would be absent from the desired set on restart and therefore invisible —
its retained config would sit on the broker forever and HA would keep showing an
unavailable entity. A corrupt ledger reads as empty, which republishes everything
(noisy but correct).

## Configuration reference

Reads are **nested** under `"hass"` in `GET /api/v1/config`; writes are **flat**
`hass_*` keys via `PATCH /api/v1/config`. This matches the existing `federation`
pattern and keeps the flat-key group coverage in
`GET /api/v1/config/structured` unchanged. The `hass` block lives in the
`integrations` config group.

| Setting | Env | Default |
|---|---|---|
| `hass_enabled` | `MINIMAPPR_HASS_ENABLED` | `false` |
| `hass_mqtt_host` | `MINIMAPPR_HASS_MQTT_HOST` | `""` |
| `hass_mqtt_port` | `MINIMAPPR_HASS_MQTT_PORT` | `1883` |
| `hass_mqtt_username` / `_password` | `MINIMAPPR_HASS_MQTT_USERNAME` / `_PASSWORD` | `""` |
| `hass_mqtt_client_id` | `MINIMAPPR_HASS_MQTT_CLIENT_ID` | `minimappr` |
| `hass_mqtt_keepalive_seconds` | `MINIMAPPR_HASS_MQTT_KEEPALIVE_SECONDS` | `60` |
| `hass_mqtt_tls_enabled` / `_insecure` | `MINIMAPPR_HASS_MQTT_TLS_ENABLED` / `_INSECURE` | `false` |
| `hass_discovery_prefix` | `MINIMAPPR_HASS_DISCOVERY_PREFIX` | `homeassistant` |
| `hass_base_topic` | `MINIMAPPR_HASS_BASE_TOPIC` | `minimappr` |
| `hass_device_id` / `hass_device_name` | `MINIMAPPR_HASS_DEVICE_ID` / `_DEVICE_NAME` | `minimappr` / `MinimapPR` |
| `hass_publish_interval_seconds` | `MINIMAPPR_HASS_PUBLISH_INTERVAL_SECONDS` | `5.0` (floor 1.0) |
| `hass_publish_min_interval_seconds` | `MINIMAPPR_HASS_PUBLISH_MIN_INTERVAL_SECONDS` | `1.0` |
| `hass_reconcile_interval_seconds` | `MINIMAPPR_HASS_RECONCILE_INTERVAL_SECONDS` | `60.0` |
| `hass_queue_size` | `MINIMAPPR_HASS_QUEUE_SIZE` | `2000` |
| `hass_reconnect_backoff_initial_seconds` | `MINIMAPPR_HASS_RECONNECT_BACKOFF_INITIAL_SECONDS` | `1.0` |
| `hass_reconnect_backoff_max_seconds` | `MINIMAPPR_HASS_RECONNECT_BACKOFF_MAX_SECONDS` | `60.0` |
| `hass_detection_off_delay_seconds` | `MINIMAPPR_HASS_DETECTION_OFF_DELAY_SECONDS` | `30` |
| `hass_detection_classes` | `MINIMAPPR_HASS_DETECTION_CLASSES` | `security,human,vehicle,wildlife` |
| `hass_track_slot_count` | `MINIMAPPR_HASS_TRACK_SLOT_COUNT` | `8` (range 0–64) |
| `hass_zone_spl_window_seconds` | `MINIMAPPR_HASS_ZONE_SPL_WINDOW_SECONDS` | `60.0` |
| `hass_discovery_ledger_path` | `MINIMAPPR_HASS_DISCOVERY_LEDGER_PATH` | `data/hass_discovery_ledger.json` |
| `hass_publish_zone_occupancy` | `MINIMAPPR_HASS_PUBLISH_ZONE_OCCUPANCY` | `true` |
| `hass_publish_zone_spl` | `MINIMAPPR_HASS_PUBLISH_ZONE_SPL` | `true` |
| `hass_publish_detection_classes` | `MINIMAPPR_HASS_PUBLISH_DETECTION_CLASSES` | `true` |
| `hass_publish_node_status` | `MINIMAPPR_HASS_PUBLISH_NODE_STATUS` | `true` |
| `hass_publish_system_health` | `MINIMAPPR_HASS_PUBLISH_SYSTEM_HEALTH` | `true` |
| `hass_publish_events` | `MINIMAPPR_HASS_PUBLISH_EVENTS` | `true` |
| `hass_publish_track_slots` | `MINIMAPPR_HASS_PUBLISH_TRACK_SLOTS` | `false` |
| `hass_base_url` / `hass_token` | `MINIMAPPR_HASS_BASE_URL` / `_TOKEN` | `""` |

`hass_base_url` and `hass_token` belong to the **not-yet-implemented** inbound
enrichment client. The MQTT bridge never reads them.

`hass_detection_classes` is matched against both a detection's `label` and its
`label_category`, so the default (the closed taxonomy category set) yields
working sensors out of the box while an operator can add a specific label like
`gunshot` for a finer-grained sensor.

`hass_detection_classes` and `hass_discovery_ledger_path` are env/YAML-level only
and deliberately absent from the PATCH allowlist.

**Enabling with no broker host raises at startup.** That is a config error, and
discovering it from a missing entity in HA would be far worse than a startup
failure naming the field.

### Secret handling

`hass_token` and `hass_mqtt_password` are redacted to `"***"` in
`GET /api/v1/config`. A PATCH carrying `"***"` back is treated as *unchanged* and
never overwrites the stored secret — guarded in both the API and the Settings
page, so neither layer alone can destroy a secret the operator did not retype.

**Caveat:** secrets are stored in plaintext in `data/config.yml`, matching the
existing behaviour for `hass_token` and `federation_auth_token`. Protect that
file with filesystem permissions.

## Status and recovery

| Endpoint | Behaviour |
|---|---|
| `GET /api/v1/integrations/hass/status` | Always 200; `connection_state: "disabled"` when the bridge is absent |
| `POST /api/v1/integrations/hass/republish-discovery` | Forces a full reconcile + snapshot next cycle; 503 when disabled |
| `POST /api/v1/integrations/hass/purge-discovery` | Blanks every retained topic and clears the ledger; 503 when disabled |

`connection_state` is one of `disabled`, `disconnected`, `connecting`,
`connected`, `error`. A `hass_status` live event is broadcast on transition only,
plus once per reconcile.

**Republish** is the recovery path after clearing retained messages on the broker
by hand: the ledger would otherwise consider every entity already-published and
skip it.

**Purge before uninstalling.** Otherwise HA keeps every entity forever as a
permanently-unavailable row in its registry. Purge also blanks the availability
topic, so a still-running bridge shows as unavailable in HA until its next
reconnect — acceptable, since purge is a pre-uninstall action, not routine
maintenance.

## Troubleshooting

Watch everything we publish:

```bash
mosquitto_sub -h mqtt.local -t 'homeassistant/#' -v      # discovery configs
mosquitto_sub -h mqtt.local -t 'minimappr/#' -v          # state and events
```

| Symptom | Check |
|---|---|
| No entities in HA | `GET /api/v1/integrations/hass/status` — is `connection_state` `connected` and `transport_available` `true`? |
| `transport_available: false` | The `hass` extra is not installed: `pip install -e '.[hass]'` |
| Status stuck on `error` | `last_connect_error` names the cause (auth, TLS, unreachable host) |
| Entities exist but never update | `messages_suppressed_unchanged` climbing is normal; `messages_dropped_queue_full` climbing is not |
| Stale entities after deleting a zone | Wait one `hass_reconcile_interval_seconds`, then use republish-discovery |
| Zone sound level reads unknown | No detection with an `spl_db` landed in that zone within `hass_zone_spl_window_seconds` |
| Entities orphaned after uninstall | Purge was not run; clear them by hand with `mosquitto_pub -t '<config topic>' -r -n` |

## Testing

All offline by default. `tests/hass_helpers.py` provides a recording transport
installed at the `bridge._build_transport` seam, and tests drive publish cycles
explicitly rather than waiting on the interval.

| File | Covers |
|---|---|
| `test_hass_discovery_payloads.py` | Golden fixtures + the HA-spec lint |
| `test_hass_config.py` | Defaults, every env var, each validation error |
| `test_hass_live_subscription.py` | The hot-path tee contract |
| `test_hass_state_mapper.py` | State mapping, retain policy, SPL window, slot allocation |
| `test_hass_bridge.py` | Lifecycle, availability, coalescing, dedupe, failure paths |
| `test_hass_bridge_reconcile.py` | Discovery add/remove and the ledger |
| `test_hass_rules_handler.py` | Rule actions and the topic guard |
| `test_hass_api.py` | Status / republish / purge endpoints |
| `test_hass_broker_integration.py` | **Env-gated**: real `aiomqtt` against mosquitto |

To run the live-broker test:

```bash
brew install mosquitto && mosquitto -p 1883 &
pip install -e '.[hass]'
MINIMAPPR_HASS_LIVE_BROKER_TEST=1 .venv/bin/python -m pytest \
    tests/test_hass_broker_integration.py -q
```

## Known risks

1. **The golden fixtures cannot prove HA compatibility.** The highest-risk
   fields are the `event` entity requirements, `device_tracker` needing
   `source_type: "gps"` for HA to accept lat/lon, `off_delay` semantics, and the
   unknown-value sentinel for a numeric sensor (we publish `"None"`, which is
   HA's `PAYLOAD_NONE`). A manual smoke test against a real HA instance is the
   first follow-up.
2. **`delivered=True` on a queued rule action** — see
   [Rule actions](#rule-actions-destination-hass).
3. **The publish interval is the poll cost.** Lowering it below a few seconds on
   a site with many zones and tracks will make `compute_occupancy` visible in
   the process's CPU profile.

## Not implemented

Deliberately out of scope in this version, and still open in `TODO.md`:

* **Inbound enrichment** (`HassEnrichmentClient`) — reading HA's environmental
  sensors or `person` presence entities. `hass_base_url` / `hass_token` exist for
  it but nothing consumes them yet.
* **REST webhook fallback** for HA installs without MQTT. MQTT is the only
  transport.
* **HA custom events, service calls, area mapping, Lovelace card,
  `media_source` audio.**
* **HA Add-on / HACS packaging** and a UI config flow on the HA side.
* **`LocalEffectorHandler`** — direct GPIO-style commands back to node hardware.
