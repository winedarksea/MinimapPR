# The Synthesis Vision: MinimapPR + IFC Digital Twin + Home Assistant

*A strategic analysis of the convergence of real-time spatial awareness, parametric building models, and smart home integration across residential, agricultural, retail, industrial, and conservation deployments.*

---

## Part 1: The Core Capability Triad

Before examining applications, it is worth naming precisely what these three systems provide and why no two of them together achieve what all three do.

| System | Core Capability | What It Cannot Do Alone |
|--------|-----------------|--------------------------|
| **MinimapPR** | Perceives events in physical space — acoustic localization, tracking, classification, rules, COP | Does not know what physically exists at event locations. Cannot control devices. |
| **IFC Digital Twin** | Knows what physically exists and exactly where — rooms, equipment, structures, building geometry, equipment lifecycle data | Cannot perceive events. Does not change. Is a static knowledge base. |
| **Home Assistant** | Controls physical devices and delivers information — the actuator layer for the physical world | Cannot reason spatially. No building geometry. No probabilistic state. Cannot do signal processing. |

The combination forms a **complete perception → knowledge → action triad**:

- *Something is happening at position (3.1, 7.2, 0.9)* — MinimapPR
- *That position is the kitchen, 1.2 m from the dishwasher* — IFC
- *Turn on the relevant HA notification, log the maintenance ticket, adjust the zone alert* — Home Assistant

None of the three pairs achieves this. All three together do. This is the fundamental multiplier.

---

## Part 2: Cross-System Synergies

These are the capabilities that are **structurally impossible without all three systems** working together.

### Synergy 1: Spatially-Anchored Automation
MinimapPR resolves an event to a 3D position. The IFC bridge queries "what is at this position?" and returns a room label and nearby equipment list. The rules engine fires a semantically rich action. HA executes a device response that is appropriate to the specific space and equipment — not just "glass break alert" but "glass break in kitchen, east wall, near window — arm the exterior lights at that corner."

This is categorically different from conventional HA automation, which has no concept of 3D position.

### Synergy 2: Equipment Acoustic Health Monitoring
The IFC model contains the 3D position of every piece of equipment. MinimapPR localizes acoustic events to within ~1 m. A sound localized near equipment position X, classified as mechanical anomaly, can be attributed to a *specific piece of equipment* — not just "unusual noise in the basement" but "compressor bearing noise, unit 2 of 3, installed 22 months ago, 6 months past service interval." HA triggers a maintenance workflow with full context. This is predictive maintenance at zero additional hardware cost beyond the existing acoustic sensor network.

### Synergy 3: Presence-Adaptive Physical Environment
MinimapPR produces room-level acoustic occupancy (confirmed human tracks in zone X). HA controls HVAC and lighting per zone. IFC defines the zones. The result: true presence-based building management — not schedule-based, not PIR-snap, but probabilistic track-based occupancy that follows people through rooms and adapts the physical environment accordingly. Commercial building energy studies consistently show 20–40% savings from true occupancy-based control vs. schedule-based.

### Synergy 4: LLM-Powered Situational Synthesis
With IFC (what exists), MinimapPR (what's happening, with history), and HA (all device states) available as structured context, an LLM can reason across all three simultaneously — something no individual system can approach. A context window containing: *"The compressor in the northwest corner has emitted acoustic anomalies for 3 consecutive days; temperature in that room is 2.1°C above the setpoint; last maintenance was 18 months ago; the homeowner is currently away (HA person entity: away)"* enables a maintenance recommendation, a priority assessment, and a specific action suggestion that no rule could have pre-specified.

### Synergy 5: Navigable Digital Twin
IFC geometry provides the navigable space — walls, openings, corridors, floor-to-floor transitions. MinimapPR provides real-time situational awareness — where are people, what events are happening, what is the operational state of equipment. HA controls access — doors, lights, locks. Together this is the complete substrate for robot and drone navigation, task assignment, and first-response dispatch. The robot knows where to go (IFC), what to avoid (MinimapPR tracks), and what to interact with (HA entities at IFC positions).

### Synergy 6: The LLM-IFC Knowledge Interface
When an LLM is reasoning about a situation, it needs to know what physical things *could be* relevant to query. The IFC model, queryable via `ifcopenshell`, provides a structured "what exists here" knowledge base that an LLM agent can use for tool calls — "query IFC: what equipment is within 2m of position (3.1, 7.2, -2.7)?" This gives the LLM grounding in physical reality that it could not infer from events alone.

---

## Part 3: Application Verticals — The Full Vision

### 3.1 Smart Home / Residential
*The primary deployment. The reference architecture for all other verticals.*

**Acoustic Security:**
- Glass break, intrusion, fire alarm, smoke detector T3 pattern detection with room-level localization
- Acoustic fingerprinting of known sounds (neighbor's dog, specific car) for suppression
- When-nobody-home priority elevation via HA presence enrichment

**Equipment Health & Maintenance:**
- HVAC compressor, air handler, heat pump acoustic baseline → anomaly scoring → maintenance dispatch
- Appliance fault detection (dishwasher pump, washer bearing, refrigerator compressor)
- Water sounds near plumbing fixtures → cross-validate with HA moisture sensors → leak detection

**Presence-Adaptive Automation:**
- Room-level acoustic occupancy → HA HVAC zone control, lighting, security arming
- Guest/occupant tracking for multi-zone climate and energy optimization
- "Last person out" acoustic confirmation before security arming

**Voice & Intelligence:**
- Wake word detection from any room → beamformed audio to STT → HA Voice (Wyoming)
- Conversational memory from detected speech
- LLM synthesis of daily home health: equipment, security, energy

**Future — Drone/Robot Dispatch:**
- Detected water leak near water heater → dispatch inspection drone to IFC-registered equipment position
- IFC provides navigation mesh (walls, openings, floor transitions)
- Robot reports back visual confirmation; LLM assesses severity and next action

---

### 3.2 Agriculture / Family Farm
*Highest ROI per dollar of deployment after residential. Equipment failure and livestock loss have immediate, large financial consequences.*

**Livestock Safety:**
- Acoustic detection of livestock distress calls (cattle, pigs, poultry) with localization to barn/pen zone in the IFC/GeoJSON farm model
- Predator intrusion detection: coyote calls, hawk screams, large animal approach — localized and cross-referenced with perimeter zone breaches
- Automated deterrent response via HA-connected devices (strobe, speaker, sprinkler)
- Birthing monitoring: unusual nighttime acoustic activity in calving zone → farmer alert
- Sick animal acoustic signature change over time (behavioral acoustic change precedes physical symptoms by hours to days)
- Poultry flock stress monitoring: crowding, heat stress, disease spread all have acoustic signatures in aggregate

**Agricultural Equipment Health (Highest Single ROI Item):**
- Combine, tractor, grain dryer, irrigation pump, auger, aeration fan — all have acoustic baselines
- A bearing about to fail sounds different from a healthy bearing by days to weeks before failure
- Preventing a combine breakdown at harvest easily justifies an entire deployment
- IFC/GeoJSON farm model registers each piece of equipment at its nominal storage/operation position
- MinimapPR builds per-equipment acoustic baseline during normal operation → anomaly scoring → HA maintenance alert
- Alert includes: which equipment, location, anomaly type, equipment age, last maintenance (from IFC equipment lifecycle property set)

**Autonomous Equipment Integration:**
- GPS/MAVLink telemetry from autonomous tractors and field robots into MinimapPR as mobile nodes
- COP shows tractor and drone positions relative to field boundaries, buildings, livestock, workers
- Worker safety: tractor approach to pedestrian zone → HA-connected warning
- Tractor route planning against IFC/GeoJSON field model (crop zones, obstacle positions)
- Drone scouting data fed as environmental observations into MinimapPR

**Crop / Field Operations:**
- IFC-extended GeoJSON spatial model of farm: field parcel boundaries, crop zones, irrigation grid, drainage
- Soil moisture sensors, weather stations via HA → MinimapPR environmental enrichment → better outdoor acoustic compensation
- Irrigation system acoustic monitoring: pump sounds, pipe anomalies, valve chatter
- Crop zone states (growth stage, moisture level) as context for automation rules
- Multi-site farm federation: each building and field sector as a MinimapPR zone; federated COP for the whole farm

**Farm Security:**
- Equipment theft detection (engine start sounds, machinery movement at unusual hours)
- Perimeter intrusion (vehicle approach, human footstep patterns on access roads)
- Worker emergency detection (distress calls, panic — HA sends to farm owner and emergency services)

---

### 3.3 Retail
*Most directly monetizable of the non-residential verticals. Occupancy intelligence, loss prevention, and HVAC optimization all have clear line-item value.*

**Customer Experience & Operations:**
- Zone occupancy density tracking → real-time staffing allocation signal to HA-connected display system
- Queue formation detection at checkout → automatic register-opening alert
- Dwell time distribution by zone → product placement optimization data
- "Dead zone" identification (zones with low dwell and low traffic despite good location) → layout feedback
- Customer sentiment proxy from ambient voice energy levels and patterns by zone

**Energy (Large-Scale Impact):**
- HVAC zone control from MinimapPR occupancy tracking vs. schedule-based control → 25–35% energy savings in large retail footprints, which have very high HVAC costs
- Lighting control by zone occupancy in back-of-house, warehousing, service corridors
- Refrigeration case monitoring (compressor acoustic anomaly detection) → reduce spoilage from undetected failures

**Loss Prevention:**
- Acoustic glass break localization → security camera cue to exact zone (not just "glass break somewhere in store")
- After-hours intrusion detection with room-level granularity
- Acoustic altercation detection → staff dispatch + camera focus
- High-value zone extended dwell monitoring (suspicious behavior pattern)

**Inventory & Operations:**
- IFC shelf/fixture positions as spatial model — stocking activity detected acoustically (box sounds, cart movement near specific shelf locations) correlates with inventory events
- Integration with WMS: equipment POIs in IFC carry inventory location IDs from warehouse management system
- Delivery/receiving zone acoustic monitoring → automated receiving workflow triggers
- Equipment maintenance: refrigeration units, escalators, HVAC — all acoustic health monitoring

**Privacy Note:** All retail applications require explicit privacy policy — acoustic event data only (no voice recording in commercial mode), anonymized occupancy counts, short retention on raw data, and zone-level aggregation rather than individual tracking.

---

### 3.4 Industrial / Warehouse / Manufacturing
*Single highest ROI application category overall: predictive acoustic maintenance on industrial machinery prevents catastrophic downtime.*

**Predictive Maintenance:**
- Conveyor belts, motors, compressors, presses, CNC machines, cooling towers — all have distinctive healthy and unhealthy acoustic signatures
- Baseline profiling during commissioning → continuous comparison → maintenance alert at first anomaly, weeks before failure
- Combined acoustic + vibration sensor nodes for industrial equipment → richer failure signatures
- IFC registers every machine at its exact position; equipment property sets carry model, age, maintenance history
- LLM synthesis: "Press #4 has shown elevated spectral energy in the 2–4 kHz band for 5 days. Historical data suggests this pattern precedes roller bearing failure. Recommended action: schedule inspection within 48 hours."

**Worker Safety:**
- Hazard zone entry detection: acoustic detection of people in robot operating zones, press safety zones, chemical zones → HA emergency stop relay
- Forklift approach warning: forklifts have distinctive acoustic signatures; pedestrian approach to forklift zone → warning
- Emergency event detection: alarm classification + localization → evacuation route optimization on COP
- PPE monitoring: certain protective equipment has acoustic signatures (respirator breathing sounds, hard hat interactions)

**Production Operations:**
- Production line COP: real-time view of production floor with equipment health indicators
- Line stoppage detection: sudden silence in normally-active zone → immediate alert
- Quality control: defective parts on certain production processes have detectable acoustic signatures
- AGV/robot tracking via acoustic or MAVLink telemetry

---

### 3.5 Healthcare / Hospital
*High value and technically feasible, but requires careful privacy architecture. HIPAA compliance and anonymization are prerequisites, not afterthoughts.*

**Patient Safety:**
- Fall detection: falls produce a distinctive acoustic signature (rapid thud + absence of normal movement sounds); MinimapPR localizes to specific room → instant nurse call
- Patient distress detection in rooms not under direct observation
- Wandering patient monitoring (acoustic room presence tracking) → alert when patient leaves designated zone at night
- Code blue: COP shows which staff are acoustically present nearest the event → fastest possible dispatch

**Equipment Alarms:**
- Medical device alarm classification is a significant clinical problem — alarms are routinely ignored because alarm fatigue is pervasive
- MinimapPR can classify IV pump, ventilator, cardiac monitor alarm patterns and localize them to the specific room
- Priority scoring (how long has this alarm been active, is there acoustic evidence of a human in the room responding?)
- HA routes to the right nurse workstation with room and alarm type

**Operations:**
- Room occupancy for bed-turn and housekeeping scheduling
- High-touch zone occupancy for infection control and disinfection scheduling
- Equipment location tracking (infusion pumps, medication carts) via acoustic identification and MinimapPR tracking
- Noise level monitoring per zone — patient recovery floors have clinical noise level requirements

---

### 3.6 Conservation / Wildlife Management
*Technically straightforward extension of MinimapPR's core. Very high value for conservation outcomes and uniquely differentiated from existing tools.*

**Anti-Poaching:**
- Gunshot detection and localization in protected areas — already in MinimapPR's core capability
- Immediate acoustic position to ranger team via HA-connected alert (satellite MQTT for remote areas)
- Vehicle intrusion: engine sounds on access paths → alert before poachers reach the interior
- Human footstep pattern detection in restricted core zones

**Biodiversity Monitoring:**
- BirdNET acoustic species identification with position and time → real-time species distribution maps
- Population density estimation from call frequency per zone over time
- Breeding season monitoring (territorial calls, nesting behavior acoustics)
- Species interaction events (predator-prey acoustic signatures)
- Long-term acoustic ecology baselines for habitat health assessment

**Ecosystem Management:**
- Acoustic monitoring of water features (stream flow levels, pump health in managed reserves)
- Invasive species detection by acoustic signature (certain invasive species have distinctive calls)
- Habitat use pattern mapping across seasons

**Deployment Notes:** Conservation deployments require the outdoor spatial model (GeoJSON field boundaries rather than IFC buildings), long-range node spacing, and satellite/cellular MQTT connectivity for remote areas. Solar-powered node hardware is essential.

---

### 3.7 Military / Tactical
*Already the conceptual origin of MinimapPR's COP design. IFC + HA provide new layers for base operations.*

**Forward Base Operations:**
- IFC model of FOB → optimal sensor placement planning, coverage analysis
- Equipment and supplies spatial registry (IFC equipment POI at logistics positions)
- Perimeter sensor health COP → degradation detection in real time
- EMCON mode: node-level acoustic recording without RF transmission

**Tactical Integration:**
- CoT export to ATAK/TAK Server from MinimapPR COP
- Drone telemetry via MAVLink into MinimapPR as mobile nodes — COP integrates UAS and acoustic tracks
- IFF integration: known friendly vehicles/personnel in IFC-linked known entity registry
- Multi-FOB federation for operational area COP

---

## Part 4: Reality Check — Value and Feasibility Matrix

*For each vertical and capability cluster: value, feasibility given the current stack, and whether it introduces fundamental conflicts.*

| Application | Value | Technical Feasibility | Conflict Risk | Core Changes Needed |
|-------------|-------|----------------------|---------------|---------------------|
| **Residential smart home** | High | High (designed for this) | None | IFC bridge, MQTT contract |
| **Farm livestock acoustic safety** | High | High (same pipeline, new labels) | None | Outdoor GeoJSON model, livestock labels |
| **Farm equipment health monitoring** | Very High | High (same pattern as home HVAC) | None | Acoustic baseline profiling module |
| **Autonomous tractor/drone COP** | Medium-High | Medium (MAVLink node type needed) | None | MAVLink telemetry node type |
| **Retail occupancy & energy** | High | High (room occupancy already produced) | Privacy policy | Deployment profiles, privacy modes |
| **Retail loss prevention** | Medium | High (core acoustic capabilities) | None | Retail-specific rules/labels |
| **Industrial predictive maintenance** | Very High | High (same baseline pattern) | None | Acoustic baseline profiling module |
| **Industrial worker safety** | High | High | Privacy policy | Safety zone types |
| **Healthcare fall detection** | High | High (falls have clear signature) | HIPAA privacy | Healthcare privacy mode, fall model |
| **Healthcare alarm localization** | High | High | HIPAA privacy | Medical device label taxonomy |
| **Conservation anti-poaching** | High | High (gunshot already core) | None | Outdoor model, satellite MQTT |
| **Biodiversity monitoring** | High | High (BirdNET already planned) | None | GeoJSON spatial model |
| **LLM synthesis layer** | Very High | Medium (needs context API) | None (sits above both) | Context aggregation API |
| **Drone/robot navigation** | High | Low (requires robot platform) | None | IFC navigation mesh export |
| **Retail inventory correlation** | Medium | Medium (speculative) | None | WMS integration |
| **Smart city infrastructure** | Medium | Low (requires city-scale deployment) | Regulatory/privacy | Federation at city scale |

### Highest-Value, Lowest-Conflict Cluster (Build These First)

1. **Acoustic baseline profiling** — single new capability that unlocks predictive maintenance across all verticals (farm equipment, industrial machinery, home appliances). No conflict with any existing function.
2. **Outdoor GeoJSON spatial model** — enables farm, conservation, and outdoor deployments without changing the indoor IFC path. The zone system already handles arbitrary polygons.
3. **Deployment configuration profiles** — enables the same codebase to serve residential, farm, retail, and industrial without code branching. Each profile sets: label taxonomy, default rules, privacy mode, spatial model source, COP display style.
4. **Context aggregation API** — a single new endpoint that makes the LLM layer possible for all verticals. Zero conflict with existing functions.
5. **Privacy modes** — prerequisite for commercial/retail/healthcare deployments. Clean architecture: privacy mode is a data retention and anonymization policy applied at the zone level, not a different code path.

---

## Part 5: Conflict and Incompatibility Analysis

### Real Conflicts (Design Choices Required)

**1. Privacy vs. Detailed Analytics**
Retail occupancy analytics, healthcare patient tracking, and industrial worker monitoring are in tension with privacy expectations and regulations (GDPR, HIPAA, CCPA). This is not a technical conflict — it is a deployment policy conflict that must be explicitly resolved in the architecture.

*Resolution:* Tiered privacy modes applied per-zone:
- `PRIVATE`: No audio storage, anonymized occupancy counts only, 60-second retention
- `ANALYTICS`: Aggregated zone occupancy, no individual tracking, configurable retention
- `OPERATIONAL`: Full tracking with appropriate data governance controls
- `HEALTHCARE`: HIPAA-specific anonymization, audit logging, access controls

Privacy mode is a first-class zone property, not a global setting — a retail store can have `ANALYTICS` in sales floor zones and `OPERATIONAL` in back-of-house equipment zones.

**2. Indoor Precision vs. Outdoor Scale**
MinimapPR's TDOA acoustic localization is calibrated for indoor/small-area use (meter-level precision over 10s of meters with 50mm arrays). Agricultural and conservation deployments span 100s to 1000s of meters. GPS-based node positioning, larger inter-node spacing, and different propagation models are required.

*Resolution:* This is already handled by MinimapPR's configurable coordinate modes and localization parameters — it is not an architectural conflict but a deployment configuration difference. Key: the outdoor localization precision is lower (tens of meters vs. sub-meter indoors) but still sufficient for livestock zone attribution, equipment proximity, and perimeter breach detection at agricultural scales.

**3. IFC (Building-Centric) vs. Agricultural/Outdoor Spatial Model**
IFC is fundamentally a building standard. Using it for farm fields requires `IfcSite` + `IfcGeographicElement` + custom property sets, which works but is awkward and not natively supported by BIM tools. For field-scale deployments, GeoJSON is a more appropriate spatial format.

*Resolution:* Keep IFC as the authoritative model for buildings (house, barns, warehouses, retail floor). Use GeoJSON for outdoor/site-level spatial data (field boundaries, crop zones, outdoor equipment positions). The MinimapPR zone system and ifcplot export system both emit the same zone JSON schema — the source (IFC or GeoJSON) is irrelevant to downstream consumers. This is the cleanest resolution: right tool for right scope.

**4. Real-Time Rules vs. LLM Latency**
LLM synthesis calls take 2–10 seconds. Security and safety rules must fire in under 1 second. Trying to route real-time alerts through an LLM creates dangerous latency.

*Resolution:* The LLM layer is explicitly NOT in the real-time path. It is an analysis and advisory service, not an alert dispatcher. Rules engine fires in real-time; LLM provides synthesis, anomaly summaries, and maintenance recommendations on a separate advisory channel. This is already the intent of the architecture — the LLM "sits above both systems."

**5. Construction Document Integrity vs. Operational Overlay**
Adding sensor positions, HA entity IDs, maintenance history, and acoustic baseline profile IDs to the IFC construction model risks polluting the document that gets submitted for permits and reviewed by structural engineers.

*Resolution:* Operational property sets use a namespace that standard BIM tools ignore (`Pset_ifcPlot_Operational_*`, `Pset_ifcPlot_HAIntegration`). BIM tools display only standard IFC property sets by default. The construction IFC and the operational overlay are the same file — the additional property sets are invisible to BIM reviewers but queryable by `ifcopenshell`. If this is still unacceptable for a specific project, a lightweight operational overlay file can reference elements in the construction IFC by GlobalId.

### Non-Conflicts (Things That Look Problematic But Are Not)

- **Multiple deployment verticals in one codebase**: MinimapPR's pluggable architecture and deployment profiles handle this cleanly.
- **HA's device-centric model vs. MinimapPR's spatial tracks**: The MQTT semantic state contract is the clean interface. Neither system needs to understand the other's internal data model.
- **IFC updates breaking MinimapPR zones**: Zone imports are explicit, triggered operations — not live coupling. The IFC can change without affecting MinimapPR until the next intentional re-import.
- **Farm and residential running on the same codebase**: Deployment profiles differentiate the behavior without code branching.

---

## Part 6: Core Changes Required Across All Systems

### MinimapPR Changes

**1. Acoustic Baseline Profiling** *(New module — unlocks the single highest-value capability across all verticals)*

A `minimappr/core/acoustic_baseline.py` module that builds per-location time-windowed statistical baselines of acoustic features (spectral centroid, energy, band-specific levels) and produces anomaly scores via Mahalanobis distance from baseline. Configurable per-equipment sensitivity thresholds. This is what transforms MinimapPR from an event detector into a predictive health monitor. Every equipment-monitoring use case across home, farm, industrial, and healthcare depends on this.

Required: baseline building phase, baseline querying API, anomaly event generation, `GET /api/v1/equipment/{equip_id}/health` endpoint.

**2. Outdoor / GeoJSON Spatial Model**

Add a `GeoJSONBridge` alongside the planned `IFCBridge` that reads GeoJSON FeatureCollections (field boundaries, crop zones, outdoor equipment positions) and populates the same zone/equipment registry. The zone system already handles arbitrary polygons — this is purely an import path addition. Farm and conservation deployments use GeoJSON for outdoor space; IFC for buildings. Both feed the same downstream systems.

**3. Deployment Configuration Profiles**

A `DeploymentProfile` configuration object that presets: label taxonomy, default rules, zone source (IFC or GeoJSON), privacy mode defaults, COP display style (indoor floor plan vs. outdoor map), and acoustic thresholds for the deployment context. Profiles: `residential`, `farm`, `retail`, `industrial`, `conservation`, `tactical`. This enables the same binary to serve all verticals with appropriate behavior out of the box.

**4. MAVLink / Telemetry Node Type**

A `TELEMETRY` node type that receives position and state from MAVLink, ROS2, or MQTT telemetry streams rather than audio frames. Tracked as mobile nodes on the COP — the same track management, zone matching, and rules engine applies. This enables autonomous tractor, drone, and robot integration. Implements `IngestTransport` protocol.

**5. Context Aggregation API** *(Prerequisite for the LLM layer)*

`GET /api/v1/context/current` — returns a single structured JSON document combining: active tracks with room labels, room occupancy states, equipment health summaries, recent alert history (last N), HA state snapshot (if enrichment client is active), and site metadata. Designed to be passed directly as LLM context. This is the "intelligence surface" — with this endpoint, any LLM service can immediately synthesize a meaningful situational picture.

**6. Privacy Mode Architecture**

Per-zone `privacy_mode` property (`private`, `analytics`, `operational`, `healthcare`) that controls: audio retention, track granularity (individual vs. anonymized count), data export eligibility, and audit logging requirements. Privacy mode is evaluated at the zone-matching stage before any persistence.

**7. Equipment Health API**

`GET /api/v1/equipment/{equip_id}/health` — returns baseline deviation score over time, recent acoustic observations near the equipment, anomaly event history, and equipment metadata from the IFC spatial model. Powers both the COP equipment overlay and the LLM maintenance synthesis.

---

### catlin-house / ifcplot Changes

**1. Generalize ifcplot as a Domain-Agnostic Spatial Modeling Library**

The library currently has residential-specific assumptions baked in (`HOUSE_WALL_2X6`, `HOUSE_ROOF` assemblies, residential storey conventions). Refactor to separate a domain-agnostic IFC utility layer from residential-specific implementations. This allows `ifcplot` to be used for barn/agricultural facility models, retail floor plan models, and industrial facility layouts without being constrained by residential conventions. The IFC standard itself supports all of these — the library just needs to stop assuming residential context.

**2. Operational Property Set Standard**

Define and document the operational property set schema as a stable versioned interface:
- `Pset_ifcPlot_HAIntegration`: `entity_id`, `device_class`, `friendly_name`, `install_date`
- `Pset_ifcPlot_EquipmentLifecycle`: `model`, `serial`, `manufacturer`, `install_date`, `service_interval_months`, `last_service_date`, `acoustic_baseline_id`
- `Pset_ifcPlot_SensorNode`: `node_id`, `node_type`, `capabilities`

These become first-class schema elements with versioned backward compatibility.

**3. Zone / Equipment Export as a Stable Versioned Contract**

The zone and equipment export scripts become a versioned interface (`v1.x`) that MinimapPR depends on. Schema changes follow a compatibility policy: additive changes only in minor versions; breaking changes require a major version bump with migration documentation. This prevents the spatial model and the perception system from drifting out of sync silently.

**4. Lightweight CMMS in IFC**

The `Pset_ifcPlot_EquipmentLifecycle` property set, combined with MinimapPR's acoustic health data, creates a lightweight Computerized Maintenance Management System where:
- The IFC model records what equipment exists, where, and its lifecycle metadata
- MinimapPR records what acoustic health signals have been observed
- The context API exposes both together for LLM or maintenance dashboard consumption

No separate CMMS software required for a residential or agricultural deployment.

**5. Farm / Outdoor Extension Module**

A `ifcplot/farm.py` module (analogous to `catlin_house.py`) that models:
- Barn and farm building structures using the same IFC utilities
- `IfcGeographicElement`-based field boundaries at site level
- Agricultural equipment positions as `IfcBuildingElementProxy` instances with `Pset_ifcPlot_EquipmentLifecycle`
- Livestock zone definitions as `IfcSpace` elements in barn structures

Field-scale outdoor boundaries export as GeoJSON from a companion script, consumed by MinimapPR's `GeoJSONBridge`.

---

### New Cross-System Components

**1. LLM Synthesis Service**

A standalone service — not embedded in MinimapPR or HA — that:
- Subscribes to MinimapPR's WebSocket for live events
- Periodically polls `GET /api/v1/context/current` from MinimapPR
- Optionally polls `GET /api/states` from HA for device context
- Maintains a rolling context window of the current operational situation
- Can be queried interactively ("What's unusual about today?", "What maintenance is overdue?", "Describe what's happening in the kitchen") or pushed proactively ("Anomaly detected: maintenance recommendation for main HVAC")
- Routes proactive insights to HA notifications or MinimapPR alerts

This service is the bridge between structured sensor data and human-readable intelligence. With the IFC spatial model, MinimapPR event history, and HA device state all in scope, it can reason about the physical world in a way no single system approaches.

**2. Shared Zone Schema (v1.0)**

```json
{
  "schema_version": "1.0",
  "coordinate_frame": "flat_xyz",
  "site_origin": {"lat": 0.0, "lon": 0.0},
  "zones": [
    {
      "zone_id": "kitchen",
      "zone_type": "interior_space",
      "source": "ifc",
      "polygon_xy": [[x0, y0], [x1, y1]],
      "z_floor": 0.0,
      "z_ceiling": 2.74,
      "storey": "main_floor",
      "privacy_mode": "private",
      "properties": {"room_name": "Kitchen", "ifc_space_guid": "..."}
    }
  ],
  "equipment": [
    {
      "equip_id": "hvac_main",
      "equip_type": "hvac",
      "position_m": [x, y, z],
      "zone_id": "mechanical_room",
      "ha_entity_id": "climate.main_hvac",
      "friendly_name": "Main Heat Pump",
      "lifecycle": {
        "install_date": "2023-04-15",
        "service_interval_months": 12,
        "last_service_date": "2024-04-15"
      }
    }
  ]
}
```

---

## Part 7: The Recommended Roadmap

### Tier 1 — Foundation (Enables All Verticals)
*These items have no conflicts, build on existing architecture, and are prerequisites for everything else.*

1. **IFC bridge + zone/equipment import** — the spatial knowledge base in MinimapPR
2. **MQTT semantic state contract** — the bidirectional HA integration
3. **HA enrichment client** — environmental and presence data into MinimapPR
4. **GeoJSON spatial bridge** — outdoor/farm/conservation zone model
5. **Operational property sets in IFC** — `HAIntegration`, `EquipmentLifecycle`, `SensorNode`

### Tier 2 — Value Multiplication (Highest ROI New Capabilities)
*These items add new capability categories that serve multiple verticals simultaneously.*

6. **Acoustic baseline profiling module** — predictive maintenance for home, farm, industrial, healthcare
7. **Deployment configuration profiles** — residential, farm, retail, industrial, conservation
8. **Context aggregation API** — prerequisite for the LLM synthesis service
9. **Privacy mode architecture** — prerequisite for commercial/healthcare deployments
10. **Equipment health API** — surfaces baseline profiling data to COP and LLM

### Tier 3 — Extended Verticals (Specific New Infrastructure)
*Each item extends the platform to a new deployment context.*

11. **MAVLink telemetry node type** — tractor, drone, robot COP integration
12. **LLM synthesis service** — intelligence layer above MinimapPR + HA
13. **Farm/outdoor IFC extension module** — agricultural facility modeling
14. **Versioned zone export contract** — stable interface between spatial model and MinimapPR

### Tier 4 — Future Capabilities (Architecturally Supported)
*These require external platforms (robot hardware, city-scale deployment) but the architecture should not prevent them.*

15. **IFC navigation mesh export** — for drone/robot path planning
16. **Satellite/cellular MQTT** — conservation and remote farm deployments
17. **Multi-site federation** — retail chains, large farms, military area operations
18. **Full LLM-IFC query interface** — LLM agent makes `ifcopenshell` tool calls for spatial queries

---

## Closing Assessment

The most important insight from this analysis is that **the three systems form a capability triad that is genuinely greater than the sum of its parts**, and the multiplier applies across every domain where physical space matters — which is nearly every domain of human activity.

The immediate practical priority is Tier 1: getting the IFC knowledge base connected to MinimapPR's spatial reasoning and the MQTT semantic state contract connected to HA. This alone produces the full residential smart home synthesis.

The highest-leverage single new capability is **acoustic baseline profiling**. It is technically straightforward (statistical modeling of spectral features over time), has zero conflict with any existing function, and unlocks predictive maintenance value in every vertical — from protecting a combine at harvest, to catching a failing hospital ventilator, to flagging a compressor a week before it shuts down a cold storage warehouse.

The LLM synthesis service is the emergent intelligence layer that becomes possible once the other pieces are in place. Its value compounds with the quality of the underlying structured context — and the structured context (IFC spatial model + MinimapPR event history + HA device state) is uniquely rich precisely because of how these three systems complement each other.
