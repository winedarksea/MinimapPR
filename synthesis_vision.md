# The Synthesis Vision: MinimapPR + IFC Digital Twin + Home Assistant

*A strategic analysis of the convergence of real-time spatial awareness, parametric building models, and smart home integration across residential, agricultural, retail, industrial, and conservation deployments.*

---

## Part 1: The Core Capability Triad

Before examining applications, it is worth naming precisely what these three systems provide and why no two of them together achieve what all three do.

| System | Core Capability | What It Cannot Do Alone |
|--------|-----------------|--------------------------|
| **MinimapPR** | Perceives events in physical space and maintains a live dynamic world model — acoustic localization, tracking, probabilistic track state, classification, rules engine, COP | Carries no *static* building knowledge: what rooms exist, what equipment is installed, where permanent structures are. Does not directly control consumer devices (lights, locks, HVAC). |
| **IFC Digital Twin** | Knows what physically exists and exactly where — rooms, equipment, structures, building geometry, equipment lifecycle data | Cannot perceive events. Does not change. Is a static knowledge base. |
| **Home Assistant** | Controls physical devices and delivers information — the actuator layer for the physical world | Cannot reason spatially. No building geometry. No probabilistic state. Cannot do signal processing. |

The combination forms a **complete perception → knowledge → action triad**:

- *Something is happening at position (3.1, 7.2, 0.9)* — MinimapPR
- *That position is the kitchen, 1.2 m from the dishwasher* — IFC
- *Turn on the relevant HA notification, log the maintenance ticket, adjust the zone alert* — Home Assistant

None of the three pairs achieves this. All three together do. This is the fundamental multiplier.

### 1.1 The Spatial Device and Spatial Automation Distinction

The key asymmetry between MinimapPR's dynamic knowledge and IFC's static knowledge is what makes the triad necessary. But it also clarifies a critical design boundary: MinimapPR's "does not control consumer devices" is a deliberate architectural choice, not a capability limit. Two distinct categories of device control exist.

**Consumer Devices (controlled by HA):**
Lights, locks, HVAC thermostats, appliances, sensors, and switches — devices whose operation does not depend on real-time spatial track state. HA controls these via its extensive protocol integrations (Z-Wave, Zigbee, Matter, cloud APIs). MinimapPR's role is to *inform* HA of spatial state (room occupied, anomaly at equipment, person detected) via the MQTT semantic state contract. HA then decides what to do with its devices. This boundary must not be crossed — HA has years of protocol integration work that cannot and should not be replicated.

**Spatial Devices (controlled by MinimapPR):**
Drones, PTZ cameras, directed speaker arrays, steerable spotlights, and other effectors whose meaningful operation depends on real-time track state and 3D position. These cannot be delegated to HA because HA has no concept of track state or probabilistic spatial position. A PTZ camera that must track a moving object through a room cannot be controlled by an HA automation — it needs the live track feed from MinimapPR. Drones may have no HA integration at all; they communicate via MAVLink or ROS2.

MinimapPR's `SpatialDevice` abstraction manages spatial devices directly under a **Rules of Engagement (RoE)** contract:
- Device receives updated pointing or routing commands when a track enters its zone of responsibility
- Device operations are constrained by exclusion zones, priority rules, and safety checks defined in the zone schema
- Device state feeds back into the COP as a tracked effector node — the COP shows where the PTZ is aimed, where the drone is, and what is being monitored

**Spatial Automations vs. HA Automations:**
- *HA automation*: "When motion sensor fires in kitchen, turn on kitchen light" — positionally fixed device, binary trigger, no spatial reasoning needed
- *Spatial automation* (MinimapPR rules engine): "When a track enters the boiler equipment proximity zone, begin acoustic baseline comparison; if anomaly score exceeds threshold, dispatch inspection drone to the boiler IFC position and alert HA notification service" — this requires live track state, 3D equipment positions, the equipment registry, and spatial device commands that HA cannot provide

The rules engine is the spatial automation engine. HA automations respond to MinimapPR's MQTT semantic state outputs the same way they respond to any sensor. The two systems compose cleanly without either needing to understand the other's internal data model.

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

## Part 7: Spatial Data Import and Environment Modeling

### 7.1 The Multi-Format Spatial Data Problem

The IFC digital twin and GeoJSON outdoor model provide *design intent* spatial knowledge — what was planned or constructed. Real deployments require additional layers of spatial truth: how a space was actually built (as-built laser scan), the exact 3D geometry of equipment for acoustic shadow and bounding-volume computation, and terrain data for outdoor acoustic propagation. The platform needs a unified **Spatial Asset Registry** that ingests from multiple sources and normalizes into a shared 3D scene representation used for visualization, acoustic modeling, and navigation mesh generation.

### 7.2 Supported Data Sources

**Structured Building and Site Models**
- IFC (`.ifc`): Rooms, equipment, structure — design intent, authoritative for static building data
- GeoJSON (`.geojson`): Outdoor parcels, field boundaries, site-scale features and equipment positions

**Equipment and Object Models (3D Geometry)**
- STEP (`.stp` / `.step`): Mechanical CAD format; precise manufacturing geometry for industrial equipment and appliances
- OBJ + MTL (`.obj`): General mesh format for furniture, equipment, terrain models
- glTF / GLB (`.gltf` / `.glb`): Web-native 3D format, optimized for the COP visualization layer (`@thatopen/components`, Three.js)

These allow a combine harvester, industrial press, or kitchen appliance to exist not just as a point in space but as a volumetric geometry — enabling acoustic shadow computation, proximity detection with actual bounding volumes, and true COP visualization. A 3D equipment model loaded alongside the IFC room geometry gives the acoustic engine an accurate model of how sound propagates around and through the object.

**As-Built and Survey Data (Point Clouds)**
- LAS / LAZ (`.las` / `.laz`): Industry-standard point clouds from terrestrial and aerial LIDAR scans
- PLY (`.ply`): Triangulated surface meshes from photogrammetry or structured-light scanning
- E57 (`.e57`): Professional scan interoperability format (Leica, FARO, NavVis scanners)
- PCD (`.pcd`): ROS2-native point cloud format for direct robotics integration

Point cloud data resolves the gap between design intent (IFC) and as-built reality: walls that shifted during construction, terrain that differs from the survey, equipment relocated after installation. For acoustic localization, room geometry must match physical reality — not the plans.

**Photogrammetry / Structure from Motion**
- Drone image sequences → dense point cloud via Colmap, ODM, or RealityCapture
- Particularly relevant for outdoor farm-scale terrain and large equipment site mapping
- Output: georeferenced point cloud (LAS/LAZ) and/or mesh (OBJ/glTF) at centimeter-level accuracy
- Enables sub-centimeter reconstruction of grain bins, barns, and field terrain from drone overflights

**Room Impulse Response (RIR) Measurements**
- Measured impulse response profiles (swept sine or MLS signal) at known speaker/microphone positions
- Capture the direct acoustic truth of how sound propagates in a specific space — reflection patterns, absorption, reverberation time (RT60) per room
- Uses: calibrate TDOA localization for multipath correction; improve classification in highly reverberant spaces; generate per-room acoustic maps for sensor placement optimization
- Can be computed by simulation (FDTD, image-source method) from point cloud geometry when direct measurement is impractical

### 7.3 The Acoustic Calibration and Labeling Interface

The combine harvester example illustrates a general challenge: when new equipment is registered in the spatial model, its acoustic signature at various operating states is unknown. No downloadable database exists for most industrial or agricultural equipment. A physical calibration process is required.

The **Acoustic Calibration UI** is critical infrastructure for all equipment health monitoring use cases:

**Guided Collection Mode:**
- Technician positions a microphone node near a piece of equipment
- System records acoustic data while equipment runs through defined operating states (idle, load, high-load, fault simulation if safe)
- Equipment position recorded from IFC model, GPS, or manual entry on floor plan
- Session saved as a labeled calibration dataset: `{equip_id, operating_state, position_m, timestamp, spectral_features[]}`

**Acoustic Profile Management:**
- Per-equipment acoustic profiles covering all defined operating states
- Baseline stored as statistical model: mean spectral features per band per state, covariance matrix for Mahalanobis anomaly scoring
- Calibration provenance metadata: when collected, by whom, equipment firmware/model at time of capture, environmental conditions
- Re-calibration workflow: flag profile as stale after service events or equipment changes, collect new data, merge or replace baseline

**Ground Truth Data for Comparative Analysis:**
- When an anomaly is detected, the agentic workflow (Part 8) can search for reference audio: manufacturer service documentation, equipment-specific forums, or audio analysis databases
- Collected reference data can be ingested as labeled comparison examples to refine local classifier thresholds

**Integration Points:**
- Calibration profiles stored alongside equipment entries in the zone/equipment JSON schema
- `acoustic_baseline_id` field in `Pset_ifcPlot_EquipmentLifecycle` references the calibration record
- Queryable via: `GET /api/v1/equipment/{equip_id}/calibration`

### 7.4 Point Cloud Integration for Acoustic Modeling and Navigation

Point cloud data serves two distinct purposes beyond visualization:

**Acoustic Space Modeling:**
Real room geometry significantly affects TDOA accuracy through multipath reflections. A point cloud of the actual space enables:
- Acoustic impulse response computation via image-source method or FDTD simulation
- Reflection-aware localization corrections for highly reverberant spaces (concrete warehouses, tile bathrooms, metal barns)
- Identification of acoustic dead zones where sensor coverage is poor due to geometry
- Optimal sensor placement simulation for new deployments before any hardware is installed

**Navigation Mesh Generation:**
For drone and robot deployment planning:
- Voxelize the point cloud (OctoMap or similar) to create obstacle representation
- Generate passable corridors from room geometry with configurable clearance envelopes per robot/drone size
- Export in standard formats: NavMesh (JSON), MAVLink polygon fences, or VDA 5050 node-edge graph

The platform does not need to perform all these computations internally. Its role is to be the data hub: aggregate point clouds, IFC models, and equipment geometry, and provide export APIs for specialized downstream tools (Blender, Isaac Sim, RViz2, NavMesh generators) to consume.

---

## Part 8: Agentic Integration Architecture

### 8.1 Design Principles

The agentic ecosystem in 2025–2026 is evolving rapidly, but several guiding principles are already clear:

1. **Real-time alerting and agentic reasoning are separate paths.** The rules engine fires in milliseconds; agents reason in seconds to minutes. Events requiring immediate action (fire, intrusion, safety zone violation) never wait for an agent. Events that benefit from contextual analysis (acoustic anomaly below hard threshold, unusual occupancy pattern, equipment degradation trend) enter the **Agentic Review Queue** for asynchronous processing.

2. **Agents do not have authority over life-safety systems.** An agent can recommend, summarize, and dispatch non-critical notifications. Only the rules engine — with explicit, human-authored Rules of Engagement — can command spatial devices or trigger safety relays.

3. **MinimapPR exposes structured data, not raw audio.** Agents receive labeled events, spectral feature vectors, track state, equipment context, and IFC spatial metadata. Raw audio streams route to the STT/classification layer; only the output (classified events, transcript fragments, spectral anomaly data) enters the agentic path.

4. **MCP is the agent-to-tool protocol.** MinimapPR implements an MCP (Model Context Protocol) server so that any MCP-compatible agent framework — Claude, LangGraph, AutoGen, custom — can query MinimapPR, the IFC model, and HA state in a standardized way without bespoke integration per agent type.

### 8.2 MinimapPR as MCP Server

MinimapPR exposes an MCP server as an additive layer alongside its REST API. MCP is an open JSON-RPC 2.0 standard for agent-to-tool communication — no changes to existing endpoints required.

**Exposed Resources (read-only context for agent reasoning):**
- `minimappr://context/current` — live operational snapshot: active tracks with room labels, equipment health summaries, recent alerts, HA state snapshot
- `minimappr://zones/{zone_id}` — zone definition with equipment list and current occupancy state
- `minimappr://equipment/{equip_id}` — equipment metadata, lifecycle data, acoustic baseline, anomaly history
- `minimappr://alerts/recent` — alert history with full spatial context, configurable lookback
- `minimappr://ifc/query` — parameterized spatial query: "what IFC elements are within 2 m of position X?"

**Exposed Tools (agent-callable actions):**
- `query_spatial_context(position_m, radius_m)` → returns IFC elements, zones, and equipment near a 3D position
- `get_equipment_health(equip_id)` → current anomaly score, trend direction, recent spectral observations
- `get_zone_occupancy(zone_id)` → current occupancy state, confidence, and recent track events
- `submit_review_result(event_id, finding, recommended_action)` → agent posts findings back to MinimapPR review queue; high-confidence findings may trigger a rules engine re-evaluation, but agents do not directly dispatch alerts
- `request_calibration_session(equip_id)` → schedules a technician-guided calibration collection

### 8.3 The Agentic Review Queue

The Agentic Review Queue is a buffer between MinimapPR's real-time event stream and agent-based analysis. It lets agents add contextual intelligence without being in the critical path.

**Queue Trigger Conditions** (rules engine disposition: "watch and evaluate"):
- Acoustic anomaly detected but below the hard alert threshold
- Equipment anomaly score in the "concern" band but not yet "critical"
- Unusual occupancy pattern (unusual hour, unusual duration, no matching rules)
- Voice event with moderate confidence: "help" heard but context is ambiguous

**Acoustic Anomaly Workflow (LangGraph orchestration):**
```
AnomalyEvent → AnomalyReviewGraph:
  1. fetch_context(event.position, event.equip_id)
       → MinimapPR MCP spatial query + IFC equipment lookup
  2. fetch_equipment_history(equip_id)
       → lifecycle record, service history, previous anomaly events
  3. [conditional] search_reference_audio(equip_type, anomaly_type)
       → web search: manufacturer service docs, equipment forums, audio analysis sources
  4. [conditional] compare_spectral_features(event.features, reference)
       → cosine similarity / Mahalanobis distance vs. reference profiles
  5. synthesize_finding(all_context)
       → LLM reasoning: what is the most probable cause given equipment age,
         anomaly signature, operating history, and reference comparison?
  6. route_finding(finding.severity):
       LOW    → log, no notification
       MEDIUM → HA advisory notification (informational)
       HIGH   → HA priority notification + MinimapPR alert entry
       CRITICAL → immediate escalation, spatial device dispatch if applicable
```

**Emergency Triage Workflow ("Help" Detection Example):**
```
VoiceEvent("help", confidence=0.72, zone="bedroom") → HelpTriageGraph:
  1. fetch_zone_context(zone_id)
       → occupancy, recent acoustic events, time of day, expected occupants
  2. fetch_ha_presence(zone_id)
       → HA person entities, last motion sensor activity, recent device interactions
  3. assess_urgency(combined_context):
       - Is the zone occupied? By whom? At what time?
       - Are there corroborating signals (distress audio pattern, elevated acoustic energy)?
       - Is this a known false positive source in this zone?
  4. LLM decision (extended thinking for complex cases):
       "Normal" disposition   → log, no notification sent
       "Uncertain" disposition → gentle check-in notification to occupant/caregiver
       "Emergency" disposition → immediate HA priority alert, optional spatial device dispatch
```

The critical principle: both workflows are fully asynchronous. The real-time rules engine has already evaluated each event and determined "not an immediate hard alert" before the queue receives it. Agent workflows complete in seconds to minutes with no impact on real-time responsiveness.

### 8.4 Audio and Language Capabilities

**Current state (2025–2026):**
- GPT-4o has native multimodal audio capability — can process audio frames directly alongside text in a single API call
- Claude (claude-sonnet-4-6, claude-opus-4-6) currently requires a transcription bridge: audio → Whisper → text + spectral metadata → Claude. This architectural limitation is expected to change
- For acoustic anomaly analysis, spectral feature vectors (MFCC, mel-spectrogram, per-band energy) are the correct representation for LLM reasoning — not raw waveforms. "Given this spectral anomaly signature and equipment context, what is the most likely failure mode?" is answerable; "analyze this WAV" is not (yet)
- Claude's extended thinking (internal reasoning up to 60 seconds for complex decisions) is well-suited to the emergency triage workflow where an unhurried but thorough contextual analysis is exactly what is needed

**Practical design:**
- Speech events: Whisper → transcript text → LLM for intent and context classification
- Acoustic anomalies: MinimapPR spectral features + anomaly score + IFC context → LLM for failure mode reasoning
- Reference comparison: agent fetches manufacturer documentation, forum discussions, or audio databases via MCP web search tool; compares against local spectral features
- Future native audio: tool interfaces should be designed to accept audio passthrough when available — the architecture should not assume transcription will always be required

### 8.5 Library Changes Required for Agentic Integration

**MinimapPR:**
- MCP server module (`minimappr/mcp/server.py`) — additive, no existing API changes
- Agentic review queue (`minimappr/core/review_queue.py`) — async queue with LangGraph-compatible event schema
- Event payloads extended with `agent_context` field: pre-assembled LLM-ready context bundle for each event
- Existing WebSocket extended with event type filter for "agentic-eligible" events (queue consumers subscribe to this filter)
- `GET /api/v1/context/current` endpoint (Part 6, already planned) is the primary MCP resource backing

**catlin-house / ifcplot:**
- Structured IFC spatial query function: `query_elements_near(position_m, radius_m, model)` → JSON list of IFC elements with type, name, GUID, property sets
- Exposed as both a Python API and an MCP tool so agents can call it without ifcopenshell knowledge

**New LLM Synthesis Service** (Part 6, new cross-system component):
- Implements LangGraph workflows for the Agentic Review Queue
- Hosts MCP client connections to MinimapPR, catlin-house IFC query API, and optionally HA
- Manages review queue consumer process — subscribes to MinimapPR WebSocket, dispatches to appropriate workflow graph per event type
- Maintains rolling context window of current operational state for proactive synthesis queries

---

## Part 9: Robotics and Autonomous Equipment Zone Routing

### 9.1 Extending Zone Logic for Autonomous Systems

MinimapPR's zone system already handles arbitrary 3D polygons for acoustic event attribution, occupancy tracking, and rule triggering. The same infrastructure is the correct foundation for autonomous equipment routing and access control — no parallel system needed. The zone schema requires only an extended `zone_type` taxonomy and associated behavioral contracts.

**Extended Zone Type Taxonomy:**

| Zone Type | Description | Relevant Standards |
|-----------|-------------|-------------------|
| `interior_space` | Room or area — occupancy and acoustic attribution | All |
| `equipment_proximity` | Acoustic health and anomaly attribution radius | All |
| `privacy_zone` | Privacy-mode region, restricted data retention | All |
| `robot_exclusion` | Hard exclusion — no autonomous ground robots may enter | VDA 5050, OpenRMF |
| `robot_slow` | Speed-reduced zone for ground robots (near humans, fragile items) | VDA 5050 |
| `robot_corridor` | Preferred routing path with inbound/outbound designation | VDA 5050, OpenRMF |
| `robot_reservation` | Dynamically reserved by an active robot, released on exit | VDA 5050, OpenRMF |
| `agri_section` | Field section for ISOBUS section control (crop row, treatment zone) | ISOBUS ISO 11783 |
| `agri_exclusion` | No-entry zone for agricultural equipment (habitat, drainage, sensitive terrain) | ISOBUS |
| `drone_exclusion` | Hard exclusion — no autonomous aerial systems may enter | MAVLink, UTM |
| `drone_slow` | Reduced-speed zone for drones (near structures, people, livestock) | MAVLink |
| `drone_corridor` | Preferred flight corridor with inbound/outbound designation | MAVLink, U-Space |
| `drone_inspection_point` | Named waypoint for inspection tasks | MAVLink |
| `utm_corridor` | Airspace corridor with altitude bounds, shared with UTM/U-Space services | ASTM F3411, U-Space |

The same zone polygon engine that handles acoustic occupancy also exports these zones to the appropriate robotics protocol — no separate spatial database is required.

### 9.2 Indoor Ground Robots — VDA 5050

VDA 5050 is the most mature standard for autonomous mobile robot (AMR) and automated guided vehicle (AGV) fleet management in indoor environments. It is widely adopted in warehousing and manufacturing, with open-source client libraries available in Python and Go.

**What VDA 5050 provides:**
- JSON/MQTT protocol between the Fleet Management System (FMS) and each robot
- Node-edge graph topology: robots navigate between named nodes via named edges — the navigation graph maps directly to a simplified zone/corridor layout
- Zone reservation and release: robot requests a zone segment before entering, holds it during transit, releases on exit — prevents collisions between robots
- Action system: pick, place, dock, wait, and custom actions as parameterized task sequences
- State reporting: position, velocity, battery state, errors, currently occupied zone

**MinimapPR integration:**
- Zone geometry → VDA 5050 node-edge graph export (zones become nodes, `robot_corridor` entries become edges)
- `robot_exclusion` and `robot_slow` zones → VDA 5050 zone blocking and speed override rules
- MinimapPR acoustic track detection in a `robot_corridor` → dynamic zone block message to FMS (human detected → robot hold)
- Robot position telemetry from VDA 5050 FMS → MinimapPR COP as `TELEMETRY` mobile nodes (same node type used for drones)

**OpenRMF** (Open Robotics Middleware Framework) provides a higher-level multi-robot coordination layer built on ROS2, covering space-time conflict scheduling and building-graph navigation. Appropriate for complex multi-robot environments where simple zone reservation is insufficient — OpenRMF and VDA 5050 are complementary, not competing.

### 9.3 Agricultural Equipment — ISOBUS

ISOBUS (ISO 11783) is the agricultural CAN bus standard governing how tractors and implements communicate: GPS receivers, section controllers, variable-rate application systems, and field terminals all use it. It is fundamentally a **physical bus**, not an IP network — integration with MinimapPR requires a gateway device.

**ISOBUS capabilities relevant to this architecture:**
- **Section control** (ISO 11783-7): Divide an implement boom into sections; the ISOBUS controller turns sections on/off based on GPS position and an application prescription map — direct field zone analog
- **Task data** (ISO 11783-11): Structured field operation records — what was applied, where, when, at what rate (ISOBUS XML task file format)
- **Telematics** (ISO 11783-10): Machine telemetry over cellular or satellite: GPS position, engine hours, fuel, fault codes
- **Variable Rate Application**: GIS prescription maps (shapefiles or GeoJSON) drive variable rate of seed, fertilizer, or chemical per GPS polygon

**Gateway integration strategy:**
ISOBUS is CAN-based and not IP-native. A certified gateway (Topcon, Trimble, or AEF-compliant ISOBUS bridge) translates to MQTT or REST:
- ISOBUS telematics → MQTT → MinimapPR `TELEMETRY` node (machine position, speed, status on COP)
- MinimapPR `agri_section` zones → ISOBUS prescription map format (zone polygons → application zone file)
- MinimapPR zone boundary crossing events → ISOBUS section control input (entering boundary → section on)
- ISOBUS task data exports → catlin-house `Pset_ifcPlot_EquipmentLifecycle` (operational history enrichment: hours, fuel, application records)

The priority is telemetry ingest (machine positions on COP) and field zone export (prescription map generation). Direct real-time ISOBUS command from MinimapPR is an advanced integration requiring farm-specific gateway hardware.

### 9.4 Aerial Drones — MAVLink and Flight Tube Routing

**MAVLink Protocol:**
MAVLink is the dominant protocol for small UAV control — ArduPilot, PX4, and most drone autopilots natively speak it. It is binary (not JSON), transported over UDP or serial, with a comprehensive message set covering navigation, telemetry, commands, and fences.

Relevant features:
- **Polygon geofences** (`FENCE_POINT` messages): define exclusion zones (no-fly), inclusion zones (must-stay-within), and altitude bounds — map directly to `drone_exclusion`, `drone_corridor`, and `utm_corridor` zone types
- **Waypoint navigation**: mission sequences as geo-referenced waypoints with altitude, action, and acceptance radius
- **Telemetry**: position, battery, attitude, flight mode at 4–10 Hz — suitable for COP display as `TELEMETRY` nodes
- **COMMAND_LONG**: command dispatch for takeoff, land, return-to-launch, mission start, and emergency stop

**MinimapPR ↔ MAVLink integration:**
- Drone position telemetry → MinimapPR COP as `TELEMETRY` nodes, zone-matched in real time
- MinimapPR `drone_exclusion` zones → MAVLink exclusion fence upload to autopilot
- Inspection task dispatch: rules engine identifies inspection need → generates waypoint mission sequence with IFC equipment position as target → uploads via MAVLink → monitors via telemetry
- Safety override: MinimapPR track detected in drone operating zone → `COMMAND_LONG` hold or RTL command

**Flight Tube Calculation:**
A flight tube is a 3D corridor — a swept volume along a path — through which a drone is cleared to navigate. Unlike a simple waypoint sequence, a flight tube defines the lateral and vertical clearance envelope around the intended path, enabling conformance monitoring and airspace deconfliction.

MinimapPR's approach to flight tubes:

1. **Static flight tubes**: defined in the zone schema as `drone_corridor` or `utm_corridor` with polygon cross-section and altitude bounds. These are the "outbound highway" and "inbound highway" corridors for routine routes (loading dock → charging station → inspection point).

2. **Dynamic flight tubes**: computed on-demand from current obstacle state (MinimapPR acoustic tracks, zone reservations, live telemetry of other drones) plus the IFC/point cloud navigation mesh. Path planning algorithms (RRT*, A* on voxel grid) produce a centerline; the clearance envelope is added based on drone size and wind margin.

3. **UTM/U-Space integration**: for beyond-visual-line-of-sight (BVLOS) operations, computed flight tubes are submitted to a UTM service as flight authorization requests.

**UTM and Airspace Standards:**
- **ASTM F3411 / F3548** (US): FAA Remote ID and UTM framework — REST/JSON API for drone flight intent, conflict detection, and airspace authorization
- **U-Space (EU EASA)**: European equivalent — UAS Flight Authorization Requests with flight tube geometry; real-time conformance monitoring via REST/SOAP services
- **SAE J3216** (Cooperative Driving Automation taxonomy): describes multi-agent intent sharing and coordination levels — its cooperative intent model is a useful architectural reference for how MinimapPR zones can express "this corridor is in use by agent X with priority Y," though J3216 itself is not a UAV-specific protocol

For farm and residential contexts: UTM registration may not be required for sub-250g drones operating within visual line of sight on private property. UTM integration is an optional advanced capability for larger or BVLOS-capable platforms.

### 9.5 The Outbound/Inbound Highway Corridor Concept

In any environment with regular autonomous equipment movement, ad-hoc path planning creates congestion and collision risk. Designated corridors with traffic flow direction — analogous to road lanes — are the proven solution.

MinimapPR zone schema extension for corridor routing:

```json
{
  "zone_id": "main_corridor_outbound",
  "zone_type": "robot_corridor",
  "corridor_direction": "outbound",
  "paired_corridor": "main_corridor_inbound",
  "priority": "high",
  "max_speed_mps": 2.0,
  "polygon_xy": [[...], [...]],
  "z_floor": 0.0,
  "z_ceiling": 2.5
}
```

**Behavioral contracts:**
- Outbound corridors carry traffic toward task areas (loading docks, inspection points, field sections)
- Inbound corridors carry return traffic toward home base, charging stations, or staging areas
- Intersection zones use priority rules based on corridor priority field, analogous to VDA 5050 zone reservation
- MinimapPR acoustic tracking detects humans in corridor zones → dynamic speed reduction or hold command to FMS/autopilot

The corridor system uses the same zone infrastructure as acoustic occupancy detection. The `zone_type` field determines which subsystems act: acoustic tracking, VDA 5050 node-edge export, MAVLink fence upload, or UTM corridor submission. No parallel spatial database needed.

---

## Part 10: The Recommended Roadmap

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
12. **MCP server module** — MinimapPR as MCP server for agentic tool access
13. **Agentic review queue** — async event queue with LangGraph orchestration for contextual anomaly analysis
14. **LLM synthesis service** — intelligence layer; hosts review queue workflows and MCP client connections
15. **Acoustic calibration UI** — guided collection mode for per-equipment operating-state acoustic profiles
16. **Farm/outdoor IFC extension module** — agricultural facility and field modeling
17. **Versioned zone export contract** — stable interface between spatial model and MinimapPR
18. **Extended zone type taxonomy** — `robot_corridor`, `drone_exclusion`, `utm_corridor`, `agri_section`, and related types

### Tier 4 — Future Capabilities (Architecturally Supported)
*These require external platforms (robot hardware, city-scale deployment) but the architecture should not prevent them.*

19. **IFC navigation mesh export** — voxelized point-cloud-derived navigation mesh for drone/robot path planning
20. **Point cloud ingest pipeline** — LAS/LAZ/E57 import, voxelization, acoustic space simulation
21. **VDA 5050 node-edge graph export** — zone geometry → robot FMS navigation graph
22. **ISOBUS gateway integration** — agricultural machine telemetry ingest and prescription map export
23. **MAVLink geofence sync** — automatic `drone_exclusion`/`drone_corridor` zone upload to drone autopilots
24. **UTM/U-Space flight tube submission** — dynamic flight tube computation and BVLOS airspace authorization
25. **Satellite/cellular MQTT** — conservation and remote farm deployments
26. **Multi-site federation** — retail chains, large farms, military area operations
27. **Full LLM-IFC query interface** — LLM agent makes `ifcopenshell` spatial query tool calls

---

## Closing Assessment

The most important insight from this analysis is that **the three systems form a capability triad that is genuinely greater than the sum of its parts**, and the multiplier applies across every domain where physical space matters — which is nearly every domain of human activity.

The immediate practical priority is Tier 1: getting the IFC knowledge base connected to MinimapPR's spatial reasoning and the MQTT semantic state contract connected to HA. This alone produces the full residential smart home synthesis.

The highest-leverage single new capability is **acoustic baseline profiling**. It is technically straightforward (statistical modeling of spectral features over time), has zero conflict with any existing function, and unlocks predictive maintenance value in every vertical — from protecting a combine at harvest, to catching a failing hospital ventilator, to flagging a compressor a week before it shuts down a cold storage warehouse.

The LLM synthesis service is the emergent intelligence layer that becomes possible once the other pieces are in place. Its value compounds with the quality of the underlying structured context — and the structured context (IFC spatial model + MinimapPR event history + HA device state) is uniquely rich precisely because of how these three systems complement each other.
