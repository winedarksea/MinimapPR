# MinimapPR, proposed architecture

Realtime sensor awareness — a Common Operating Picture, inspired by military COP and Link 16 concepts, designed to tie into Home Assistant with sound localization as the initial core tracking technology.

MinimapPR was inspired in part by BirdNetPi, but with the added goal of sound localization. The name combines "minimap" (as seen in video games) with "PR" for probabilistic reckoning. Sound streaming from microphones is probabilistically localized, processed sound from each location is fed to classifiers, and each classified sound is tracked as an event on a map — like radar, but sound based.

## Use Cases

The goal of MinimapPR is to create a true "common operating picture" for all users (from residential to commerical), tied into sound localization as the first phase tracking instrument (later including much more). It aims to draw heavily from lessons learned from military COP systems (even direclty copy them), and even integrate with ATAK type systems directly.

The system is designed to serve a wide range of acoustic and multi-sensor awareness scenarios:

### Environmental & Wildlife
- **Bird and wildlife monitoring** — BirdNET-style classification with added localization, tracking birds as they sing and fly. Track predator/prey interactions (e.g., coyotes approaching livestock, hawks approaching poultry).
- **Environmental research** — long-term acoustic ecology monitoring, biodiversity surveys, habitat usage mapping.
- **Noise pollution monitoring** — industrial equipment, construction sites, events, traffic. Generate sound intensity heatmaps with localized source attribution.

### Security & Safety
- **Gunshot / explosion detection** — with localization, confidence scoring, and immediate alerting. Includes artillery fire, fusillade, and fireworks discrimination.
- **Alert sound detection** — fire alarms (T3 pattern), civil defense sirens, smoke detectors, screams.
- **Perimeter monitoring** — detect and track movement (people, vehicles, drones) with IFF gating to suppress alerts for known friendlies.
- **Incident detection at events** — localize crowd distress (shouting, calls for help) and route first responders via location.

### Military & Defense
- **Drone and heavy equipment detection** — acoustic classification and tracking of UAVs, vehicles, and aircraft.
- **Gunfire and artillery localization** — with passive/active localization using echoes to map terrain.
- **Speech detection by language** — classify detected speech by language for intelligence gathering.
- **Integration with ATAK/TAK ecosystem** — CoT interchange for interoperability with existing military COP systems.
- **Mil-Spec Readiness** — Native MGRS support, Built-In Test (BIT) reporting, and tamper-evident data provenance for certification and interoperability.

### Retail & Commercial
- **Activity and foot traffic mapping** — sound-based occupancy heatmaps for retail spaces, parks, museums.
- **Customer sentiment mapping** — extract speech, process via STT → LLM to assess mood and topics by location, helping optimize display arrangements.
- **Conversational memory** — store useful information from overheard conversations ("are these leftovers being saved for dinner?") and answer with an LLM.

### Smart Home & Agriculture
- **Home Assistant integration** — alerts by class, speed, position. Familiar pet and people recognition.
- **Livestock tracking and protection** — detect predators, track herd movement, alert on anomalous sounds.
- **Voice assistant integration** — wake word detection → speech-to-text → Home Assistant Voice (via Rhasspy/Wyoming).

### Mapping & Navigation
- **Passive acoustic terrain mapping** — use echoes (e.g., from gunshots, thunder) to gradually build 3D terrain models.
- **UAV navigation aid** — acoustic localization as a supplementary navigation source.
- **Microclimate mapping** — temperature, pressure, and wind data from distributed nodes to map local conditions.

### Seismology & Geophysics
- **Ground vibration monitoring** — IMU/accelerometer data from ground-mounted nodes for seismic event detection and localization.
- **Infrasound detection** — low-frequency pressure waves for weather, volcanic, and industrial monitoring.

## Architecture

The system has three tiers: **nodes**, **backend server(s)**, and a **frontend**.

- **Nodes** are sensors that stream data upstream. They range from extremely minimal solar-powered sensors (stream-only, no local processing) to capable array nodes that can run local fusion and classification.
- **Backend** handles sensor fusion, tracking, classification, alerting, and storage.
- **Frontend** provides the COP map view. Initially very minimal, with focus on a highly featured backend.

All three tiers may run on a single device (e.g., a Raspberry Pi 5 with an attached array node) or be distributed across many devices.

### Fusion Topology

There is one server role: a **fusion server**. Every fusion server runs the same full backend logic — ingestion, localization, classification, tracking, alerting, and storage. A deployment with a single fusion server is effectively centralized; a deployment with multiple fusion servers is federated.

**Peer-based Federation (Link 16 Pattern):** 
In federated mode, fusion servers operate as peers rather than in a rigid hierarchy. This matches the "common operating picture" (COP) concept where every participant shares their local perspective to build a global one. This is the wisest path for the project's goal of "splittable/combinable" networks.

- **Fault tolerance and segmentability** — If a fusion server goes offline or is physically separated (e.g., a mobile segment moves out of range), its peers continue operating independently. When segments recombine, they automatically resume track exchange. 
- **Deconfliction via Track Quality (TQ)** — To prevent "track ghosting" (where multiple peers report the same source as separate tracks), the system uses a **Track Quality** metric. Peers exchange tracks including a TQ score (based on confidence, sensor count, and age). If two received tracks correlate spatially, the server with the higher TQ "owns" the track for the shared COP; the other maintains its track as a passive standby.
- **Reduced bandwidth** — Point nodes stream raw audio only to their local fusion server. Only lightweight fused tracks are exchanged between peers, making the architecture highly scalable over long-range or low-bandwidth links.

To enable this, the data model uses the same message formats for both raw observations and fused tracks, distinguished by a `source_type` field (e.g., `raw_sensor`, `local_track`, `peer_track`). Downstream consumers (rules engine, alerting, COP view) do not need to know the topology.

**Role of Array Nodes:** 
Point nodes (ESP32, Pico W) are too resource-constrained to run server logic; they always stream raw sensor data to an assigned fusion server. Array nodes on more capable hardware (e.g., Raspberry Pi) *can* run the fusion server software locally. An array node running fusion that has no configured peer links behaves as a standalone centralized server. Adding peer links turns it into a federated participant with no code changes.

### Heterogeneous Nodes

Not all nodes are equal. Some edge nodes (e.g., ESP32 point nodes on solar power) are extremely minimal and can only stream raw sensor data. Others (e.g., array nodes on a Raspberry Pi) can run local localization and classification. The system must handle this heterogeneity gracefully — a node declares its capabilities, and the backend adapts accordingly.

## Data Model

### Nodes

Every node in the system is registered with its identity, location, capabilities, and health state. This is the foundation for fusion, cueing, health monitoring, and degradation analysis.

- **Node ID** — unique, persistent identifier
- **Node type** — `point`, `array`, `gateway`, etc.
- **Capabilities** — declared set: `audio_stream`, `local_fusion`, `classification`, `gps`, `imu`, `ble_scan`, `temperature`, `pressure`, `speaker`, `camera`, etc.
- **Location** (lat/long/altitude) — hardcoded or live GPS, with location uncertainty
- **Mobility** — `stationary` or `mobile`
- **Status** — `online`, `degraded`, `offline`
- **Last heartbeat** timestamp
- **Hardware metadata** — firmware version, MCU type, mic type, power source, etc.

### Observations

Raw sensor data. Each observation carries:

- **Timestamp** with a **time quality indicator** (`gps_locked`, `ntp_sync`, `freerunning`) so fusion algorithms can weight by time certainty.
- **Time of Applicability (TOA) vs. Time of Receipt (TOR)** — always distinguish when the sensor observed something vs. when the server received it.
- **Sensor type** (audio, accelerometer, pressure, temperature, BLE, etc.)
- **Value(s)**
- **Location** (lat/long/altitude) with location uncertainty
- **Node ID**

Raw observations should be kept in sensor-local measurement space (e.g., TDOA, bearing) as long as possible. Conversion to geographic coordinates happens at the fusion stage to preserve the uncertainty structure.

### Environment

Environmental readings are separated from high-volume audio streams because they are tiny per-record
, useful long-term, and directly feed localization corrections and use cases like microclimate mapping.

- **Node ID**, **Timestamp**
- **Temperature** (K) — feeds speed-of-sound calculation
- **Pressure** (Pa) — atmosphere mapping, wind estimation, altitude
- **Humidity** (%)
- **Wind speed / direction** (derived from pressure array or mic array correlation)
- **Solar lux**

Retention is long-term. Records are small and serve ongoing calibration, trend analysis, and microclimate mapping.

### Detections

A classified event tied to a localization and (optionally) a source audio extract.

- **Detection ID**
- **Timestamp** (TOA + TOR)
- **Location** + location uncertainty (geographic, converted from measurement-space at fusion time)
- **Source node(s)** — list of contributing node IDs
- **Source observation references** — provenance link to raw observations that produced this detection
- **SPL** (sound pressure level in dB)
- **Label ID** — foreign key to Labels table
- **Classification confidence** — If no classifier produces a result above a configured threshold, a default `Unknown` label can be assigned, allowing the event to be tracked without a specific class.
- **Audio extract reference** — file path or blob reference to the extracted clip (if retained)
- **Track ID** — foreign key to assigned track, if any (NULL for unassociated detections)
- **Retention class** — `default`, `promoted`, `permanent` — controls cleanup behavior. Maps to storage tiers: `default` → Short (~1 month), `promoted` → Long, `permanent` → Long/permanent. See Storage & Retention section for tier definitions.

The provenance chain — raw observation → detection → track → alert — is essential for debugging, review, and accountability.

### Tracks

A track is a fused, persistent entity representing something being monitored. Tracks follow a lifecycle:

| State | Meaning |
|---|---|
| **Tentative** | Single detection, not yet corroborated |
| **Confirmed** | Multiple detections corroborate existence |
| **Coasting** | No recent updates, position predicted only |
| **Dropped** | Stale beyond threshold, removed from COP |

Each track carries:
- **Track ID** — prefixed with the originating node/server ID to avoid collisions in distributed fusion (e.g., `srv1-T0042`)
- **Track Quality Index (TQI)** — a composite score from position uncertainty, classification confidence, age-of-last-update, and number of corroborating sensors
- **Label ID** — foreign key to Labels
- **Classification confidence**
- **IFF category** (see Blue Force section)
- **Current state** (position, velocity, covariance)
- **Source type** — `raw_sensor`, `local_track`, `peer_track` — supports federated fusion by distinguishing locally-produced from peer-received tracks

Tracking initially uses a Kalman Filter, with later expansion to Multi-Hypothesis Tracking (MHT) or JPDA for complex multi-target scenarios. All tracks are always probabilistic.

**Track Updates** record each state change: the detection(s) that caused it, updated position, TQI, and timestamp. This is the join between tracks and detections and provides the track history needed for replay.

### Pings

A **ping** is a lightweight derived record — a timestamped presence indicator for a source at a location. Pings are the atomic unit for heatmap generation, activity mapping, noise monitoring trends, and long-term analytics. They are far cheaper to store than full detections or observations.

Each ping carries:
- **Timestamp**
- **Location** + location uncertainty
- **Ping type** — `acoustic`, `ble_device`, `wifi_device`, `vibration`, `checkin`, etc.
- **Label ID** (optional) — what was detected (e.g., `bird`, `speech`, `device:AA:BB:CC`)
- **SPL** (optional) — for noise monitoring trends
- **Source detection or observation ID** — provenance

Pings are extracted from:
- Acoustic detections (a classified sound at a location)
- BLE/WiFi device sightings
- Motion/vibration threshold crossings
- Explicit check-ins from mobile nodes

### Labels

A unified classification taxonomy used by all classifiers, rules, and fingerprints. Hierarchical, following the [Google AudioSet](https://research.google.com/audioset/) ontology as the base, extensible with site-specific and model-specific labels.

- **Label ID**
- **Name** — e.g., `Gunshot`, `Bird`, `Turdus migratorius` (American Robin)
- **Category** — `security`, `safety`, `wildlife`, `human`, `vehicle`, `weather`, `environment`, etc.
- **Parent label ID** — for hierarchy (e.g., `Bark` → `Dog` → `Wildlife`)
- **Source** — which model or system defines this label (`yamnet`, `birdnet`, `fingerprint`, `user`, `dcase`)

The rules engine, alerts, IFF, and spectral fingerprints all reference labels by ID, ensuring consistent naming across the system.

### Zones

Spatial regions used by the rules engine, IFF exclusion logic, and coverage analysis.

- **Zone ID**
- **Name** — e.g., `perimeter`, `house_interior`, `chicken_coop`, `parking_lot`
- **Geometry** — polygon (lat/long vertices), optionally with altitude bounds
- **Zone type** — `alert_zone`, `exclusion_zone`, `coverage_zone`, `interest_zone`
- **Properties** — key-value pairs for zone-specific config (e.g., suppressed label list for exclusion zones)

### Known Entities (Blue Force)

Registry of known-friendly entities for IFF gating.

- **Entity ID**
- **Name** — e.g., `Colin's iPhone`, `UPS truck`, `neighbor's dog Max`
- **Identifiers** — list of associated IDs: BLE MAC, WiFi MAC, spectral fingerprint ID, etc.
- **IFF category** — default classification (`friendly`, `assumed_friend`)
- **Properties** — schedule, expected zones, notes

### Alerts

History of fired alerts, providing an audit trail and preventing duplicate alerting.

- **Alert ID**
- **Timestamp**
- **Rule ID** — which rule fired
- **Detection ID** and/or **Track ID** — what triggered it
- **Priority**
- **Destination** — where it was sent (Home Assistant, push notification, TAK, etc.)
- **Status** — `sent`, `acknowledged`, `dismissed`, `escalated`

### Annotations (Future)

User-provided labels on tracks, detections, or locations. Feed the spectral fingerprint database and local model fine-tuning.

- **Annotation ID**
- **Target type** — `track`, `detection`, `location`
- **Target ID**
- **Text** — free-form note (e.g., "neighbor's dog, barks every morning at 7am")
- **Label ID** (optional) — if the annotation assigns or corrects a classification
- **Timestamp**, **User**

### Text Extracts (Future)

For use cases like conversational memory and sentiment mapping, speech detections can be processed through STT and stored as text linked to a location and time.

- **Extract ID**
- **Detection ID** — source detection
- **Timestamp**, **Location**
- **Text** — transcribed speech
- **Language** (detected)
- **Sentiment** (optional, from LLM)
- **Keywords / entities** (optional, from LLM)

Retention must be carefully managed for privacy. May require explicit opt-in per zone or deployment.

### Database Summary

| Table | Contents | Retention | Volume |
|---|---|---|---|
| **Nodes** | Node registry, capabilities, health | Permanent | Tiny |
| **Observations** | Raw sensor data (audio chunks, accelerometer) | Short (hours to days) | Very high |
| **Environment** | Temperature, pressure, humidity, wind, lux | Long-term / permanent | Low |
| **Detections** | Classified events with audio extract refs, location, SPL | Medium (default ~1 month); promoted detections longer | Medium |
| **Tracks** | Fused track current state | Permanent (active); dropped tracks archived | Low |
| **Track Updates** | Per-update state changes linking detections to tracks | Long-term | Medium |
| **Pings** | Lightweight location+time+type+SPL summary records | Long-term / permanent | Medium |
| **Labels** | Classification taxonomy (hierarchical) | Permanent | Tiny |
| **Zones** | Spatial regions for rules, IFF, coverage | Permanent | Tiny |
| **Known Entities** | Blue force / IFF registry | Permanent | Tiny |
| **Alerts** | Alert history (rule, trigger, destination, status) | Long-term | Low |
| **Annotations** | User notes on tracks/detections (future) | Permanent | Tiny |
| **Text Extracts** | STT transcriptions linked to locations (future) | Configurable, privacy-sensitive | Low |

SQLite is appropriate for testing and light production. The schema should be straightforward to migrate to PostgreSQL (with PostGIS for spatial queries on zones and locations) for larger deployments.

### Provenance & Replay

Every derived product (track, classification, alert) links back to the source observations that created it via explicit foreign keys. The chain is: Observation → Detection → Track Update → Track → Alert. The system should store enough to fully replay past states for algorithm tuning, review, and debugging.

## Time Synchronization

Accurate timing across nodes is critical for acoustic localization. The initial version assumes each networked point node has a GPS connection with PPS for high precision timing. Each timestamp should carry a time quality indicator so the fusion engine knows how much to trust it.

Future versions may add PTP, ClockSync, or other mechanisms for environments with limited GPS availability. The initial design does not need to be GPS-jam-resistant, but should not make assumptions that would prevent adding alternative time sources later.

For testing, validate time synchronization by streaming known patterns (sine waves) and verifying no gaps or drift in the captured data.

## Example Localization Algorithms

Acoustic source localization is the core differentiator of this system. The following algorithms should be supported, with a pluggable architecture allowing selection based on array geometry, computational budget, and scenario:

### Time-Difference-of-Arrival (TDOA) Methods
- **GCC-PHAT** (Generalized Cross-Correlation with Phase Transform) — the baseline algorithm. Robust to reverberation, computationally lightweight. Suitable for real-time use on both edge and server.
- **SRP-PHAT** (Steered Response Power with Phase Transform) — grid-based search using steered beamforming. More computationally expensive but produces a spatial power map directly useful for the COP heatmap view.
- **SVD-PHAT** — SVD-based variant for improved robustness in high-noise environments.

### Subspace Methods
- **MUSIC** (Multiple Signal Classification) — high-resolution DOA estimation. Computationally heavier but excels at resolving closely spaced sources. A neural network approximation of MUSIC may be explored for real-time edge use.
- **ESPRIT** (Estimation of Signal Parameters via Rotational Invariance Techniques) — efficient subspace method, particularly suited to uniform array geometries.

### Beamforming
- **MVDR** (Minimum Variance Distortionless Response) beamforming — for extracting audio from a specific direction while suppressing interference. Used both for localization refinement and for producing beamformed audio extracts to feed classifiers.
- **Delay-and-sum beamforming** — simpler baseline for initial implementations.

### Environmental Corrections
- **Speed of sound calculation** — derived from local temperature and humidity. The dry-air baseline is $c = \sqrt{1.4 \times 287.053 \times T_K}$ where $T_K$ is temperature in Kelvin. Humidity affects speed of sound non-trivially (up to ~0.5% at high humidity, translating to ~1.7 m/s — enough to shift TDOA localizations by decimeters at longer ranges). Where humidity sensors are present, their readings should be incorporated into the correction. Temperature sensors on the network feed this calculation, when available, critical for accurate TDOA at longer ranges.
- **Wind gradient compensation** — pressure sensor data from distributed nodes can estimate wind direction and speed, which affects sound propagation paths. Initially a correction factor; future versions may use full ray-tracing models.
- **Filter out-of-band frequencies** — apply configurable bandpass filtering before localization to reduce noise (default: remove below 50 Hz and above usable range; adjustable for local mains frequency and deployment conditions). Subtract accelerometer or barometer signals to remove vibration artifacts.

### Networked vs. Local Localization
For **array nodes** (e.g., Sirith Tetrahedral), localization runs locally using the array's own microphones with known geometry. For **networked point nodes**, TDOA localization is performed at the backend server using cross-correlated signals from multiple nodes, requiring precise time synchronization.

A single array node may simultaneously produce local (array-level) localizations and contribute its averaged signal to the network-level localization — two independent localization sources from one device.

Subgroupings of point nodes may be required for large networks, with a layer above that filters and combines separate probabilistic localizations (distributed fusion architecture). For areas with buildings or large separations between node clusters, Multi-Hypothesis Tracking (MHT) or Joint Probabilistic Data Association (JPDA) handles the association problem.

Related implementations: [pyroomacoustics](https://pyroomacoustics.readthedocs.io/), [ODAS](https://github.com/introlab/odas), [OpenSoundscape](http://opensoundscape.org/), [CGrassin acoustic beamforming](https://github.com/CGrassin/acoustic_beamforming). Also see [Bayesian focusing](https://blogs.sw.siemens.com/simcenter/bayesian-focusing-allrounder-for-localization-and-quantification-of-sound-sources/) for advanced localization and quantification.

## Signal Processing Pipeline

### Sampling Rate

Standard operation at 48 kHz. For ultrasonic capture (bat monitoring, ultrasonic sensors, leak detection), 96 kHz or higher sampling rate is required — MEMS microphones are better suited for ultrasonic frequencies than electret.

### Core Processing Pipeline

The central data flow of the system is a streaming pipeline:

1. **Ingest** — Time-stamped audio (lossless PCM) and sensor data streams arrive from nodes. Each frame carries a timestamp with time quality indicator and node identity.
2. **Localization** — TDOA cross-correlation (GCC-PHAT baseline) across node streams, or local array-level DOA estimation. Produces measurement-space observations (bearings, TDOAs) with uncertainty.
3. **Fusion** — Observations are associated and fused into geographic coordinates. Kalman filter updates existing tracks or initiates new tentative tracks. Track lifecycle management (tentative → confirmed → coasting → dropped).
4. **Classification** — Audio extracts (raw or beamformed toward a localized source) are passed to classifiers (YAMNet, then optionally chained to BirdNET, STT, etc.). Classification labels attach to detections and propagate to tracks.
5. **Rules & Alerting** — Detections and track updates are evaluated against user-defined rules. Matching rules fire alerts, which are routed to configured destinations (Home Assistant, push notifications, CoT/TAK, etc.).
6. **Storage** — Raw observations expire quickly (ephemeral tier). Detections, tracks, pings, and environment data persist per retention policy. Promotion rules can escalate data to longer retention based on classification.
7. **Output** — COP map view updated. Optionally: ambisonic soundscape rendering, heatmap generation, telemetry export.

Steps 2–5 run as a continuous streaming pipeline. Backpressure, failure handling, and queue management are implementation concerns that should be addressed early — if a classifier is slow, it must not block localization. Each stage should be independently scalable and loosely coupled (message passing between stages).

### Streaming & Timestamps

The critical requirement for all audio streams is **perfect timestamping** — every audio frame must carry a high-precision timestamp tied to the node's time source (GPS PPS where available). Without this, TDOA localization across networked nodes is impossible.

Every stream message carries:
- **Timestamp** (with time quality indicator: `gps_locked`, `ntp_sync`, `freerunning`)
- **Type** — `audio`, `accelerometer`, `pressure`, `temperature`, etc.
- **Value(s)**
- **Location** (lat/long) + location uncertainty + altitude
- **Node ID** (multiple tags optional)

Additional stream types (e.g., `camera`, `radar`, `ble_scan`) may be defined as multi-source awareness expands (see Phasing & Prioritization).

### Wind Speed Measurement

The microphone array itself can be used to estimate wind speed and direction by analyzing correlated low-frequency noise patterns across the array elements. This supplements dedicated pressure sensors for atmospheric profiling.

## Interchange Format & Protocols

**Cursor on Target (CoT)** is the primary interchange format for events and tracks. CoT is the XML protocol used by ATAK (Android Team Awareness Kit), providing a well-tested schema for situational awareness with fields for position, type, detail, track, and stale times. Using CoT gives interoperability with the TAK ecosystem (ATAK, WinTAK, iTAK, TAK Server) at low cost.

CoT's hierarchical `type` field (based on MIL-STD-2525C symbology, e.g., `a-f-G-U-C`) maps well to the classification hierarchy used here.

Transport protocols: MQTT for lightweight sensor streaming, WebSockets for frontend, UDP where low latency is critical. MAVLink may be used for node control in future versions.

Generally, prefer high quality open source standard protocols. Military protocol patterns (Link 16, CoT) are referenced because they are the most battle-tested designs for these use cases.

## Node Hardware

### Point Node

A simple ESP32 point node (extensible to Raspberry Pi Pico W) with a single I2S microphone and a GPS module (PPS + UART TX/RX). No on-device processing — the goal is high quality networked audio with precisely linked timestamps. Future versions may add temperature, IMU, BLE scanning, and other sensors. Point nodes are always networked with other nodes.

For stationary point nodes, the GPS should use a high degree of location smoothing. Mobile nodes must disable location smoothing and stream GPS location continuously (see Moving Nodes).

### High-Fidelity Point Node

A custom design optimized for maximum SNR and timing precision:
- **MCU**: Raspberry Pi Pico 2 W (RP2350) — PIO subsystem enables <1 µs timing precision by syncing audio clock domains with GPS PPS.
- **Microphone**: Infineon IM73A135 (PDM) — 73 dB SNR, 135 dBSPL AOP. PDM decoded via PIO.
- **GPS**: u-blox SAM-M10Q — PPS signal for timing, "Stationary" dynamic model.
- **Environment**: Bosch BME280 — Temperature/Humidity for speed-of-sound correction.
- **Power**: CN3791 MPPT Solar Charger + 21700 LiPo cell.

### Array Node: Sirith Tetrahedral

A four-microphone array in a tetrahedral geometry (configurable spacing, default 50 mm between microphones). At 50 mm spacing, the spatial aliasing limit is ~3.4 kHz. While some target sounds (bird vocalizations, alarms) have harmonics well above this, the dominant spectral energy for most targets of interest — including most bird song fundamentals — falls within the localizable range, which should provide adequate localization accuracy for the primary use cases. Wider spacing can be configured for lower-frequency-only applications; tighter spacing can extend coverage to higher frequencies at the cost of reduced aperture. High quality MEMS microphones processed by the ADAU7112 ([datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ADAU7112.pdf)), configurable via jumpers to output 2 separate I2S streams or a single TDM stream. Connects to any AdaFruit Feather or Raspberry Pi Pico 2 W.

The Sirith board should also support a Teensy footprint, which has dedicated I2S/PDM/TDM pins and established audio library support (including quad-channel USB for testing).

GPS is optional on the Sirith. It can run standalone as a single-node system — with all four microphones hardwired, precise cross-network timing is not required.

An array node may produce localizations from its own array, while also contributing an averaged single-channel signal as a point node to the wider network for a separate, network-level localization.

### Node Potential Sensor Suite

Each node may include a subset of:
- **GPS** (location + PPS timing)
- **Accelerometer / IMU** (vibration sensing, seismic events)
- **Microphone(s)** (2× for noise cancellation or basic ranging, 4× via I2S)
- **Barometer** (BMP390 — pressure, temperature, altitude)
- **Humidity sensor**
- **Solar lux / photodiode**
- **BLE scanner** (device presence detection, via ESP32-Paxcounter patterns)
- **Speaker** (for self-localization, alerts, or directional audio output)

### Moving Nodes

Some nodes may be mobile (vehicle-mounted, drone-carried, or animal-worn). Moving nodes:
- Must stream GPS location continuously as part of their data stream
- Have higher location uncertainty that changes dynamically
- May have IMU data used to supplement GPS for dead reckoning during GPS gaps
- Are marked as `mobile` in their capability declaration so the fusion engine weights their contributions appropriately

## Classification Pipeline

The initial classifier uses **YAMNet** (TensorFlow / TensorFlow Lite). The design should support:

- **Model swapping** — ability to replace or update the classifier without code changes
- **Model chaining** — e.g., audio classified as speech → speech-to-text model → if wake word detected → Home Assistant Voice agent (via Rhasspy/Wyoming)
- **Multiple input sources** — classifiers may run on beamformed audio from a specific location, raw audio sums from nearby nodes, or other processed signals
- **Shared signal processing** — intermediate processing (noise filtering, normalization) should use standardized, reusable components across pipelines
- **Triggered classifiers** — a classifier may be triggered by an upstream classifier (e.g., speech detected → activate language identification model)

### Example Priority Labels

From [YAMNet class map](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv), these labels are examples of primary interest:

| Category | Labels |
|---|---|
| **Security** | Gunshot/gunfire, Explosion, Fusillade, Artillery fire, Civil defense siren, Siren |
| **Safety** | Smoke detector/smoke alarm, Fire alarm, Scream |
| **Wildlife** | Bird, Cat, Meow, Dog, Bark, Howl, Yip |
| **Human** | Speech, Conversation |
| **Vehicles** | Car passing by, Police car (siren), Train horn, Train whistle, Jet engine, Aircraft engine, Aircraft, Fixed-wing aircraft/airplane |
| **Weather** | Thunder |
| **Pyrotechnics** | Fireworks |

Additional TFLite models should be built or adapted for: specific alarm patterns (T3 fire alarm cadence), drone motor signatures, and language identification on speech segments.

### BirdNET Integration

[BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) (and [BirdCAGE](https://github.com/mmcc-xx/BirdCAGE)) should be available as an alternative or supplementary classifier for avian detections, providing species-level identification that YAMNet cannot match. BirdNET runs as a chained model: YAMNet or a simple energy detector identifies "bird" → BirdNET classifies species.

### DCASE Alignment

Align classification taxonomy and evaluation methodology with the [DCASE](https://dcase.community/) (Detection and Classification of Acoustic Scenes and Events) community where feasible. Target integration with DCASE 2026 challenge outputs for best-in-class models. Use [Google AudioSet](https://research.google.com/audioset/) ontology as the base classification hierarchy.
These two tasks of this competition in particular are most relevant:
Task 3: This task introduces a paradigm shift in sound event localization: Acoustic Imaging SELD. Moving beyond traditional sparse vector estimation, participants are challenged to model the acoustic field as high-resolution, dense energy maps. Using the STARSS dataset at its full 32-channel resolution for training, the goal is to perform acoustic super-resolution: reconstructing high-fidelity semantic energy maps from standard 4-channel (First-Order Ambisonics) inputs. Participants will output dynamic polygon masks that simultaneously encode localized event class and instantaneous acoustic energy. This task bridges the gap between audio signal processing and computer vision, inviting the use of modern approaches that are directly applicable to both modalities but for complex acoustic scene analysis.
Task 4: The spatial semantic segmentation of sound scenes (S5) task proposal aims to enhance technologies for sound event detection and separation from multi-channel input signals that mix multiple sound events with spatial information. This is a fundamental basis of immersive communication. The ultimate goal is to separate sound event signals with 6 Degrees of Freedom (6DoF) information into dry sound object signals and metadata about the object type (sound event class) and representing spatial information, including direction. This task is a sequel to DCASE 2025 Task 4, but it introduces two new challenges. One involves handling overlapping sound sources primarily within the same class, while the other addresses cases where the separation target is absent. These are essential for real-world applications and highlight new research challenges.

### Spectral Fingerprinting

Known recurring sounds (specific HVAC units, particular vehicles, a neighbor's dog) can be fingerprinted spectrally. Once fingerprinted, these sounds are automatically re-identified without requiring the general classifier, reducing compute and improving accuracy for site-specific patterns. User annotations (see below) feed this fingerprint database.

## Alerting & Rules

Rather than hardcoded alert classes, alerting should use a **rules engine** where users define conditions, for example:

> IF classification == 'gunshot' AND confidence > 0.7 AND zone == 'perimeter' THEN alert('security', priority=1)

This can start simple (config file with basic rules) and grow to support complex logic. Alerts may route to Home Assistant, push notifications, or other destinations.

## Blue Force / IFF

Known-friendly entities (phones via BLE, registered vehicles, known people) generate tracks in a separate "blue" layer. These are used for **gating** — suppressing alerts when a detection correlates with a known friendly.

IFF confidence levels, loosely following NATO conventions:

| Level | Meaning |
|---|---|
| **Friendly** | Positively identified as known/friendly |
| **Assumed Friend** | Likely friendly based on behavior or partial match |
| **Unknown** | No identification |
| **Suspect** | Behavior or classification suggests possible threat |
| **Hostile** | Confirmed threat (e.g., positive gunshot classification) |

**Exclusion zones** can suppress specific classifications in areas where they are expected (e.g., don't alert on "speech" inside the house).

## Sensor Cueing

When a detection of interest occurs, the system should be able to cue other sensors — for example, directing a PTZ camera to slew to the estimated coordinates, or increasing the sample rate on nearby acoustic nodes. Initially this is a simple notification/webhook mechanism; more complex tasking can be added later.

Cueing targets include:
- **PTZ cameras** — slew to acoustic detection bearing/location
- **Acoustic nodes** — increase sample rate, activate beamforming toward target
- **Directional speaker** — directional audio projection for alerts or deterrence (future)
- **Recording promotion** — trigger extended raw audio retention for the detection area

## Self-Localization & Node Discovery

Nodes should be able to locate each other using light and sound patterns. A node emits a known acoustic pattern (chirp, pseudo-random sequence) from its speaker; other nodes detect and time this pattern to compute inter-node distances. Combined with GPS (where available) and IMU, this builds a self-consistent geometry of the array network without requiring manual measurement of every node position.

This is particularly important for:
- Rapid deployment (place nodes without surveying exact positions)
- Moving nodes that need continuous geometry updates
- Indoor environments where GPS is unavailable
- Verifying and correcting GPS-reported positions

## Health Monitoring

Each node should emit periodic **heartbeats**. A missing heartbeat should appear on the COP as a degraded sensor indicator. The system should define a **graceful degradation model**: at N nodes, full 3D localization is possible; as nodes drop, capability degrades to 2D, then classification-only, then alerting-only.

Nodes that lose connectivity should **buffer data locally** and sync when reconnected (store-and-forward).

## Effector Subsystem (Actions & Reactions)

MinimapPR is designed not just to observe, but to respond to detected and tracked events. The effector subsystem routes decisions from the rules engine to physical systems for deterrence, confirmation, or automation.

- **Threat Deterrence** — Automated response to specific predator or pest classifications. When a 'cat' or 'hawk' is confirmed in a 'chicken_coop' zone, the system can trigger a smart sprinkler, high-intensity strobe, or directional deterrent sound (audible predators or unpleasant frequencies).
- **Physical Cueing (Cross-Modality)** — Directing auxiliary sensors to points of interest. An acoustic detection of a 'fixed-wing aircraft' can cue a PTZ (Pan-Tilt-Zoom) camera to slew to the estimated coordinates for visual verification.
- **Smart Home Integration** — Gating automation based on spatial context. Turning on exterior floodlights only when an 'unknown human' track enters a specific property zone, or pausing a robotic lawnmower when 'speech' is localized in its path.
- **Dynamic Node Tasking** — Commanding sensor nodes to change behavior. Increasing the sample rate on specific microphones, activating beamforming, or swapping classification models when a target of interest is detected nearby.

**Safety Architecture for Effectors:**
All physical effectors share a common safety interlock model enforced at the rules engine level:
1. **Blue force lockout** — no effector fires if any friendly track is within a configurable danger radius of the target bearing
2. **Zone-based arming** — effectors are only armed within designated "effector-permitted" zones; residential, road, and neighbor zones are excluded bearings
3. **Fire weather interlock** — pyrotechnic effectors inhibited when environmental sensors report low humidity + high wind above configurable thresholds
4. **Rate limiting** — maximum rounds/launches per hour per effector, per zone, and system-wide
5. **Human-in-the-loop option** — configurable per effector class: fully automatic, confirm-to-fire (push notification with 10-second countdown), or manual-only
6. **Audit trail** — every effector activation is logged as an Alert with full provenance (detection → track → rule → effector command → result) for legal and safety review
7. **Master arm switch** — physical hardware safety on each effector platform, plus a software master arm in the COP UI
8. **Hardware Failsafe / Watchdog** — physical effectors must require a continuous heartbeat from the control system; loss of signal forces an immediate return to the safe (off) state.

From the software perspective, all of these are identical: the rules engine evaluates a condition, selects an effector, computes a bearing/coordinate, checks safety interlocks, and sends a command (GPIO relay, HTTP webhook, MAVLink waypoint, or MQTT message).

## Storage & Retention

Database design focuses on **self-cleaning** due to storage-intensive raw audio. Retention tiers:

| Tier | Data | Default Retention | Promotion |
|---|---|---|---|
| **Ephemeral** | Raw audio streams, raw accelerometer | Hours to days | Auto-promote on detection of interest |
| **Short** | Audio extracts tied to detections | ~1 month | Promote by label (e.g., gunshot → permanent) |
| **Long** | Detections, tracks, track updates, pings, environment, alerts | Permanent or configurable | — |
| **Config** | Nodes, labels, zones, known entities, annotations | Permanent | — |

**Automatic promotion**: rules can promote data from shorter to longer retention based on classification. For example, a gunshot detection auto-promotes its raw audio and all contributing observations to long-term storage. Promotion rules are defined alongside alert rules.

**Environmental data** (temperature, pressure, humidity) is explicitly long-term — it is tiny per-record and essential for ongoing speed-of-sound corrections, microclimate mapping, and trend analysis.

SQLite is appropriate for testing and light production. For spatial queries on zones and locations at scale, PostgreSQL with PostGIS is the natural upgrade path.

## Map & COP View

The primary output is a **2D map** showing tracks, detections, and sensor health. The underlying data is 3D, with altitude rendered as annotations on the 2D view. GDOP (Geometric Dilution of Precision) should be displayed or available as a layer to show localization quality across the coverage area.

Future extensions: acoustic coverage probability maps, sound intensity heatmaps, track history trails.

## Ambisonic Output & Soundscape Streaming

A secondary goal is the output of **ambisonic soundscapes** from the array. This is a downstream consumer of the localization pipeline — localized sounds are assigned to spatial channels based on their estimated positions, reconstructing the soundscape as multi-channel audio. Ambisonic output therefore requires successful localization first and shares the same environmental corrections and beamforming infrastructure.

Output formats:
- **First-order Ambisonics** (B-format) as PCM/WAV
- **IAMF / Eclipsa Audio** via [iamf-tools](https://github.com/AOMediaCodec/iamf-tools) for immersive streaming
- **5.1 surround** downmix for conventional playback
- Spatial audio processing via [SPARTA](https://leomccormack.github.io/sparta-site/) and [ambix](https://github.com/kronihias/ambix)

**Voice removal** enables privacy-safe public streaming (e.g., a park's bird sounds without birders' conversations). Vehicle noise removal is also supported for nature-focused streams.

The inverse — extracting and processing human speech via STT and LLM — supports use cases like incident detection at events or sentiment mapping in retail. Extracted voices can be normalized and routed to conference calls, with localization determining which speaker feed to use.

Audio distribution may use [Snapcast](https://github.com/badaix/snapcast) for synchronized multi-room playback, with [PipeWire](https://pipewire.org/) as the audio routing layer.

## Multi-Source Awareness

Beyond sound, the system is designed to incorporate:
- **Bluetooth / WiFi** device triangulation from equipped nodes (using [ESP32-Paxcounter](https://github.com/cyberman54/ESP32-Paxcounter) and [BLE-Scanner](https://github.com/gromeck/BLE-Scanner) patterns)
- **IMU / vibration** data from ground-mounted nodes — localization via vibration TDOA, threshold classifiers for seismic events
- **Temperature** for local microclimate mapping and speed-of-sound correction (critical for accurate acoustic localization)
- **Pressure** for atmosphere mapping, wind gradient estimation, airflow sensing
- **RTL-SDR** for wide-spectrum RF signal detection and localization (future)
- **ADS-B** for aircraft tracking ([rpi-led-flight-tracker](https://github.com/Weslex/rpi-led-flight-tracker) pattern) — correlate acoustic aircraft detections with ADS-B transponder data
- **Camera** integration for visual confirmation of acoustic detections
- **Echoes and reflections** for gradual 3D terrain mapping using ambient and active sound sources
- **Cross-cueing** between sensor types (e.g., BLE detects new device → cue acoustic to listen at that bearing; acoustic detects drone → cue camera to track)

## Integrations

- **Home Assistant** — primary destination for alerts; may serve as the frontend for some users
- **TAK ecosystem** (ATAK, WinTAK, iTAK) — via CoT interchange. Track and detection data fed as CoT events, appearing on TAK maps alongside other tactical data.
- **MAVLink** — for node control and integration with autonomous platforms (drones, rovers)
- **BirdNET** — specialized bird classification alongside or replacing YAMNet for avian detections
- **Rhasspy / Wyoming** — voice assistant integration, wake word detection, command processing

## Possible Integrations
- **Snapcast / PipeWire** — synchronized audio distribution and routing
- **OpenSoundscape** — acoustic localization and bioacoustic analysis ([opensoundscape.org](http://opensoundscape.org/))
- **ODAS** (Open embeddeD Audition System) — real-time sound source localization and tracking ([introlab/odas](https://github.com/introlab/odas))
- **pyroomacoustics** — room acoustics simulation and array processing ([LCAV/pyroomacoustics](https://github.com/LCAV/pyroomacoustics))
- **arduino-audio-tools** — audio processing on embedded platforms ([pschatzmann/arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools))
- **UI Forms** — optional voluntary telemetry collection from users for product improvement, perhaps also Change Data Orders

## Telemetry

Optional, voluntary telemetry via UI Forms or similar low-friction mechanism. Collects deployment configuration, node counts, use case categories, and anonymized performance metrics. No audio or location data. Telemetry is strictly opt-in and clearly documented.

## User Annotations (Future)

Users should be able to annotate tracks and detections (e.g., "this is the neighbor's dog, barks every morning at 7am") via the frontend or API. Annotations serve as:
- **Training labels** for improving local classification over time
- **Fingerprint seeds** — an annotation on a recurring sound triggers spectral fingerprint extraction for automatic future re-identification
- **Known entity registration** — annotating a track as friendly adds it to the Known Entities registry
- **Ground truth** for localization accuracy analysis

## Phasing & Prioritization

This document intentionally describes future extensions alongside the core system. The goal is to ensure the core architecture — data model, message formats, pipeline structure — is designed to accommodate these extensions without major incompatibilities or rewrites. However, only the core should be built first. Extensions should not be built until the core is solid.

### Phase 1 — Core (Build First)

The minimum viable system and the primary focus of initial development:

1. **Time-synchronized audio ingestion** — lossless PCM streams from point and array nodes with GPS PPS timestamps
2. **TDOA localization** — GCC-PHAT baseline with environmental corrections (temperature, humidity → speed of sound)
3. **Classification** — YAMNet, then BirdNET chaining for avian species identification
4. **Tracking** — Kalman filter with full track lifecycle (tentative → confirmed → coasting → dropped)
5. **Alerting** — simple rules engine with configurable conditions, delivering via COP WebSocket and structured logging (HA transport delivery in Phase 2)
6. **2D COP map** — tracks, detections, zones, sensor health, GDOP overlay, category-distinct track symbology
7. **Storage** — self-cleaning retention tiers, SQLite for initial deployments
8. **Node health** — heartbeats, graceful degradation model, store-and-forward buffering
9. **Geographic coordinate system** — site origin config, local-Cartesian ↔ lat/long conversion, enabling the COP map to render on real map tiles and all geographic output to use standard coordinates
10. **Interface contracts** — Protocol/ABC definitions for pluggable subsystems (localization, tracking, storage, preprocessing, transport, environment, taxonomy, rules actions) ensuring clean Phase 2+ extensibility

### Phase 2 — Near-term Extensions

Built on the core once it is validated. Architecture must not prevent these:

- **Federated fusion** — configure peer links between fusion servers for fault-tolerant, splittable/combinable networks
- **Ambisonic soundscape output** — downstream of localization, with voice removal for public streaming
- **BLE / WiFi device tracking** — IFF gating, blue force registry
- **Spectral fingerprinting** — known recurring sounds auto-identified without general classifier
- **User annotations** — training labels, fingerprint seeds, known entity registration
- **Additional localization algorithms** — SRP-PHAT, MUSIC, MVDR beamforming
- **TAK / CoT integration** — interoperability with military COP systems

### Phase 3+ — Future Extensions

Documented here for architectural awareness. These should not constrain Phase 1 design, but the data model and message formats should not be incompatible with them either:

- **Text extraction** — STT on speech detections for conversational memory, sentiment mapping, incident detection
- **Multi-source sensor fusion** — IMU/vibration, pressure/wind, RTL-SDR, ADS-B, camera
- **Passive acoustic terrain mapping** — echo analysis for 3D terrain models
- **Self-localization & node discovery** — acoustic chirp-based inter-node ranging
- **Seismology / infrasound** — ground vibration monitoring via IMU/accelerometer
- **Advanced tracking** — MHT, JPDA for complex multi-target scenarios
- **Microclimate mapping** — temperature/pressure/humidity trend mapping
- **Voice assistant integration** — wake word detection → STT → Home Assistant Voice
- **MAVLink node control** — autonomous platform integration
- **Effector subsystem** — physical-response effectors (deterrents, sprinklers, strobes, coordinated sequences) with C2-grade safety interlocks

Throughout this document, features marked **(future)** or described in the context of future use cases belong to Phase 2 or Phase 3+. The core data model (Nodes, Observations, Detections, Tracks, Labels, Zones, Alerts) is designed to support all phases. Tables like Text Extracts and Annotations are included in the schema design for awareness but should not be implemented until their respective phase.

## Testing Strategy

Systematic testing is critical for validating hardware and algorithm choices:

### Software Validation
- Frequency filtering: verify configurable highpass (default 50 Hz) removes out-of-band content
- Accelerometer/barometer subtraction from audio to remove vibration artifacts
- Time sync validation: known-pattern streaming across all nodes to verify alignment
- Localization accuracy: known source positions, measure angular and distance error

---

## Home Assistant Integration Contract

### Core Design Principle

**HA is a device I/O bus. MinimapPR is a perception and spatial reasoning engine.**

HA is the authoritative source for physical device state, actuator commands, and notification delivery. It has deep integrations with Zigbee, Z-Wave, Matter, and virtually every consumer smart home protocol. MinimapPR owns signal processing, spatial localization, track management, probabilistic rule evaluation, building spatial knowledge (IFC), and the COP. Neither system replicates the other's core capabilities.

The integration is bidirectional but asymmetric: MinimapPR publishes semantic state that HA consumes; HA provides sensor and presence data that MinimapPR consumes. Complex spatial reasoning and probabilistic conditions always happen inside MinimapPR before anything is published. HA automations stay simple.

### Division of Responsibility

| Concern | Owner | Rationale |
|---------|-------|-----------|
| Device I/O (lights, locks, HVAC, switches) | HA | HA has the protocol integrations |
| Physical device state persistence | HA | HA's entity registry is authoritative |
| Notification delivery (push, SMS, email) | HA (triggered by MinimapPR MQTT) | HA has the notification integrations |
| Simple device automations | HA | State machine style, HA's strength |
| Acoustic signal processing, localization | MinimapPR | HA cannot do DSP/TDOA |
| Track management and fusion | MinimapPR | Probabilistic, spatial — HA cannot |
| Room detection (which room is event in) | MinimapPR (from IFC) | HA has no building geometry |
| Equipment proximity queries | MinimapPR (from IFC) | Same reason |
| Complex multi-condition probabilistic rules | MinimapPR | Confidence thresholds, covariance — HA cannot |
| Semantic state publishing | MinimapPR → MQTT → HA | MinimapPR controls the contract |
| Environmental sensor data (temp, humidity) | HA (source) → MinimapPR (consumer) | HA already has these devices deployed |
| Person presence context | HA (source) → MinimapPR (consumer) | HA `person` entities are authoritative |
| COP visualization | MinimapPR frontend | HA Lovelace cannot do spatial COP |
| Building spatial knowledge (rooms, equipment) | IFC → MinimapPR | Single source of truth is the IFC model |
| LLM analysis and anomaly synthesis | Above both systems | Consumes from MinimapPR + HA |

### MQTT Semantic State Contract (MinimapPR → HA)

MinimapPR publishes **named semantic state** rather than raw detection payloads. HA treats these topics as standard sensors via MQTT Auto-Discovery. `ActionDescriptor.payload["topic"]` controls which topic a rule fires to — the `HassRuleActionHandler` is a thin transport; all reasoning happens in the rules engine before publish.

```
minimappr/rooms/{room_id}/occupancy
    Payload:  "occupied" | "vacant"
    HA type:  binary_sensor (device_class: occupancy)

minimappr/rooms/{room_id}/activity
    Payload:  {"label": str, "confidence": float, "timestamp_s": int}
    HA type:  event entity (event_type: minimappr_room_activity)

minimappr/equipment/{equip_id}/anomaly
    Payload:  {"type": str, "severity": "low"|"medium"|"high", "timestamp_s": int}
    HA type:  event entity (event_type: minimappr_equipment_anomaly)

minimappr/alerts/high
    Payload:  {"room": str, "label": str, "message": str, "track_id": str|null}
    HA type:  event entity (event_type: minimappr_alert)

minimappr/alerts/normal
    Payload:  {"room": str, "label": str, "message": str}
    HA type:  event entity (event_type: minimappr_alert)

minimappr/system/health
    Payload:  {"active_nodes": int, "degraded_nodes": int, "active_tracks": int}
    HA type:  sensor (diagnostic), periodic heartbeat

minimappr/device_tracker/{track_id}
    Payload:  {"room_id": str, "lat": float, "lon": float, "accuracy": float}
    HA type:  device_tracker (confirmed human-class tracks only)
```

### Data Enrichment Contract (HA → MinimapPR)

MinimapPR optionally reads from HA to enrich situational awareness. No HA modifications required — read-only via long-lived access token.

**Environmental enrichment**: Poll `GET /api/states/{entity_id}` for temperature/humidity from Zigbee/Z-Wave sensors and push to MinimapPR's `/api/v1/ingest/environment`. The `EnvironmentProvider` then uses real measured values rather than static defaults, improving TDOA speed-of-sound accuracy.

**Presence enrichment**: Subscribe to HA `person` entity state (home/away) and expose as a rule-evaluable context field (`ha_persons_home: bool`). Enables adaptive rule priority — glass break when `ha_persons_home: false` is critical; the same event with occupants home may be lower priority.

### Anti-Patterns to Avoid

- **Do not** push raw `DetectionEvent` payloads to HA expecting automations to parse position/confidence — HA cannot reason about this and automations become fragile
- **Do not** replicate MinimapPR's spatial rule logic in HA — single source of truth for spatial reasoning
- **Do not** try to make HA maintain track lifecycle state
- **Do not** make MinimapPR depend on HA being available for core operation — enrichment is optional, pipeline runs in degraded mode without it
- **Do not** put the LLM inside either MinimapPR or HA — it sits above both, consuming from MinimapPR's API and HA's state endpoint to synthesize context neither system has alone

### IFC ↔ MinimapPR Coordinate Alignment

The IFC model and MinimapPR share the same flat-XYZ meter convention:

| System | Origin | X | Y | Z |
|--------|--------|---|---|---|
| MinimapPR (`flat` mode) | `MINIMAPPR_SITE_ORIGIN_LAT/LON` | East | North | Up |
| catlin-house IFC | SW corner of foundation exterior | East | North | Up |

Alignment requires only a one-time origin offset in config:
```
MINIMAPPR_COORDINATE_MODE=flat
MINIMAPPR_IFC_MODEL_PATH=data/catlin_house.ifc
MINIMAPPR_IFC_ORIGIN_OFFSET_M=[0.0, 0.0, 0.0]
```
