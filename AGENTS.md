# MinimapPR Agent Guidelines

This document provides guidelines for agents working on the MinimapPR codebase. Following these rules ensures consistency, maintainability, and architectural integrity.

---

## 1. Project Layout

```
minimappr/            # Python package root
  main.py             # FastAPI app, lifespan, error handlers
  config.py           # Settings + domain sub-configs (dataclasses)
  interfaces.py       # Protocol definitions for all subsystems
  models.py           # Pydantic models (API schemas, domain objects)
  api/                # HTTP + WebSocket routes
  classifiers/        # Audio classification (heuristic, YAMNet, chaining)
  core/               # Localization, tracking, beamforming, zones, rules, etc.
  frontend/           # Static HTML/JS/CSS served by FastAPI
  sim/                # Simulation, benchmarking, soak tests
  storage/            # SQLite-backed async storage
  utils/              # Shared helpers (audio utilities)
firmware/             # Embedded C/C++ for sensor nodes (ESP32, RP2350)
tests/                # pytest test suite (pytest-asyncio, asyncio_mode=auto)
scripts/              # One-off operational scripts
```

## 2. Architectural Principles

### 2.1 Small Files & Single Responsibility (SRP)
- **Aim to keep files < 500 lines.** Large files are difficult to reason about and cause merge conflicts.
- **Single Responsibility per File:** Each module should have one reason to change.
- **Decompose "God Classes":** Avoid monolithic classes. Extract responsibilities into focused components (e.g., `IngestProcessor`, `ClassificationOrchestrator`, `DetectionAssembler`).

### 2.2 Interface-First Design (Protocols)
- **Use Protocols for Pluggability:** All major subsystems must implement a `Protocol` defined in [minimappr/interfaces.py](minimappr/interfaces.py).
- **Current Interfaces:** `Localizer`, `TrackAssociator`, `TrackFilter`, `StorageBackend`, `EnvironmentProvider`, `IngestTransport`, `AudioPreprocessor`, `Beamformer`, `TaxonomyProvider`, `RuleActionHandler`, `RuleEngine`.
- **Composition over Inheritance:** Use the strategy pattern. Inject implementations via constructors rather than inheriting from base classes.
- **`@runtime_checkable`:** All Protocols are decorated with `@runtime_checkable` to support `isinstance` checks at wiring time.

### 2.3 Module Boundaries & Dependency Injection
- **Explicit Boundaries:** Avoid deep coupling between modules.
- **No Module-Level Singletons:** Do not create global instances at the top level of a file. Use FastAPI's `app.state` to store singletons and pass them into constructors.
- **Constructor Injection:** Pass dependencies (storage, trackers, etc.) into classes during initialization.
- **`app.state` is the DI root:** The `lifespan` context manager in `main.py` wires all subsystems onto `app.state` (e.g., `app.state.storage`, `app.state.classifier`, `app.state.tracker`). Dependency functions (`get_state`) extract these for route handlers.

### 2.4 Configuration-Driven Logic
- **No Magic Numbers:** Move operational constants (thresholds, multipliers, timings) into domain-specific sub-configs.
- **Sub-Configs over God-Settings:** Pass only relevant sub-configs to subsystems instead of the entire `Settings` object. Current sub-configs in [minimappr/config.py](minimappr/config.py): `LocalizationConfig`, `TrackingConfig`, `ClassifierConfig`, `StorageConfig`, `FusionConfig`, `RulesConfig`, `FederationConfig`, `FederationPeerConfig`.

## 3. Coding Standards

### 3.1 Hyper-Descriptive Naming
- **Favor Explicit Over Concise:** Use long, descriptive names that explain intent

### 3.3 Explicit State & Error Handling
- **Explicit State Transitions:** State changes (e.g., `TrackState` / `TrackStatus` transitions) should be clear and traceable.
- **Robustness at the Edge:** Wrap potentially unstable operations (like `np.linalg.solve`) with try/except blocks to handle `LinAlgError` or singular matrices gracefully.

### 3.4 High-Signal Comments
- **Explain "Why", Not "What":** Comments should explain the reasoning behind complex algorithms or architectural decisions.
- **Be Token-Efficient in Comments:** Use concise, informative language. Focus on documenting interface contracts and capability tiers.

## 4. Data Integrity & Provenance
- **Maintain Traceability:** Every event must follow the chain: `observation` → `detection` → `track update` → `track` → `alert`.
- **Stable IDs:** Ensure `event_id`, `node_id`, and `track_id` are consistent across the pipeline.
- **Timestamp Accuracy:** Always include `TOA` (Time of Applicability), `TOR` (Time of Receipt), and `time_quality` (`gps_locked`, `ntp_sync`, `freerunning`).


## 5. Testing & Benchmarking
- **Shared Fixtures:** Place reusable fixtures in [tests/conftest.py](tests/conftest.py). Extract common test utilities (synthetic signal generators, stub storage) into a shared helper module if reused across 3+ files.
- **Avoid Test Duplication:** If the same setup appears in multiple test files, factor it into a conftest fixture or a helper module.
