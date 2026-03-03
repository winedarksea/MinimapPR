# MinimapPR Agent Guidelines

This document provides guidelines for agents working on the MinimapPR codebase. Following these rules ensures consistency, maintainability, and architectural integrity.

## Architectural Principles

### 1.1 Small Files & Single Responsibility (SRP)
- **Aim to keep files < 500 lines.** Large files are difficult to reason about and cause merge conflicts.
- **Single Responsibility per File:** Each module should have one reason to change.
- **Decompose "God Classes":** Avoid monolithic classes. Extract responsibilities into focused components.

### 1.2 Interface-First Design (Protocols)
- **Use Protocols for Pluggability:** All major subsystems must implement a `Protocol` defined in [minimappr/interfaces.py](minimappr/interfaces.py)
- **`@runtime_checkable`:** All Protocols are decorated with `@runtime_checkable` to support `isinstance` checks at wiring time.

### 1.3 Module Boundaries & Dependency Injection
- **Explicit Boundaries:** Avoid deep coupling between modules.
- **Constructor Injection:** Pass dependencies (storage, trackers, etc.) into classes during initialization.
- **`app.state` is the DI root:** The `lifespan` context manager in `main.py` wires all subsystems onto `app.state` (e.g., `app.state.storage`, `app.state.classifier`, `app.state.tracker`). Dependency functions (`get_state`) extract these for route handlers.

### 1.4 Configuration-Driven Logic
- **No Magic Numbers:** Move operational constants (thresholds, multipliers, timings) into domain-specific sub-configs.
- **Sub-Configs over God-Settings:** Pass only relevant sub-configs to subsystems instead of the entire `Settings` object.

## 2. Coding Standards

### 2.1 Hyper-Descriptive Naming
- **Favor Explicit Over Concise:** Use long, descriptive names that explain intent

### 2.3 Explicit State & Error Handling
- **Explicit State Transitions:** State changes (e.g., `TrackState` / `TrackStatus` transitions) should be clear and traceable.
- **Robustness at the Edge:** Wrap potentially unstable operations (like `np.linalg.solve`) with try/except blocks to handle `LinAlgError` or singular matrices gracefully.

### 2.4 High-Signal Comments
- **Explain "Why", Not "What":** Comments should explain the reasoning behind complex algorithms or architectural decisions.
- **Be Token-Efficient in Comments:** Use concise, informative language. Focus on documenting interface contracts and capability tiers.

## 3. Data Integrity & Provenance
- **Maintain Traceability:** Every event must follow the chain: `observation` → `detection` → `track update` → `track` → `alert`.
- **Stable IDs:** Ensure `event_id`, `node_id`, and `track_id` are consistent across the pipeline.
- **Timestamp Accuracy:** Always include `TOA` (Time of Applicability), `TOR` (Time of Receipt), and `time_quality` (`gps_locked`, `ntp_sync`, `freerunning`).


## 4. Testing & Benchmarking
- **Shared Fixtures:** Place reusable fixtures in [tests/conftest.py](tests/conftest.py). Extract common test utilities (synthetic signal generators, stub storage) into a shared helper module if reused across 3+ files.
- **Avoid Test Duplication:** If the same setup appears in multiple test files, factor it into a conftest fixture or a helper module.
- **Refactor When Useful:** Code base is not deployed in production. Breaking changes are fine when they add clear value.
