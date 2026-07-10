# MinimapPR Agent Guidelines

This document provides guidelines for agents working on the MinimapPR codebase.

## Architectural Principles

### 1.1 Small Files & Single Responsibility (SRP)
- **Aim to keep files < 500 lines.** Large files are difficult to reason about and cause merge conflicts.
- **Single Responsibility per File:** Each module should have one reason to change.
- **Decompose "God Classes":** Avoid monolithic classes. Extract responsibilities into focused components.

### 1.2 Module Boundaries & Dependency Injection
- **Explicit Boundaries:** Avoid deep coupling between modules.

### 1.3 Configuration-Driven Logic
- **No Magic Numbers:** Move operational constants (thresholds, multipliers, timings) into domain-specific sub-configs.

## 2. Coding Standards

### 2.1 Hyper-Descriptive Naming
- **Favor Explicit Over Concise:** Use long, descriptive names that explain intent

### 2.3 Error Handling
- **Robustness at the Edge:** Wrap potentially unstable operations (like `np.linalg.solve`) with try/except blocks to handle `LinAlgError` or singular matrices gracefully.
- **Avoid Pointless Fallbacks** Only include fallbacks where the fallback is effective. Heuristic fallbacks can hide errors and slow debugging, so avoid them.

### 2.4 High-Signal Comments
- **Explain "Why", Not "What":** Comments should explain the reasoning behind complex algorithms or architectural decisions.
- **Be Token-Efficient in Comments:** Use concise, informative language. Focus on documenting interface contracts and capability tiers.

## 3. Testing & Benchmarking
- **Shared Fixtures:** Place reusable fixtures into a shared helper module if reused across 3+ files.
- **Avoid Test Duplication:** If the same setup appears in multiple test files, factor it into a fixture or a helper module.
- **Refactor When Useful:** Code base is not deployed in production. Breaking changes are fine when they add clear value.

# Core Release Requirements
* Make sure there are no gaps (lost packets) in the detection audio.
* Make sure realistic localization is occurring. Our goal is to be able to localize from 1 meter out to a kilometer or more (when enough nodes are active)
* Make sure nothing is blocking the audio pipeline (ie UI requests don't hold up audio processing requests)
* Make sure detections have lat/long of their track (or of their node, if omnidirectional)
* Precise TDOA is important. This means using carefully precise GPS (or NTP if no GPS) timestamps on firmware's outgoing audio packets which the server then uses (using the timestamp on packet, not server time) to process localizations.
* Streaming audio should not be landing on disk until it is localized and classified. Most audio should be discarded (exceptions: some detections, IAMF recordings) without ever being written to disk on the server. This also should have minimal other tracking bloat (indexes, etc) being landed to disk, and generally what is landing should be landing in the SQLite, which should have proper cleanup policies automatically cleaning up old data.
* IAMF audio and ambisonic audio able to be recorded and export, with the localized and tracked audio passed in as proper objects to the IAMF (Atmos style) audio format. The goal is also for this to have a basic video start to a server connected webcam such that high quality video is recorded perfectly in sync with the audio, ready for upload to YouTube.
