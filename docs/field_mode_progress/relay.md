# W3-relay: notes

Branch: `fm/relay`, off `field-mode` at `8eca3b4` (Merge fm/core-sync).

## What landed

| Path | Contents |
|---|---|
| `minimappr-relay-wire/` (new root member) | The relay wire contract shared by the Worker and the client: 10-byte relay frame header, `decide_relay_delivery` routing policy, `SessionUsageCounters` caps, JSON control notices, HTTP API bodies, join-code lookup alphabet. It depends only on serde, so the policy is unit-tested natively. |
| `minimappr-transport/` (new root member) | A sans-IO relay client (`RelayClientSession`), the outbound priority scheduler, clip chunking and reassembly, join codes, the SPAKE2 exchange over relay flights, and a native adapter behind the `native-websocket` feature (reqwest plus tokio-tungstenite). |
| `field-relay-worker/` (own `[workspace]`, excluded from the root) | A Rust Cloudflare Worker (workers-rs 0.7.5): HTTP API, D1 join codes, a cron cleanup, and the `SessionRelayObject` Durable Object (WebSocket hibernation). Local migrations live in `migrations/`. `package.json` pins `wrangler` 4.158.0 (released 2026-11-05) as a devDependency. |

The root `Cargo.toml` changed in only two places: two new members, and `field-relay-worker` added to `exclude`.

## Join codes (lookup/secret split)

A spoken code looks like `K7QX-731904`.
- The **lookup** part (`K7QX`) has 4 symbols from a 32-symbol alphabet with 0/O and 1/I removed, giving about 1M live codes. The Worker mints it and stores it in D1. It is the only part the Worker ever sees.
- The **secret** part (`731904`) is 6 digits minted on the host by `FieldJoinCode::with_generated_secret`, using unbiased rejection sampling. It is the SPAKE2 password and never appears in any relay request. `RelayHttpClient::resolve_join_code` takes only the lookup code. This meets the core_codec_proto contract: a Worker that saw the password could run the PAKE itself.
- `POST /v1/join` turns a lookup code into a **fresh per-joiner role token** minted by the session's Durable Object, plus the public Field session id needed for the SPAKE2 identities. Each code has a resolve budget (`JOIN_CODE_MAXIMUM_RESOLVES`, default 32). The status codes are:
  - 404: unknown code, or already cleaned up
  - 410: expired but not yet swept
  - 429: resolve budget spent
  - 403: a Host role was requested
- TTL: a code expires after `JOIN_CODE_TTL_SECONDS` (default 900, capped at session expiry). It is refused at read time once expired. The cron (`*/15 * * * *`) deletes expired codes, codes of expired sessions, and expired sessions.
- The host can also call `POST /v1/sessions/{name}/join-codes` to issue a new code and `DELETE /v1/join-codes/{code}` to revoke one, both with a Bearer host token. The intended use is after `HostRelayJoinCodeResponder::is_exhausted()`, i.e. after 5 wrong guesses.

## Relay (Durable Object per session)

- **Connect:** `GET /v1/relay/{name}/connect?peer=&token=[&resume=]`. The token is in the query string because browsers cannot set headers on a WebSocket.
- **Hibernation:** sockets are accepted with `accept_websocket_with_tags` (tags `peer:<id>` and `role:<role>`). Peer id and role live in the socket attachment. The only timer is one alarm at session expiry, which closes all sockets and runs `storage.delete_all()`.
- **Role tokens:** session creation returns session-wide host, node and viewer tokens, for the QR code and view links. Join codes mint per-joiner node or viewer tokens. Only SHA-256 digests are stored, in D1 (host token) and Durable Object storage (all tokens).
- **Routing policy** (`decide_relay_delivery`, pure and unit-tested):
  - The header sender must equal the socket's bound peer id.
  - Sealed frames go to one recipient or are broadcast to everyone else.
  - Viewers may not send sealed frames.
  - Unsealed `JoinCodeKeyExchange` frames from nodes or viewers may only go to the connected host. The host's replies go to exactly one peer and are never broadcast.
- **Peer id binding and resume:**
  - The first connect to a peer id stores a resume-token digest. A later claim of that peer id must present the matching resume token and the same role, or it gets 409. A test checks this: a thief holding a valid node token cannot take over the peer id after the real peer drops.
  - The resume token is rotated on every accepted connect, and a resumed connection closes the stale socket with code 4000.
- **One host at a time:** a second host-role socket on a different peer id gets 409.
- **Caps** (`SessionUsageCounters::try_admit_frame`, checked before parsing because every inbound message is billed): per-session inbound messages, inbound bytes, frame size, and connection count, all from wrangler `[vars]`. A rejected frame is dropped, and the sender gets a `frame_rejected` text notice. Counters are persisted every `USAGE_COUNTER_PERSIST_EVERY_MESSAGES` admitted frames and on every close, so an eviction can lose at most N-1 counts.
- **Control notices:** `welcome`, `peer_joined`, `peer_left` and `frame_rejected` are sent as text frames, so they can never be confused with binary relay frames.

## Transport client

- **Sans-IO:** time (`now_milliseconds`) and the CSPRNG are passed in, and there is no tokio. The crate builds for `wasm32-unknown-unknown` with `--no-default-features`. The adapter loop is `poll_transmit` → write → `on_binary_frame`/`on_text_frame` → `poll_event`.
- **Sealing:**
  - Frames are sealed at dequeue time, so sender sequence numbers follow the actual send order.
  - The receiver checks AEAD, envelope sender against header sender, recipient (self or broadcast), and then the replay window. Drops are counted in `RelayClientInboundStatistics`, never surfaced one by one.
  - A non-key-exchange message arriving in the unsealed lane is refused.
- **Priority:**
  - `OutboundScheduler` always sends control frames first. Clip chunks go earliest-deadline-first.
  - The adapter writes one frame and only then asks for the next, so a control frame waits behind at most one chunk.
  - Default chunk size is 4 KiB, so a 40 KB clip is about 10 messages.
  - The control queue and the clip byte budget are bounded and return errors (backpressure).
- **Late-clip abandonment:**
  - The sender drops a clip whole once `abandon_at` passes, mid-transfer included. The deadline comes from `ClipRequest::abandon_after_milliseconds` or the config default of 5 s.
  - The receiver drops partial clips after 8 s. It also caps concurrent partial clips (evicting the oldest) and total reassembled size.
- **Reconnect with resume:**
  - The client uses exponential backoff (250 ms → 15 s).
  - The welcome carries the resume token.
  - On a resumed welcome, every clip still inside its deadline restarts from chunk 0. Fully written clips are kept until their deadline for this purpose, because the dead socket may have swallowed their tail chunks. The receiver remembers completed (sender, request id) pairs, so a clip is never delivered twice.
  - Control frames are at-most-once per connection. The relay does not buffer for absent peers, and durable state belongs to the host event log.
- **Join over relay:**
  - `JoinerRelayKeyExchange` and `HostRelayJoinCodeResponder` wrap the core SPAKE2 states and key concurrent attempts by (relay peer id, attempt id), with a bounded pending table.
  - The host always answers with a flight. A failure returns `Rejected` and `failure`, and `JoinCodeAttemptsExhausted` signals that the code should be re-issued.
  - Code joiners bind a provisional peer id ≥ `0x8000_0000`, since host-assigned node ids stay below that, and reconnect with their assigned id after the sealed `JoinAccepted`.

## Security: what the relay can and cannot see

**It can see:**
- Relay session names, peer ids, roles, and frame sizes and timing.
- The cleartext envelope routing prefix (sender, recipient, sequence).
- The public Field session id.
- Join-code lookup halves.
- SPAKE2 flight bytes. These are uniformly random-looking, the confirmation tag is an HMAC, and flight 4 is AEAD-sealed under the PAKE key.

**It cannot see:**
- The session key, since it lives in the QR code or arrives via PAKE.
- The join-code secret.
- Any message body, audio or clip, or message kind. Kinds are inside the AEAD.
- Who joined by code versus by QR, beyond token provenance.

**What a malicious relay can still do:**
- Drop, delay, reorder or replay frames. Receivers drop replays via the window, but the relay can still deny service.
- Make **one online PAKE guess per attempt**, bounded by the host's 5-failure retirement.
- Mint itself tokens. It is the token authority, but tokens only grant relay access, never decryption.

Rewriting a header sender or recipient fails the envelope checks. A test covers this.

## Commands and results

- `cargo test --workspace` passes. New suites:
  - `minimappr-relay-wire`: 11 unit tests
  - `minimappr-transport`:
    - 2 unit tests
    - `clip_chunking_and_scheduling`: 7
    - `relay_client_session_loopback`: 8. These go through an in-memory relay that applies the same policy and caps code as the Durable Object.
    - `relay_worker_integration`: 1 scenario test, about 11 s warm
- `cargo build --target wasm32-unknown-unknown` (in `field-relay-worker/`) and `worker-build --release` both succeed. The bundle is a 402 KB wasm plus a 31 KB JS shim.
- `cargo build --target wasm32-unknown-unknown -p minimappr-transport --no-default-features` succeeds.
- **Integration test:** `relay_worker_integration` does the following:
  - Applies the D1 migrations locally (`wrangler d1 migrations apply --local --persist-to target/tmp/...`).
  - Starts `wrangler dev --local --test-scheduled` on a free port with `--var` overrides: TTL 4 s, 120 messages, 256 KiB, 32 KiB frames, persist every message.
  - Runs two native clients through the full scenario: join by code, SPAKE2 over the relay, sealed unicast and broadcast, a chunked clip overtaken by a later control message, abrupt drop with a thief 409 and then resume, role and token checks, TTL (410, then 404 after the cron), code reissue and revoke, the message cap, the frame cap and the byte cap.
  - Skips loudly when `npm`, `worker-build` or the pinned wrangler install is unavailable. `MINIMAPPR_REQUIRE_RELAY_INTEGRATION=1` turns a skip into a failure.
  - Kills the wrangler/workerd process group on drop.
- No deploys and no account calls were made: no login, no remote D1, telemetry was off (`WRANGLER_SEND_METRICS=false`), and the `CLOUDFLARE_*` environment variables were removed.

## Deviations

- **Extra crate `minimappr-relay-wire`.** The Worker needs the same framing, policy and caps as the client. Depending on `minimappr-core` would pull rustfft/nalgebra into the Worker bundle, and duplicating the logic would let the two drift. The tiny crate also lets the policy run in `cargo test --workspace`.
- **Relay frame header separate from the sealed envelope.** Key-exchange flights travel unsealed and have no envelope, so the relay routes on its own 10-byte header. Receivers cross-check it against the authenticated envelope.
- **worker 0.7.5 rather than the latest 0.9.** It matches the installed `worker-build` 0.7.5 and the pcplayerpicker pattern. worker-macros 0.7.5 forgets to enable syn's `full` feature, so `field-relay-worker/Cargo.toml` adds `syn` with `full` as a build-dependency to fix the feature unification.

## Open issues

- **Head-of-line blocking in kernel buffers.** The scheduler orders frames correctly, but on a slow uplink the OS socket buffer can still hold several chunks. Native shells could shrink `SO_SNDBUF`; the browser adapter should gate on `WebSocket.bufferedAmount`.
- **No web-sys WebSocket adapter yet.** That belongs to W4-pwa, which drives the same `RelayClientSession`.
- **Join-code endpoints are not rate-limited per IP.** Only the per-code resolve budget and the host's 5-guess retirement apply. Consider Cloudflare rate-limiting rules at deploy time.
- **No host handover on the relay.** The "one host socket" rule means a new host needs a host token. Handover (W4/Phase 4) needs either a host-token transfer inside the sealed session or a DO endpoint to promote a peer.
- **Clip keys can collide.** `ClipPayload` and `TapToHearPayload` from the same node share the `(sender, request_id)` reassembly key, so callers must keep request ids distinct across both.
- **Sequence numbers do not survive a process restart.** A shell restarting mid-session must persist `next_sender_sequence` or seed it, for example from time; otherwise receivers' replay windows reject it.
- **Relay usage is not exposed beyond `welcome`.** The Phase 3 messages/bytes measurement could add a host-only usage notice.
- **No deploy path.** `wrangler.toml` has a placeholder D1 id and no routes or assets by design. Before any deploy, `wrangler d1 create` and a route must be added.
