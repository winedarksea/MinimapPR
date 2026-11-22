// Starts the local stack the Field PWA browser checks run against:
//   1. the relay Worker under `wrangler dev --local` (miniflare/workerd), with
//      local D1 migrations applied into a throwaway state directory;
//   2. field/field-dev-server.mjs serving dist-field/, the gitignored model
//      directory and a same-origin /v1 proxy to that relay.
// Strictly local, same guarantees as minimappr-transport's wrangler harness:
// no login, no deploy, telemetry off, CLOUDFLARE_* removed.

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");

const FRONTEND_DIRECTORY = path.resolve(__dirname, "../..");
const WORKER_DIRECTORY = path.resolve(FRONTEND_DIRECTORY, "../field-relay-worker");
const WRANGLER_BINARY = path.join(WORKER_DIRECTORY, "node_modules/.bin/wrangler");
const LOOPBACK_HOST = "127.0.0.1";
// First `wrangler dev` runs `worker-build --release` (30-90 s cold).
const WORKER_STARTUP_TIMEOUT_MILLISECONDS = 540_000;
const SERVER_STARTUP_TIMEOUT_MILLISECONDS = 15_000;
const HEALTH_POLL_INTERVAL_MILLISECONDS = 500;

function freeLocalPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, LOOPBACK_HOST, () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function localOnlyEnvironment() {
  const environment = { ...process.env, WRANGLER_SEND_METRICS: "false", CI: "1" };
  delete environment.CLOUDFLARE_API_TOKEN;
  delete environment.CLOUDFLARE_ACCOUNT_ID;
  return environment;
}

async function waitForHttpOk(url, timeoutMilliseconds, child, logPath) {
  const started = Date.now();
  while (Date.now() - started < timeoutMilliseconds) {
    if (child.exitCode !== null) {
      throw new Error(`process exited early (${child.exitCode}):\n${fs.readFileSync(logPath, "utf8")}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (_notYetListening) {
      // keep polling
    }
    await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_INTERVAL_MILLISECONDS));
  }
  throw new Error(`${url} not healthy after ${timeoutMilliseconds} ms:\n${fs.readFileSync(logPath, "utf8")}`);
}

function killProcessGroup(child) {
  if (!child || child.exitCode !== null) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch (_alreadyGone) {
    child.kill("SIGTERM");
  }
}

async function startLocalRelayWorker(stateDirectory) {
  if (!fs.existsSync(WRANGLER_BINARY)) {
    throw new Error(`pinned wrangler missing: run \`npm ci\` in ${WORKER_DIRECTORY}`);
  }
  const migrated = spawnSync(
    WRANGLER_BINARY,
    ["d1", "migrations", "apply", "JOIN_CODES_DB", "--local", "--persist-to", stateDirectory],
    { cwd: WORKER_DIRECTORY, env: localOnlyEnvironment(), encoding: "utf8" },
  );
  if (migrated.status !== 0) throw new Error(`local D1 migrations failed:\n${migrated.stdout}${migrated.stderr}`);
  const port = await freeLocalPort();
  const inspectorPort = await freeLocalPort();
  const logPath = path.join(stateDirectory, "wrangler-dev.log");
  const log = fs.openSync(logPath, "w");
  const child = spawn(
    WRANGLER_BINARY,
    [
      "dev", "--local", "--ip", LOOPBACK_HOST, "--port", String(port),
      "--inspector-port", String(inspectorPort), "--persist-to", stateDirectory,
      "--show-interactive-dev-session=false",
    ],
    { cwd: WORKER_DIRECTORY, env: localOnlyEnvironment(), stdio: ["ignore", log, log], detached: true },
  );
  // A runner that aborts this hook must not leave workerd behind.
  process.once("exit", () => killProcessGroup(child));
  const origin = `http://${LOOPBACK_HOST}:${port}`;
  try {
    await waitForHttpOk(`${origin}/v1/health`, WORKER_STARTUP_TIMEOUT_MILLISECONDS, child, logPath);
  } catch (error) {
    killProcessGroup(child);
    throw error;
  }
  return { child, origin, logPath };
}

async function startFieldDevServer(relayOrigin, stateDirectory) {
  const port = await freeLocalPort();
  const logPath = path.join(stateDirectory, "field-dev-server.log");
  const log = fs.openSync(logPath, "w");
  const child = spawn(process.execPath, [path.join(FRONTEND_DIRECTORY, "field/field-dev-server.mjs"), "--port", String(port)], {
    cwd: FRONTEND_DIRECTORY,
    env: { ...process.env, FIELD_RELAY_ORIGIN: relayOrigin },
    stdio: ["ignore", log, log],
    detached: true,
  });
  process.once("exit", () => killProcessGroup(child));
  const origin = `http://${LOOPBACK_HOST}:${port}`;
  await waitForHttpOk(`${origin}/index.html`, SERVER_STARTUP_TIMEOUT_MILLISECONDS, child, logPath);
  return { child, origin, logPath };
}

async function startLocalFieldStack() {
  const stateDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mmp-field-pwa-"));
  const relay = await startLocalRelayWorker(stateDirectory);
  let server;
  try {
    server = await startFieldDevServer(relay.origin, stateDirectory);
  } catch (error) {
    killProcessGroup(relay.child);
    throw error;
  }
  return {
    pageOrigin: server.origin,
    relayOrigin: relay.origin,
    stop() {
      killProcessGroup(server.child);
      killProcessGroup(relay.child);
      fs.rmSync(stateDirectory, { recursive: true, force: true });
    },
  };
}

module.exports = { startLocalFieldStack };
