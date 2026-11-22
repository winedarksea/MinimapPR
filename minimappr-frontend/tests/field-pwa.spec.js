// Field PWA browser checks (W4-pwa), desktop Chromium via Playwright:
//   - offline after first load (service worker shell + cached model),
//   - mic capture from Chrome's fake device fed by a fixture WAV,
//   - Stage-1 (minimappr-core in wasm) triggers on the bird,
//   - BirdNET V3 FP16 (ONNX Runtime Web) labels it,
//   - two tabs join one relay session by code and exchange sealed messages
//     through a local `wrangler dev` relay.
// Prerequisites: `npm ci` here and in ../field-relay-worker,
// `python scripts/fetch_birdnet_v3.py` at the repo root,
// `trunk build --release --config Trunk.field.toml`.
// Run: npx playwright test tests/field-pwa.spec.js --reporter=line

const { test, expect } = require("@playwright/test");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { buildFakeCaptureFixture } = require("./field/build-fake-capture-fixture.cjs");
const { startLocalFieldStack } = require("./field/local-field-stack.cjs");

const RESULTS_DIRECTORY = path.resolve(__dirname, "../test-results");
const FAKE_CAPTURE_PATH = buildFakeCaptureFixture(path.join(RESULTS_DIRECTORY, "field-fake-capture.wav"));
const MEASUREMENTS_PATH = path.join(RESULTS_DIRECTORY, "field-pwa-measurements.json");
const MODEL_LOAD_TIMEOUT_MILLISECONDS = 120_000;
const CLASSIFICATION_TIMEOUT_MILLISECONDS = 90_000;
const RELAY_STEP_TIMEOUT_MILLISECONDS = 30_000;
const EXPECTED_FIXTURE_SPECIES = "American Robin";
const KILOBYTES_PER_MEGABYTE = 1024;
// Cold `wrangler dev` runs worker-build first (minutes on a busy machine).
const STACK_STARTUP_TIMEOUT_MILLISECONDS = 600_000;

test.describe.configure({ mode: "serial" });
test.setTimeout(300_000);
test.use({
  permissions: ["microphone"],
  launchOptions: {
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${FAKE_CAPTURE_PATH}`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  },
});

let stack;
const measurements = {};

test.beforeAll(async () => {
  test.setTimeout(STACK_STARTUP_TIMEOUT_MILLISECONDS);
  stack = await startLocalFieldStack();
});

test.afterAll(async () => {
  if (stack) stack.stop();
  fs.mkdirSync(RESULTS_DIRECTORY, { recursive: true });
  fs.writeFileSync(MEASUREMENTS_PATH, JSON.stringify(measurements, null, 2));
  console.log(`field PWA measurements: ${JSON.stringify(measurements, null, 2)}`);
});

async function waitForModelLoaded(page) {
  await expect(page.getByTestId("birdnet-status")).toContainText("Loaded", { timeout: MODEL_LOAD_TIMEOUT_MILLISECONDS });
  return page.evaluate(() => window.mmpFieldDiagnostics.birdnetLoad);
}

// Resident memory of every renderer process (macOS/Linux `ps`).
async function rendererResidentMegabytes(browser) {
  const session = await browser.newBrowserCDPSession();
  const { processInfo } = await session.send("SystemInfo.getProcessInfo");
  await session.detach();
  return processInfo
    .filter((process) => process.type === "renderer")
    .map((process) => {
      try {
        const kilobytes = Number(execFileSync("ps", ["-o", "rss=", "-p", String(process.id)], { encoding: "utf8" }).trim());
        return Math.round(kilobytes / KILOBYTES_PER_MEGABYTE);
      } catch (_exited) {
        return null;
      }
    })
    .filter((megabytes) => megabytes !== null);
}

test("app shell and model load offline after one online load", async ({ page, context }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(`${stack.pageOrigin}/`);
  await expect(page.getByRole("heading", { name: "MinimapPR Field" })).toBeVisible();
  const firstLoad = await waitForModelLoaded(page);
  measurements.firstLoad = firstLoad;
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => true));
  const wasmBytes = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .filter((entry) => entry.name.endsWith("_bg.wasm"))
      .map((entry) => ({ name: entry.name.split("/").pop(), decodedBodySize: entry.decodedBodySize })),
  );
  measurements.appWasm = wasmBytes;

  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole("heading", { name: "MinimapPR Field" })).toBeVisible();
  const offlineLoad = await waitForModelLoaded(page);
  measurements.offlineLoad = offlineLoad;
  expect(offlineLoad.modelFromCache).toBe(true);
  expect(await page.evaluate(() => navigator.serviceWorker.controller !== null)).toBe(true);
  await context.setOffline(false);
  expect(pageErrors).toEqual([]);
});

test("fake mic -> Stage-1 trigger -> BirdNET labels the fixture bird", async ({ page, browser }) => {
  await page.goto(`${stack.pageOrigin}/`);
  await waitForModelLoaded(page);
  await page.getByTestId("mic-toggle").click();
  await expect(page.getByTestId("capture-status")).toContainText("Capturing at", { timeout: 15_000 });
  measurements.captureSettings = await page.evaluate(() => window.mmpFieldDiagnostics.captureSettings);

  await expect
    .poll(async () => Number(await page.getByTestId("stage1-trigger-count").textContent()), {
      timeout: CLASSIFICATION_TIMEOUT_MILLISECONDS,
    })
    .toBeGreaterThan(0);
  measurements.stage1 = await page.evaluate(() => window.mmpFieldDiagnostics.stage1);

  await expect(page.getByTestId("latest-label")).toContainText(EXPECTED_FIXTURE_SPECIES, {
    timeout: CLASSIFICATION_TIMEOUT_MILLISECONDS,
  });
  measurements.classification = await page.evaluate(() => window.mmpFieldDiagnostics.lastClassification);
  await expect(page.getByTestId("detection-feed")).toContainText(EXPECTED_FIXTURE_SPECIES.toLowerCase());
  measurements.captureGaps = await page.evaluate(() => window.mmpFieldDiagnostics.captureGaps || { count: 0 });
  measurements.jsHeapUsedMegabytes = await page.evaluate(() =>
    performance.memory ? Math.round(performance.memory.usedJSHeapSize / (1024 * 1024)) : null,
  );
  measurements.rendererResidentMegabytes = await rendererResidentMegabytes(browser);
});

test("two tabs join one relay session by code and exchange sealed messages", async ({ context }) => {
  const hostPage = await context.newPage();
  const joinerPage = await context.newPage();
  await hostPage.goto(`${stack.pageOrigin}/`);
  await joinerPage.goto(`${stack.pageOrigin}/`);

  await hostPage.getByTestId("host-session").click();
  await expect(hostPage.getByTestId("join-code")).toHaveText(/^[A-Z0-9]{4}-\d{6}$/, { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await expect(hostPage.getByTestId("relay-status")).toContainText("Connected as peer 1", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  const joinCode = await hostPage.getByTestId("join-code").textContent();

  await joinerPage.getByTestId("join-code-input").fill(joinCode);
  await joinerPage.getByTestId("join-session").click();
  await expect(joinerPage.getByTestId("session-key-state")).toHaveText("installed", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await expect(hostPage.getByTestId("relay-log")).toContainText("joined by code", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });

  await joinerPage.getByTestId("send-heartbeat").click();
  await expect(hostPage.getByTestId("relay-log")).toContainText(/sealed from \d+: Heartbeat/, {
    timeout: RELAY_STEP_TIMEOUT_MILLISECONDS,
  });
  await hostPage.getByTestId("send-heartbeat").click();
  await expect(joinerPage.getByTestId("relay-log")).toContainText("sealed from 1: Heartbeat", {
    timeout: RELAY_STEP_TIMEOUT_MILLISECONDS,
  });
  await expect(hostPage.getByTestId("connection-badge")).toHaveText("Relay online");

  // A wrong secret must fail the PAKE without the relay learning the code.
  const wrongCodePage = await context.newPage();
  await wrongCodePage.goto(`${stack.pageOrigin}/`);
  const wrongSecret = joinCode.replace(/\d{6}$/, (secret) => String((Number(secret) + 1) % 1_000_000).padStart(6, "0"));
  await wrongCodePage.getByTestId("join-code-input").fill(wrongSecret);
  await wrongCodePage.getByTestId("join-session").click();
  await expect(wrongCodePage.getByTestId("relay-log")).toContainText("join failed", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await expect(wrongCodePage.getByTestId("session-key-state")).toHaveText("none");
  measurements.relay = {
    hostSealedReceived: Number(await hostPage.getByTestId("sealed-received-count").textContent()),
    joinerSealedReceived: Number(await joinerPage.getByTestId("sealed-received-count").textContent()),
  };
});
