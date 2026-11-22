// Field PWA checks in Playwright WebKit (the closest automatable stand-in
// for Safari; real Safari needs safaridriver + "Allow Remote Automation").
// WebKit has no file-backed fake microphone, so the bird-label check stays
// Chromium-only; here: offline shell + cached model, BirdNET inference on a
// synthetic window through the same interop, mic open with WebKit's mock
// capture device, the two-tab sealed relay exchange, and a reload with the
// page server stopped (real offline).
// Run: npx playwright test tests/field-pwa-webkit.spec.js --browser=webkit --reporter=line

const { test, expect, webkit } = require("@playwright/test");
const fs = require("node:fs");
const path = require("node:path");
const { startLocalFieldStack } = require("./field/local-field-stack.cjs");

const RESULTS_PATH = path.resolve(__dirname, "../test-results/field-pwa-webkit-measurements.json");
const MODEL_LOAD_TIMEOUT_MILLISECONDS = 180_000;
const RELAY_STEP_TIMEOUT_MILLISECONDS = 30_000;
const STACK_STARTUP_TIMEOUT_MILLISECONDS = 600_000;
const BIRDNET_INPUT_SAMPLE_RATE_HZ = 32000;
const PROBE_WINDOW_SECONDS = 3;
const MICROPHONE_SETTLE_MILLISECONDS = 10_000;

test.describe.configure({ mode: "serial" });
test.setTimeout(300_000);

let stack;
let browser;
const measurements = {};

test.beforeAll(async () => {
  test.setTimeout(STACK_STARTUP_TIMEOUT_MILLISECONDS);
  stack = await startLocalFieldStack();
  browser = await webkit.launch();
});

test.afterAll(async () => {
  if (browser) await browser.close();
  if (stack) stack.stop();
  fs.mkdirSync(path.dirname(RESULTS_PATH), { recursive: true });
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(measurements, null, 2));
  console.log(`field PWA WebKit measurements: ${JSON.stringify(measurements, null, 2)}`);
});

async function waitForModelLoaded(page) {
  await expect(page.getByTestId("birdnet-status")).toContainText("Loaded", { timeout: MODEL_LOAD_TIMEOUT_MILLISECONDS });
  return page.evaluate(() => window.mmpFieldDiagnostics.birdnetLoad);
}

test("WebKit: model loads, BirdNET runs, mic attempt recorded", async () => {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(`${stack.pageOrigin}/`);
  measurements.firstLoad = await waitForModelLoaded(page);
  measurements.inference = await page.evaluate(
    async ({ sampleRate, seconds }) => {
      const samples = new Float32Array(sampleRate * seconds);
      for (let index = 0; index < samples.length; index += 1) samples[index] = 0.01 * Math.sin(index * 0.7);
      const first = await window.mmpFieldBirdnet.classify(samples, 1);
      const second = await window.mmpFieldBirdnet.classify(samples, 1);
      return { firstMilliseconds: first.inferenceMilliseconds, secondMilliseconds: second.inferenceMilliseconds };
    },
    { sampleRate: BIRDNET_INPUT_SAMPLE_RATE_HZ, seconds: PROBE_WINDOW_SECONDS },
  );
  // Recorded, not asserted: headless WebKit may leave getUserMedia pending
  // (no permission UI to answer), which says nothing about real Safari.
  try {
    await context.grantPermissions(["microphone"]);
  } catch (error) {
    measurements.microphonePermissionGrant = `unsupported: ${error.message.split("\n")[0]}`;
  }
  await page.getByTestId("mic-toggle").click();
  await page.waitForTimeout(MICROPHONE_SETTLE_MILLISECONDS);
  measurements.micStatus = await page.getByTestId("capture-status").textContent();
  await context.close();
});

test("WebKit: two tabs join one relay session and exchange sealed messages", async () => {
  const context = await browser.newContext();
  const hostPage = await context.newPage();
  const joinerPage = await context.newPage();
  await hostPage.goto(`${stack.pageOrigin}/`);
  await joinerPage.goto(`${stack.pageOrigin}/`);
  await hostPage.getByTestId("host-session").click();
  await expect(hostPage.getByTestId("join-code")).toHaveText(/^[A-Z0-9]{4}-\d{6}$/, { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await expect(hostPage.getByTestId("relay-status")).toContainText("Connected as peer 1", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await joinerPage.getByTestId("join-code-input").fill(await hostPage.getByTestId("join-code").textContent());
  await joinerPage.getByTestId("join-session").click();
  await expect(joinerPage.getByTestId("session-key-state")).toHaveText("installed", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await joinerPage.getByTestId("send-heartbeat").click();
  await expect(hostPage.getByTestId("relay-log")).toContainText(/sealed from \d+: Heartbeat/, { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  await hostPage.getByTestId("send-heartbeat").click();
  await expect(joinerPage.getByTestId("relay-log")).toContainText("sealed from 1: Heartbeat", { timeout: RELAY_STEP_TIMEOUT_MILLISECONDS });
  measurements.relayExchange = "ok";
  await context.close();
});

// Last: it stops the page server. `context.setOffline` + reload is an
// internal error in Playwright WebKit when a service worker is involved.
test("WebKit: shell + model load with the page server gone", async () => {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(`${stack.pageOrigin}/`);
  await waitForModelLoaded(page);
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => true));
  stack.stopPageServer();
  await page.reload();
  measurements.offlineLoad = await waitForModelLoaded(page);
  expect(measurements.offlineLoad.modelFromCache).toBe(true);
  await context.close();
});
