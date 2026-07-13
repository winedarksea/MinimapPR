const { test, expect } = require("@playwright/test");

const baseUrl = process.env.MINIMAPPR_FRONTEND_URL || "http://127.0.0.1:18080";

test.use({
  launchOptions: {
    args: ["--use-gl=swiftshader"],
  },
});

test("COP workspace renders full-bleed map chrome", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(`${baseUrl}/cop`, { waitUntil: "domcontentloaded" });

  await expect(page.locator(".cop-workspace")).toBeVisible();
  await expect(page.locator("#mmp-map")).toBeVisible();
  await expect(page.locator("#mmp-map canvas").first()).toBeVisible();
  await expect(page.locator(".workspace-dock-left")).toBeVisible();
  await expect(page.locator(".workspace-dock-right")).toBeVisible();
  await expect(page.locator(".workspace-status-ribbon")).toBeVisible();
  await expect(page.getByRole("button", { name: "Map layers" })).toBeVisible();
  await page.getByRole("button", { name: "Map layers" }).click();
  await expect(page.locator(".workspace-map-controls")).toBeVisible();
  await expect(page.locator(".system-strip")).toHaveCount(0);

  const expectedLayerIds = [
    "nodes",
    "tracks",
    "detections",
    "omni",
    "zones",
    "overlays",
    "acoustic",
  ];
  for (const layerId of expectedLayerIds) {
    await expect(page.locator(`[data-layer-id="${layerId}"]`)).toBeVisible();
  }

  for (let routeCycle = 0; routeCycle < 2; routeCycle += 1) {
    await page.getByRole("link", { name: "Analysis" }).click();
    await expect(page).toHaveURL(/\/analysis$/);
    await expect(page.locator(".subnav")).toBeVisible();
    await expect(page.locator("#mmp-map-parking-lot > [data-mmp-original-map-id]")).toHaveCount(1);

    await page.getByRole("link", { name: "COP" }).click();
    await expect(page).toHaveURL(/\/cop$/);
    await expect(page.locator("#mmp-map")).toBeVisible();
    await expect(page.locator("#mmp-map canvas")).toHaveCount(1);
    await expect(page.locator("#mmp-map canvas").first()).toBeVisible();
    await expect(page.locator("#mmp-map-parking-lot > [data-mmp-original-map-id]")).toHaveCount(0);
  }

  expect(pageErrors).toEqual([]);
});

test("Nodes settings opens node table and add-node wizard", async ({ page }) => {
  await page.goto(`${baseUrl}/settings/nodes`, { waitUntil: "domcontentloaded" });

  await expect(page.getByRole("heading", { name: "Nodes" })).toBeVisible();
  // Either the node table or the empty state renders depending on seeded data.
  await expect(page.locator(".nodes-page")).toBeVisible();
  await page.getByRole("button", { name: /Add Node/ }).click();
  await expect(page.locator(".node-wizard")).toBeVisible();
  await expect(page.getByRole("button", { name: "Audio / Sensor" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Camera / PTZ" })).toBeVisible();
  // Scan & Pair is a disabled placeholder card.
  await expect(page.locator(".node-wizard-card.is-disabled")).toHaveAttribute(
    "aria-disabled",
    "true",
  );
});

test("transcript review displays the complete persisted text", async ({ page }) => {
  const fullTranscript = "This is the complete transcript, including the words that do not fit in the COP preview.";
  await page.route("**/api/v1/transcripts/txt-review-ui", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        id: "txt-review-ui",
        node_id: "node-a",
        sensor_id: "node-a:ch0",
        start_ns: 1700000000000000000,
        end_ns: 1700000005000000000,
        text: fullTranscript,
        model: "moonshine",
        trigger_confidence: 0.92,
        detection_id: null,
        created_ns: 1700000006000000000,
      }),
    });
  });

  await page.goto(`${baseUrl}/audio/t/txt-review-ui`, { waitUntil: "domcontentloaded" });

  await expect(page.locator(".transcript-review-card")).toBeVisible();
  await expect(page.locator(".transcript-review-text")).toHaveText(fullTranscript);
  await expect(page.getByRole("link", { name: "Analysis" })).toHaveClass(/active/);
});
