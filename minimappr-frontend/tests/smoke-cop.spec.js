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
  await expect(page.locator(".compact-table")).toBeVisible();
  await page.getByRole("button", { name: "Add Node" }).click();
  await expect(page.getByText("Add Node").nth(1)).toBeVisible();
  await expect(page.getByRole("button", { name: "Audio / Sensor" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Camera / PTZ" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Scan & Pair" })).toBeDisabled();
});
