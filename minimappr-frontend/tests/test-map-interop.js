const assert = require("node:assert/strict");
const test = require("node:test");

globalThis.maplibregl = {
  addProtocol() {},
};

require("../maplibre-interop.js");

test("covariance ellipse produces a closed polygon", () => {
  const coordinates = globalThis.mapInterop._test.covarianceEllipseCoordinates(
    44.987,
    -93.258,
    [[25, 0], [0, 9]],
  );
  assert.ok(Array.isArray(coordinates));
  assert.equal(coordinates.length, 73);
  assert.ok(Math.abs(coordinates[0][0] - coordinates[coordinates.length - 1][0]) < 1e-12);
  assert.ok(Math.abs(coordinates[0][1] - coordinates[coordinates.length - 1][1]) < 1e-12);
});

test("invalid covariance is ignored", () => {
  assert.equal(
    globalThis.mapInterop._test.covarianceEllipseCoordinates(44.987, -93.258, null),
    null,
  );
});

test("bearing wedge starts at origin and points east for 90 degrees", () => {
  const coordinates = globalThis.mapInterop._test.bearingWedgeCoordinates(
    44.987,
    -93.258,
    90,
    10,
    1000,
  );
  assert.ok(Array.isArray(coordinates));
  assert.ok(coordinates.length > 8);
  assert.equal(coordinates[0][0], -93.258);
  assert.equal(coordinates[0][1], 44.987);
  assert.deepEqual(coordinates[0], coordinates[coordinates.length - 1]);
  const farthestLon = Math.max(...coordinates.map((point) => point[0]));
  assert.ok(farthestLon > -93.258);
});

test("css color fallback is returned without a DOM", () => {
  const color = globalThis.mapInterop._test.readCssColor("--missing", "#abc123");
  assert.equal(color, "#abc123");
});

test("zone lat/lon input is converted to closed maplibre coordinates", () => {
  const coordinates = globalThis.mapInterop._test.latLngsToClosedCoordinates([
    [45.0, -93.0],
    [45.0, -92.99],
    [45.01, -92.99],
  ]);
  assert.deepEqual(coordinates[0], [-93.0, 45.0]);
  assert.deepEqual(coordinates[coordinates.length - 1], coordinates[0]);
});

test("closed maplibre coordinates convert back to open lat/lon vertices", () => {
  const latlngs = globalThis.mapInterop._test.coordinatesToLatLngs([
    [-93.0, 45.0],
    [-92.99, 45.0],
    [-92.99, 45.01],
    [-93.0, 45.0],
  ]);
  assert.deepEqual(latlngs, [
    [45.0, -93.0],
    [45.0, -92.99],
    [45.01, -92.99],
  ]);
});

test("basemap theme paint changes brightness budget", () => {
  const dark = globalThis.mapInterop._test.basemapPaintForTheme("dark");
  const light = globalThis.mapInterop._test.basemapPaintForTheme("light");
  assert.ok(dark["raster-brightness-max"] < light["raster-brightness-max"]);
  assert.ok(dark["raster-brightness-min"] < light["raster-brightness-min"]);
});

test("existing map is reused only for the same live container element", () => {
  const container = { id: "mmp-map", isConnected: true };
  const existingMap = {
    getContainer() {
      return container;
    },
  };

  assert.equal(
    globalThis.mapInterop._test.canReuseExistingMap(existingMap, container, "mmp-map"),
    true,
  );
});

test("same map id with a replaced container forces map recreation", () => {
  const oldContainer = { id: "mmp-map", isConnected: false };
  const newContainer = { id: "mmp-map", isConnected: true };
  const existingMap = {
    getContainer() {
      return oldContainer;
    },
  };

  assert.equal(
    globalThis.mapInterop._test.canReuseExistingMap(existingMap, newContainer, "mmp-map"),
    false,
  );
});

test("parked map can be reattached to a replacement live container", () => {
  const parkedContainer = { id: "mmp-map-parked", isConnected: true };
  const replacementContainer = { id: "mmp-map", isConnected: true };
  const existingMap = {
    getContainer() {
      return parkedContainer;
    },
  };

  assert.equal(
    globalThis.mapInterop._test.canReattachParkedMap(
      existingMap,
      replacementContainer,
      "mmp-map",
      parkedContainer,
      "mmp-map",
    ),
    true,
  );
});

test("parked map is not reattached to the wrong container id", () => {
  const parkedContainer = { id: "mmp-map-parked", isConnected: true };
  const replacementContainer = { id: "other-map", isConnected: true };
  const existingMap = {
    getContainer() {
      return parkedContainer;
    },
  };

  assert.equal(
    globalThis.mapInterop._test.canReattachParkedMap(
      existingMap,
      replacementContainer,
      "mmp-map",
      parkedContainer,
      "mmp-map",
    ),
    false,
  );
});
