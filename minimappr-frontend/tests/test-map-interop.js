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

test("covariance ellipse preserves an anisotropic covariance shape", () => {
  const centerLat = 44.987;
  const centerLon = -93.258;
  const coordinates = globalThis.mapInterop._test.covarianceEllipseCoordinates(
    centerLat,
    centerLon,
    [[100, 0], [0, 4]],
  );
  const eastRadiusDegrees = Math.abs(coordinates[0][0] - centerLon);
  const northPoint = coordinates[18];
  const northRadiusDegrees = Math.abs(northPoint[1] - centerLat);

  // The expected axes are 10 m east/west and 2 m north/south.  Longitude
  // degrees are shorter at this latitude, so compare after the conversion.
  const eastRadiusM = eastRadiusDegrees * 111_320 * Math.cos(centerLat * Math.PI / 180);
  const northRadiusM = northRadiusDegrees * 111_320;
  assert.ok(Math.abs(eastRadiusM - 10) < 0.01);
  assert.ok(Math.abs(northRadiusM - 2) < 0.01);
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

function stubMap() {
  const sources = {};
  const layers = {};
  return {
    styleLoaded: false,
    isStyleLoaded() { return this.styleLoaded; },
    getSource(id) { return sources[id]; },
    addSource(id, spec) {
      sources[id] = { data: spec.data, setData(data) { this.data = data; } };
    },
    getLayer(id) { return layers[id]; },
    addLayer(spec) { layers[spec.id] = spec; },
    removeLayer(id) { delete layers[id]; },
    removeSource(id) { delete sources[id]; },
  };
}

test("generic layer upserts while the style loads coalesce and keep the latest data", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const interop = globalThis.mapInterop;
  const map = stubMap();
  interop._test.useMapForTest(map);
  const spec = JSON.stringify({ type: "geojson", layer: { type: "circle", paint: {} } });
  const first = { type: "FeatureCollection", features: [{ id: 1 }] };
  const second = { type: "FeatureCollection", features: [{ id: 2 }] };
  interop.ensureLayer("entity-a", spec);
  interop.setLayerData("entity-a", JSON.stringify(first));
  interop.ensureLayer("entity-a", spec);
  interop.setLayerData("entity-a", JSON.stringify(second));
  interop.ensureLayer("entity-b", spec);
  interop.removeLayer("entity-b");
  t.mock.timers.tick(500);
  assert.equal(map.getSource("entity-a"), undefined); // still loading: nothing added yet
  map.styleLoaded = true;
  t.mock.timers.tick(500);
  assert.deepEqual(map.getSource("entity-a").data, second); // latest wins, not reset to empty
  assert.ok(map.getLayer("entity-a"));
  assert.equal(map.getSource("entity-b"), undefined); // removal cancelled the queued add
  interop.ensureLayer("entity-a", spec); // an existing source keeps its data
  assert.deepEqual(map.getSource("entity-a").data, second);
  interop._test.useMapForTest(null);
});
