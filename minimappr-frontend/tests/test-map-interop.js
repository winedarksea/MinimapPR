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
