# MapLibre GL JS

- Version: 5.24.0
- Source: https://registry.npmjs.org/maplibre-gl/-/maplibre-gl-5.24.0.tgz
- License: BSD-3-Clause, copied in `LICENSE`

Upgrade steps:

1. Check the current stable version with `npm view maplibre-gl version dist.tarball license`.
2. Download the tarball and replace `maplibre-gl.js`, `maplibre-gl.css`, and `LICENSE`.
3. Run `npm test`, `cargo check`, and `NO_COLOR=false trunk build`.
4. Smoke-test COP and Analysis heatmap in online and offline basemap conditions.
