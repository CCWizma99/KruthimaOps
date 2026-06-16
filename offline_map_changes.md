# FloodGuard SL — Offline Capabilities & Map Enhancements

This document summarizes the changes made to the dashboard codebase during this session to fully implement offline capabilities, fix the base map tile rendering issue, and add the "My Location" geolocation feature.

## 1. PWA & Service Worker (Offline Capabilities)

We introduced a Progressive Web App (PWA) architecture to allow the dashboard to be installed on mobile devices and accessed completely offline.

### `app/static/manifest.json` (New)
- Created the Web App Manifest defining the app's name, standalone display mode, background colors, and icon.
- This allows the app to be "Installed to Home Screen" on iOS and Android devices.

### `app/static/sw.js` (New/Rewritten)
- **Version 4 (`floodguard-cache-v4`)**: Created a robust Service Worker for caching all assets.
- **Cache-First (Stale-While-Revalidate) for Same-Origin**: All static shell assets (`index.html`, `style.css`, `app.js`) and API data are served from the cache instantly, while the SW fetches a fresh copy in the background for the *next* launch.
- **Network-First for Cross-Origin (Map Tiles)**: Added special handling to cache "opaque" CORS responses (status `0`). This ensures that the external CartoDB base map tiles, CesiumJS web workers, and fonts are successfully cached and available offline after the user's first visit.

### `app/static/index.html` (Updates)
- Linked the `manifest.json`.
- Added the `<script>` block to register `sw.js`.
- Implemented an aggressive auto-update mechanism (`reg.update()`) so new Service Worker versions immediately take control, bypassing the browser's default waiting lifecycle.

### `app/main.py` (Updates)
- Added a specific route for `/sw.js` so it's served from the root scope (`/`) instead of `/static/sw.js`. A Service Worker can only cache files within or below its own scope.

---

## 2. CesiumJS Base Map Fix

The dark CartoDB base map tiles were failing to render under the 3D risk graph.

### `app/static/app.js` (Updates)
- **Deprecated API Fix**: In CesiumJS 1.122, the `imageryProvider` constructor option and `addImageryProvider()` function were deprecated and silently failing.
- **Modern API**: Switched to the modern approach: `new Cesium.ImageryLayer(provider)` followed by `viewer.imageryLayers.add(layer)`.
- **Fallback Strategy**: The viewer now initializes with the default Cesium Ion satellite imagery (which is guaranteed to work with the provided token). We then dynamically add the CartoDB dark tiles on top and remove the satellite layer. If CartoDB fails to load due to network issues, the system gracefully falls back to the satellite map.

---

## 3. "My Location" Geolocation Feature

Added a feature allowing users to track their physical location on the 3D globe.

### `app/static/index.html` & `app/static/style.css` (Updates)
- Added a floating GPS crosshair button (`#my-location-btn`) to the top-right corner of the map.
- Added glowing CSS animations for the active tracking state.

### `app/static/app.js` (Updates)
- **`toggleMyLocation()`**: Requests GPS permissions via the browser's `navigator.geolocation` API.
- **`flyToMyLocation()`**: Automatically pans and tilts the Cesium 3D camera to focus on the user's coordinates.
- **`plotMyLocation()`**: Adds two entities to the globe:
  1. A bright blue pulsing dot with a "You are here" label floating above the terrain.
  2. A translucent blue circle on the ground representing the GPS accuracy radius.
- **Live Tracking**: Uses `watchPosition()` to continuously update the marker on the globe as the user moves.

---

## 4. "Last Refreshed" Offline Indicator

To help users trust the data when entirely offline, a timestamp indicator was added.

### `app/static/index.html` & `app/static/style.css` (Updates)
- Added a `refresh-chip` to the top header status bar (next to the LIVE chip).

### `app/static/app.js` & `app/static/sw.js` (Updates)
- **Service Worker Storage**: Whenever the SW successfully fetches a fresh copy of the app data from the network, it stores the current ISO timestamp in the cache as `/__last_refresh_time__`.
- **UI Logic**: `app.js` reads this pseudo-file from the cache on startup. It then calculates and displays a human-readable relative time (e.g., "Just now", "5m ago", "12h ago") so the user knows exactly how old their offline cache is.
