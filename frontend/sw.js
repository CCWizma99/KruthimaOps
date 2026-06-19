const CACHE_NAME = 'flood-timeline-cache-v4';

const PRECACHE_URLS = [
  '/',
  '/static/index.html',
  '/static/style.css',
  '/static/app.js',
  '/static/evacuation_points.json',
  '/static/manifest.json'
];

// ── Install: pre-cache core shell assets ────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// ── Activate: purge ALL old caches ──────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch ───────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // ── Same-origin: Stale-While-Revalidate ──
  // Serves from cache instantly, background-fetches fresh copy for next time
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.open(CACHE_NAME).then(cache =>
        cache.match(event.request).then(cached => {
          const networkFetch = fetch(event.request).then(resp => {
            if (resp && resp.ok) {
              cache.put(event.request, resp.clone());
              // Store timestamp of last successful fetch
              cache.put(
                new Request('/__last_refresh_time__'),
                new Response(new Date().toISOString())
              );
            }
            return resp;
          }).catch(() => undefined);
          return cached || networkFetch;
        })
      )
    );
    return;
  }

  // ── Cross-origin (map tiles, CesiumJS, fonts): Network-First with cache ──
  // These are opaque responses (status=0) so we cache them with special handling.
  // This lets the full map work offline after first visit.
  event.respondWith(
    fetch(event.request).then(response => {
      // Cache successful (200) AND opaque (type=opaque, status=0) responses
      if (response && (response.ok || response.type === 'opaque')) {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
      }
      return response;
    }).catch(() => {
      // Network failed — serve from cache if available
      return caches.match(event.request);
    })
  );
});


/* --- EVACUATION MODULE SERVICE WORKER LOGIC --- */
// Flood Timeline service worker — caches OSM raster tiles

const TILE_CACHE = 'flood-timeline-tiles-v1';
const TILE_HOSTS = ['a.tile.openstreetmap.org', 'b.tile.openstreetmap.org', 'c.tile.openstreetmap.org'];

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (TILE_HOSTS.includes(url.hostname)) {
    event.respondWith(handleTileRequest(event.request));
  }
});

async function handleTileRequest(request) {
  const cache = await caches.open(TILE_CACHE);
  const cached = await cache.match(request);
  if (cached) {
    fetch(request).then((resp) => { if (resp && resp.ok) cache.put(request, resp); }).catch(() => {});
    return cached;
  }
  try {
    const resp = await fetch(request);
    if (resp && resp.ok) cache.put(request, resp.clone());
    return resp;
  } catch (err) {
    return new Response(
      new Uint8Array([0x47,0x49,0x46,0x38,0x39,0x61,0x01,0x00,0x01,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x21,0xf9,0x04,0x01,0x00,0x00,0x00,0x00,0x2c,0x00,0x00,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0x02,0x02,0x44,0x01,0x00,0x3b]),
      { status: 200, headers: { 'Content-Type': 'image/gif' } }
    );
  }
}

function tileUrl(z, x, y, host) {
  return `https://${host}/${z}/${x}/${y}.png`;
}

function lngToTileX(lng, z) {
  return Math.floor((lng + 180) / 360 * Math.pow(2, z));
}

function latToTileY(lat, z) {
  const rad = lat * Math.PI / 180;
  return Math.floor((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * Math.pow(2, z));
}

self.addEventListener('message', async (event) => {
  const msg = event.data || {};
  if (msg.type === 'CACHE_TILES') {
    const { bounds, zooms } = msg;
    const cache = await caches.open(TILE_CACHE);
    const tasks = [];
    zooms.forEach((z) => {
      const xMin = lngToTileX(bounds.w, z);
      const xMax = lngToTileX(bounds.e, z);
      const yMin = latToTileY(bounds.n, z);
      const yMax = latToTileY(bounds.s, z);
      for (let x = xMin; x <= xMax; x++) {
        for (let y = yMin; y <= yMax; y++) {
          const host = TILE_HOSTS[(x + y) % TILE_HOSTS.length];
          const url = tileUrl(z, x, y, host);
          tasks.push(
            fetch(url).then((resp) => { if (resp && resp.ok) cache.put(url, resp); }).catch(() => {})
          );
        }
      }
    });
    // Use allSettled to not block on individual failures
    await Promise.allSettled(tasks);
    console.log(`[SW] Cached ${tasks.length} tiles`);
  } else if (msg.type === 'CLEAR_TILES') {
    await caches.delete(TILE_CACHE);
    console.log('[SW] Tile cache cleared');
  }
});