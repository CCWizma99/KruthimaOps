const CACHE_NAME = 'floodguard-cache-v4';

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
