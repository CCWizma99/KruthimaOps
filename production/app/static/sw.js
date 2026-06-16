const CACHE_NAME = 'floodguard-cache-v2';

const PRECACHE_URLS = [
  '/',
  '/static/index.html',
  '/static/style.css',
  '/static/app.js',
  '/static/evacuation_points.json',
  '/static/manifest.json'
];

// ── Install: pre-cache all core shell assets ────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// ── Activate: purge old caches ──────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch: Cache-First for everything ───────────────────────────────
// Serves from cache immediately so the PWA loads instantly offline.
// After serving the cached copy, a background fetch updates the cache
// so the next launch gets fresh data (stale-while-revalidate).
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.open(CACHE_NAME).then(cache =>
      cache.match(event.request).then(cachedResponse => {
        // Background revalidation: fetch from network and update cache
        const networkFetch = fetch(event.request).then(networkResponse => {
          // Cache successful responses AND opaque responses (CORS tiles)
          if (networkResponse && (networkResponse.ok || networkResponse.type === 'opaque')) {
            cache.put(event.request, networkResponse.clone());
          }
          return networkResponse;
        }).catch(() => {
          // Network unavailable — nothing to update
        });

        // Return cached version immediately, or wait for network if no cache
        return cachedResponse || networkFetch;
      })
    )
  );
});
