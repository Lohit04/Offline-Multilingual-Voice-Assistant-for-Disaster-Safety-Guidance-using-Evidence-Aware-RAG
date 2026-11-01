// Minimal service worker -- only exists so the browser treats this as an
// installable PWA. Deliberately does NOT cache/serve the app offline from
// the phone's own storage: the AI backend (Ollama, retrieval, ASR/TTS)
// still has to run on a server somewhere on the local network. Without
// that server reachable, the app has nothing to talk to regardless of
// what the service worker caches -- caching the static shell would be
// misleading about what "offline" means here.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.clients.claim());
