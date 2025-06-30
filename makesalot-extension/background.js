// background.js
// Service worker para MakesALot Trading Extension

self.addEventListener("install", (event) => {
  console.log("MakesALot Extension instalado.");
});

self.addEventListener("activate", (event) => {
  console.log("MakesALot Extension ativado.");
});

// Handler para mensagens do popup/content script
self.addEventListener("message", async (event) => {
  if (!event.data || !event.data.type) return;

  if (event.data.type === "API_REQUEST") {
    try {
      const { endpoint, method = "GET", body } = event.data;
      const response = await fetch(endpoint, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await response.json();
      event.ports[0].postMessage({ success: true, data });
    } catch (error) {
      event.ports[0].postMessage({ success: false, error: error.message });
    }
  }
});
