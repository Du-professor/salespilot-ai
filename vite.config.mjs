import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const apiOrigin = process.env.SALESPILOT_API_ORIGIN || "http://127.0.0.1:8000";
const proxy = { "/api/v1": { target: apiOrigin } };

export default defineConfig({
  build: {
    outDir: "dist/client",
  },
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    proxy,
    warmup: {
      clientFiles: ["./src/main.js"],
    },
  },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy },
  plugins: [vue()],
});
