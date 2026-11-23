import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 127.0.0.1, not "localhost": on macOS with Node 17+ "localhost" can resolve to
// ::1 (IPv6) while uvicorn listens on 127.0.0.1, which makes every /api call fail.
// Override with FACEEVAL_API_URL if the API runs elsewhere.
const apiTarget = process.env.FACEEVAL_API_URL || "http://127.0.0.1:8000";
const proxy = { "/api": { target: apiTarget, changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
