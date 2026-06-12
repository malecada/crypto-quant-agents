/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",          // served by FastAPI at site root
  build: { outDir: "dist" },
  test: { environment: "node" },
});
