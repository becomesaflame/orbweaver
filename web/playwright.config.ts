import { defineConfig, devices } from "@playwright/test";

// Keep off :8080 so a local run cannot reuse the production gateway on this host.
const port = Number(process.env.ORBWEAVER_E2E_PORT || 18080);
const baseURL = process.env.ORBWEAVER_BASE_URL || `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 30_000,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL,
    trace: "off",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // GitHub-hosted Ubuntu already has Google Chrome; skip downloading Chromium.
        ...(process.env.CI ? { channel: "chrome" } : {}),
      },
    },
  ],
  webServer: {
    command: `python -m uvicorn orbweaver.app:app --host 127.0.0.1 --port ${port}`,
    url: `${baseURL}/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    cwd: "../backend",
    env: {
      ...process.env,
      ORBWEAVER_STORE: "memory",
      ORBWEAVER_CRON: "0",
      // Local smoke run with the default JWT secret; serve refuses it otherwise.
      ORBWEAVER_DEV_INSECURE: "1",
    },
  },
});
