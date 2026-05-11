import fs from "node:fs";
import { once } from "node:events";
import { chromium } from "playwright-core";
import { startServer } from "./server.js";
import { loadDotEnv } from "./env.js";

loadDotEnv();

const PORT = Number(process.env.PORT || 8787);
const chromeCandidates = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
].filter(Boolean);

function findChrome() {
  const found = chromeCandidates.find(p => fs.existsSync(p));
  if (!found) throw new Error("Chrome/Edge not found; set CHROME_PATH");
  return found;
}

if (!process.env.PRIVATE_KEY) {
  throw new Error("PRIVATE_KEY env is required");
}

const server = startServer(PORT);
await once(server, "listening");

const browser = await chromium.launch({
  executablePath: findChrome(),
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU,UseSkiaRenderer",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--no-first-run",
    "--no-default-browser-check"
  ]
});

const page = await browser.newPage();
page.on("console", msg => {
  const text = msg.text();
  if (/relay|gpu ready|state|retarget|MH\/s|FOUND|TX|ERROR/.test(text)) {
    console.log(text);
  }
});
page.on("pageerror", error => console.error("[pageerror]", error.message));

process.on("SIGINT", async () => {
  console.log("stopping...");
  await browser.close().catch(() => {});
  server.close();
  process.exit(0);
});

await page.goto(`http://127.0.0.1:${PORT}`, { waitUntil: "domcontentloaded" });
await page.click("#start");
console.log("CLI miner started. Ctrl+C to stop.");

await new Promise(() => {});
