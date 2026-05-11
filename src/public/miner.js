const logEl = document.getElementById("log");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
let running = false;
let device, pipeline, uniformBuf, resultBuf, stagingBuf, bindGroup;

function log(...args) {
  const line = args.map(x => typeof x === "string" ? x : JSON.stringify(x, (_, v) => typeof v === "bigint" ? v.toString() : v)).join(" ");
  console.log(line);
  logEl.textContent += `${new Date().toISOString()} ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function hexToBytes(hex) {
  const s = hex.startsWith("0x") ? hex.slice(2) : hex;
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function bytesToHex(bytes) {
  return "0x" + [...bytes].map(b => b.toString(16).padStart(2, "0")).join("");
}

function challengeWords(bytes) {
  const out = new Uint32Array(8);
  for (let i = 0; i < 8; i++) out[i] = bytes[4*i] | (bytes[4*i+1] << 8) | (bytes[4*i+2] << 16) | (bytes[4*i+3] << 24);
  return out;
}

function difficultyWords(bytes) {
  const out = new Uint32Array(8);
  for (let i = 0; i < 8; i++) out[i] = ((bytes[4*i] << 24) | (bytes[4*i+1] << 16) | (bytes[4*i+2] << 8) | bytes[4*i+3]) >>> 0;
  return out;
}

function decodeResult(words) {
  const found = words[0] > 0;
  const nonce = new Uint8Array(32);
  if (found) {
    const lo = words[1] >>> 0, hi = words[2] >>> 0;
    nonce[24] = hi >>> 24; nonce[25] = hi >>> 16; nonce[26] = hi >>> 8; nonce[27] = hi;
    nonce[28] = lo >>> 24; nonce[29] = lo >>> 16; nonce[30] = lo >>> 8; nonce[31] = lo;
  }
  const hash = new Uint8Array(32);
  for (let i = 0; i < 8; i++) {
    const x = words[4 + i] >>> 0;
    hash[4*i] = x >>> 24; hash[4*i+1] = x >>> 16; hash[4*i+2] = x >>> 8; hash[4*i+3] = x;
  }
  return { found, nonce: bytesToHex(nonce), hash: bytesToHex(hash) };
}

async function loadWgsl() {
  const js = await fetch("/mine-page-js.txt").then(r => r.text());
  const match = js.match(/let y=(.*?);function w/s);
  if (!match) throw new Error("cannot extract WGSL from mine-page-js.txt");
  return Function(`return ${match[1]}`)();
}

async function initGpu() {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no high-performance GPU adapter");
  device = await adapter.requestDevice();
  const wgsl = await loadWgsl();
  const module = device.createShaderModule({ code: wgsl });
  pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  uniformBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  resultBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  stagingBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }, { binding: 1, resource: { buffer: resultBuf } }]
  });
  log("gpu ready", adapter.info?.device || adapter.info?.vendor || "webgpu");
}

async function getState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "state failed");
  return data;
}

async function submit(nonce, hash) {
  log("FOUND", { nonce, hash }, "submitting...");
  const r = await fetch("/api/submit", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ nonce, hash })
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "submit failed");
  log("TX", data);
}

async function mineLoop() {
  if (!device) await initGpu();
  running = true;
  let state = await getState();
  log("state", state);
  if (!state.open) {
    log("mining is not open yet; waiting");
  }

  let challenge = challengeWords(hexToBytes(state.challenge));
  let difficulty = difficultyWords(hexToBytes(state.difficulty));
  const uniforms = new Uint32Array(20);
  uniforms.set(challenge, 0);
  uniforms.set(difficulty, 8);
  const seed = new Uint32Array(2);
  crypto.getRandomValues(seed);
  let lo = seed[0] >>> 0, hi = seed[1] >>> 0;
  const workgroups = Math.min(65535, device.limits.maxComputeWorkgroupsPerDimension || 65535);
  const hashesPerDispatch = 64 * workgroups * 16;
  let total = 0;
  const startedAt = performance.now();
  let lastStatePoll = 0;
  const statePollMs = 3000;

  while (running) {
    const pollNow = performance.now();
    if (pollNow - lastStatePoll >= statePollMs) {
      lastStatePoll = pollNow;
      const fresh = await getState();
      if (fresh.epoch !== state.epoch || fresh.challenge !== state.challenge || fresh.difficulty !== state.difficulty) {
        state = fresh;
        challenge = challengeWords(hexToBytes(state.challenge));
        difficulty = difficultyWords(hexToBytes(state.difficulty));
        uniforms.set(challenge, 0);
        uniforms.set(difficulty, 8);
        log("retarget", state);
      }
    }

    uniforms[16] = lo; uniforms[17] = hi; uniforms[18] = 0; uniforms[19] = 0;
    device.queue.writeBuffer(uniformBuf, 0, uniforms);
    device.queue.writeBuffer(resultBuf, 0, new Uint32Array(12));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    enc.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, 48);
    device.queue.submit([enc.finish()]);
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();

    total += hashesPerDispatch;
    const now = performance.now();
    const elapsedSec = Math.max(0.001, (now - startedAt) / 1000);
    const avgRate = total / elapsedSec;
    if (total % (hashesPerDispatch * 8) === 0) log(`${(avgRate / 1e6).toFixed(2)} MH/s avg`, `total=${total.toLocaleString()}`);

    const hit = decodeResult(result);
    if (hit.found) {
      await submit(hit.nonce, hit.hash);
      state = await getState();
      challenge = challengeWords(hexToBytes(state.challenge));
      difficulty = difficultyWords(hexToBytes(state.difficulty));
      uniforms.set(challenge, 0);
      uniforms.set(difficulty, 8);
    }

    const nextLo = (lo + hashesPerDispatch) >>> 0;
    if (nextLo < lo) hi = (hi + 1) >>> 0;
    lo = nextLo;
  }
}

startBtn.onclick = () => mineLoop().catch(e => log("ERROR", e.message || String(e)));
stopBtn.onclick = () => { running = false; log("stopping"); };

fetch("/api/ping").then(r => r.json()).then(x => log("relay", x)).catch(e => log("relay error", e.message));
