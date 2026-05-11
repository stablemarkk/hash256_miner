import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ethers } from "ethers";
import { getState, submitNonce } from "./server.js";
import { loadDotEnv } from "./env.js";

loadDotEnv();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const backend = (process.env.MINER_BACKEND || "opencl").toLowerCase();
const exeName = backend === "cuda" ? "hash256-cuda.exe" : "hash256-opencl.exe";
const exe = path.resolve(__dirname, "..", "native", "bin", exeName);
const workerCount = Number(process.env.MINER_WORKERS || process.env.OPENCL_WORKERS || 1);
const openclGlobalWorkItems = process.env.OPENCL_GLOBAL || String(1 << 22);
const cudaBlocks = process.env.CUDA_BLOCKS || "0";
const cudaThreads = process.env.CUDA_THREADS || "384";
const cudaHashesPerThread = process.env.CUDA_HASHES_PER_THREAD || "96";
const txProvider = new ethers.JsonRpcProvider(process.env.ETH_TX_RPC_URL || "https://rpc.mevblocker.io/fast", 1);

if (!process.env.PRIVATE_KEY) {
  throw new Error("PRIVATE_KEY env is required");
}

const wallet = new ethers.Wallet(process.env.PRIVATE_KEY);
let children = [];
let activeKey = null;
let submitting = false;
const startedAt = Date.now();
const colorsEnabled = process.env.MINER_LOG_COLOR !== "0";
const stats = {
  found: 0,
  submitted: 0,
  txConfirmed: 0,
  txPending: 0,
  lastMhs: 0,
  bestMhs: 0,
  totalHashes: 0
};

function stamp() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

const levelColors = {
  START: "\x1b[36m",
  STATE: "\x1b[36m",
  WAIT: "\x1b[33m",
  PROGRESS: "\x1b[32m",
  STATS: "\x1b[35m",
  FOUND: "\x1b[92m",
  TX_SUBMIT_START: "\x1b[33m",
  TX_BROADCAST: "\x1b[32m",
  TX_CONFIRMED: "\x1b[92m",
  TX_PENDING: "\x1b[33m",
  TX_SUBMIT_ERROR: "\x1b[31m",
  WORKER_STOP: "\x1b[33m",
  MINER_EXIT: "\x1b[31m",
  MINER_STDERR: "\x1b[31m",
  MONITOR_ERROR: "\x1b[31m",
  STOP: "\x1b[33m"
};

function colorLevel(level) {
  if (!colorsEnabled) return `[${level}]`;
  const color = levelColors[level] || "\x1b[37m";
  return `${color}[${level}]\x1b[0m`;
}

function stringify(value) {
  return JSON.stringify(value, (_, item) => typeof item === "bigint" ? item.toString() : item);
}

function log(event, fields = {}) {
  const suffix = Object.entries(fields)
    .map(([key, value]) => `${key}=${typeof value === "object" && value !== null ? stringify(value) : value}`)
    .join(" ");
  console.log(`${stamp()} ${colorLevel(event)}${suffix ? ` ${suffix}` : ""}`);
}

function logError(event, error, fields = {}) {
  log(event, { ...fields, error: error?.message || String(error) });
}

async function waitForReceipt(hash, timeoutMs = 45000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const receipt = await txProvider.getTransactionReceipt(hash).catch(() => null);
    if (receipt) return receipt;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  return null;
}

function stopChild(reason = "restart") {
  for (const child of children) {
    if (child && !child.killed) {
      child.stopReason = reason;
      child.kill();
    }
  }
  children = [];
}

async function startMiner(state) {
  stopChild("state_change");
  activeKey = `${state.challenge}:${state.difficulty}:${state.epoch}`;
  log("STATE", {
    miner: state.miner,
    open: state.open,
    epoch: state.epoch,
    epochBlocksLeft: state.epochBlocksLeft,
    difficulty: state.difficulty,
    challenge: state.challenge
  });
  if (!state.open) {
    log("WAIT", { reason: "mining_not_open" });
    return;
  }
  for (let worker = 0; worker < workerCount; worker++) {
    const minerArgs = backend === "cuda"
      ? [state.challenge, state.difficulty, cudaBlocks, cudaThreads, cudaHashesPerThread]
      : [state.challenge, state.difficulty, openclGlobalWorkItems];
    const child = spawn(exe, minerArgs, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"]
    });
    children.push(child);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", async chunk => {
    for (const line of chunk.split(/\r?\n/).filter(Boolean)) {
      const progress = /^PROGRESS\s+([0-9.]+)\s+MH\/s\s+total=(\d+)/.exec(line);
      if (progress) {
        const mhs = Number(progress[1]);
        const total = Number(progress[2]);
        stats.lastMhs = mhs;
        stats.bestMhs = Math.max(stats.bestMhs, mhs);
        stats.totalHashes += total;
        log("PROGRESS", {
          worker,
          mhs: mhs.toFixed(2),
          bestMhs: stats.bestMhs.toFixed(2),
          batchHashes: total,
          totalGh: (stats.totalHashes / 1e9).toFixed(2)
        });
        continue;
      }
      const found = /^FOUND\s+(0x[0-9a-fA-F]{64})\s+(0x[0-9a-fA-F]{64})/.exec(line);
      if (found && !submitting) {
        submitting = true;
        stats.found++;
        stopChild("found_nonce");
        try {
          log("FOUND", { worker, nonce: found[1], hash: found[2], foundCount: stats.found });
          log("TX_SUBMIT_START", { nonce: found[1] });
          const tx = await submitNonce(found[1]);
          stats.submitted++;
          log("TX_BROADCAST", {
            hash: tx.hash,
            nonce: tx.nonce,
            submitMs: tx.submitMs,
            gasLimit: tx.gasLimit,
            maxFeeGwei: ethers.formatUnits(tx.maxFeePerGas, "gwei"),
            tipGwei: ethers.formatUnits(tx.maxPriorityFeePerGas, "gwei"),
            rpcs: tx.broadcast.map(item => item.rpc).join(",")
          });
          const receipt = await waitForReceipt(tx.hash);
          if (receipt) {
            stats.txConfirmed++;
            log("TX_CONFIRMED", {
              hash: tx.hash,
              block: receipt.blockNumber,
              status: receipt.status,
              gasUsed: receipt.gasUsed
            });
          } else {
            stats.txPending++;
            log("TX_PENDING", { hash: tx.hash, waitMs: 45000 });
          }
        } catch (error) {
          logError("TX_SUBMIT_ERROR", error, { nonce: found[1] });
        } finally {
          submitting = false;
          const next = await getState(wallet.address);
          await startMiner(next);
        }
      }
      log("MINER", { worker, line });
    }
    });
    child.stderr.on("data", chunk => {
    for (const line of chunk.split(/\r?\n/).filter(Boolean)) {
      if (!/warning/i.test(line)) log("MINER_STDERR", { worker, line });
    }
    });
    child.on("exit", (code, signal) => {
      if (child.stopReason) {
        log("WORKER_STOP", { worker, reason: child.stopReason, code, signal });
        return;
      }
      if (!submitting && code !== 0) log("MINER_EXIT", { worker, code, signal });
    });
  }
}

async function monitor() {
  const state = await getState(wallet.address);
  if (`${state.challenge}:${state.difficulty}:${state.epoch}` !== activeKey) {
    await startMiner(state);
  }
}

process.on("SIGINT", () => {
  log("STOP", { signal: "SIGINT" });
  stopChild("sigint");
  process.exit(0);
});

log("START", {
  backend: backend.toUpperCase(),
  signer: wallet.address,
  exe
});
if (backend === "cuda") {
  log("START", { cudaBlocks, cudaThreads, cudaHashesPerThread });
} else {
  log("START", { globalWorkItems: openclGlobalWorkItems });
}
log("START", { workers: workerCount });
await monitor();
setInterval(() => monitor().catch(error => logError("MONITOR_ERROR", error)), 3000);
setInterval(() => log("STATS", {
  uptimeSec: Math.floor((Date.now() - startedAt) / 1000),
  lastMhs: stats.lastMhs.toFixed(2),
  bestMhs: stats.bestMhs.toFixed(2),
  totalGh: (stats.totalHashes / 1e9).toFixed(2),
  found: stats.found,
  submitted: stats.submitted,
  confirmed: stats.txConfirmed,
  pending: stats.txPending
}), 30000);
