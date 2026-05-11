import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ethers } from "ethers";
import { loadDotEnv } from "./env.js";

loadDotEnv();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const publicDir = path.join(__dirname, "public");

const CONTRACT = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";
const READ_RPC = process.env.ETH_RPC_URL || "https://ethereum.publicnode.com";
const TX_RPC = process.env.ETH_TX_RPC_URL || "https://rpc.mevblocker.io/fast";
const BROADCAST_RPCS = (process.env.BROADCAST_RPCS ||
  "https://rpc.mevblocker.io/fast,https://rpc.flashbots.net/fast")
  .split(",")
  .map(item => item.trim())
  .filter(Boolean);
const TIP_GWEI = process.env.TIP_GWEI || "25";
const PORT = Number(process.env.PORT || 8787);

const abi = [
  "function genesisComplete() view returns (bool)",
  "function miningState() view returns (uint256 era,uint256 reward,uint256 difficulty,uint256 minted,uint256 remaining,uint256 epoch,uint256 epochBlocksLeft)",
  "function getChallenge(address miner) view returns (bytes32)",
  "function mine(uint256 nonce)",
  "event Mined(address indexed miner,uint256 nonce,uint256 reward,uint256 era)"
];

const readProvider = new ethers.JsonRpcProvider(READ_RPC, 1);
const txProvider = new ethers.JsonRpcProvider(TX_RPC, 1);
const iface = new ethers.Interface(abi);
const readContract = new ethers.Contract(CONTRACT, abi, readProvider);

function json(res, status, body) {
  const data = JSON.stringify(body, (_, value) => typeof value === "bigint" ? value.toString() : value);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*"
  });
  res.end(data);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", chunk => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".txt")) return "text/plain; charset=utf-8";
  if (file.endsWith(".json")) return "application/json; charset=utf-8";
  return "application/octet-stream";
}

function requireWallet() {
  if (!process.env.PRIVATE_KEY) {
    throw new Error("PRIVATE_KEY is not set");
  }
  return new ethers.Wallet(process.env.PRIVATE_KEY, txProvider);
}

export async function getState(address) {
  const [open, state, challenge] = await Promise.all([
    readContract.genesisComplete(),
    readContract.miningState(),
    readContract.getChallenge(address)
  ]);
  return {
    contract: CONTRACT,
    miner: address,
    open,
    era: state.era,
    reward: state.reward,
    difficulty: "0x" + BigInt(state.difficulty).toString(16).padStart(64, "0"),
    minted: state.minted,
    remaining: state.remaining,
    epoch: state.epoch,
    epochBlocksLeft: state.epochBlocksLeft,
    challenge
  };
}

export async function submitNonce(nonceHex) {
  const wallet = requireWallet();
  const nonce = BigInt(nonceHex);
  const data = iface.encodeFunctionData("mine", [nonce]);
  const txContract = new ethers.Contract(CONTRACT, abi, wallet);
  let gasLimit = 300000n;
  try {
    const estimate = await txContract.mine.estimateGas(nonce);
    gasLimit = estimate * 3n / 2n;
  } catch {
    gasLimit = 300000n;
  }
  if (gasLimit < 200000n) gasLimit = 200000n;
  if (gasLimit > 450000n) gasLimit = 450000n;

  const block = await txProvider.getBlock("latest");
  const tip = ethers.parseUnits(TIP_GWEI, "gwei");
  const base = block?.baseFeePerGas || ethers.parseUnits("5", "gwei");
  const tx = {
    to: CONTRACT,
    data,
    gasLimit,
    maxPriorityFeePerGas: tip,
    maxFeePerGas: base * 3n + tip,
    nonce: await txProvider.getTransactionCount(wallet.address, "pending"),
    chainId: 1
  };

  const started = Date.now();
  const signed = await wallet.signTransaction(tx);
  const txHash = ethers.keccak256(signed);
  const payload = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "eth_sendRawTransaction",
    params: [signed]
  });
  const results = await Promise.allSettled(BROADCAST_RPCS.map(async rpc => {
    const response = await fetch(rpc, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: payload
    });
    const body = await response.json();
    if (body.error && !/already known|known transaction|already imported/i.test(body.error.message || "")) {
      throw new Error(`${rpc}: ${body.error.message || JSON.stringify(body.error)}`);
    }
    return { rpc, result: body.result || txHash };
  }));
  const accepted = results
    .filter(result => result.status === "fulfilled")
    .map(result => result.value);
  if (accepted.length === 0) {
    throw new Error(results.map(result => result.reason?.message || String(result.reason)).join("; "));
  }
  return {
    hash: txHash,
    from: wallet.address,
    nonce: nonceHex,
    submitMs: Date.now() - started,
    gasLimit,
    maxPriorityFeePerGas: tx.maxPriorityFeePerGas,
    maxFeePerGas: tx.maxFeePerGas,
    broadcast: accepted
  };
}

export function createServer() {
  return http.createServer(async (req, res) => {
  try {
    if (req.method === "OPTIONS") return json(res, 204, {});
    const url = new URL(req.url || "/", `http://${req.headers.host}`);

    if (url.pathname === "/api/state") {
      const walletAddress = process.env.PRIVATE_KEY
        ? new ethers.Wallet(process.env.PRIVATE_KEY).address
        : url.searchParams.get("address");
      if (!walletAddress || !ethers.isAddress(walletAddress)) {
        return json(res, 400, { error: "Set PRIVATE_KEY or pass ?address=0x..." });
      }
      return json(res, 200, await getState(walletAddress));
    }

    if (url.pathname === "/api/submit" && req.method === "POST") {
      const payload = JSON.parse(await readBody(req));
      if (!/^0x[0-9a-fA-F]{64}$/.test(payload.nonce || "")) {
        return json(res, 400, { error: "nonce must be 0x + 32 bytes" });
      }
      return json(res, 200, await submitNonce(payload.nonce));
    }

    if (url.pathname === "/api/ping") {
      return json(res, 200, {
        ok: true,
        signer: process.env.PRIVATE_KEY ? new ethers.Wallet(process.env.PRIVATE_KEY).address : null,
        readRpc: READ_RPC,
        txRpc: TX_RPC,
        broadcastRpcs: BROADCAST_RPCS,
        tipGwei: TIP_GWEI
      });
    }

    const file = url.pathname === "/"
      ? path.join(publicDir, "miner.html")
      : path.normalize(path.join(url.pathname === "/mine-page-js.txt" ? root : publicDir, url.pathname));
    const safeBase = url.pathname === "/mine-page-js.txt" ? root : publicDir;
    if (!file.startsWith(safeBase) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "content-type": contentType(file), "cache-control": "no-store" });
    fs.createReadStream(file).pipe(res);
  } catch (error) {
    json(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
  });
}

export function startServer(port = PORT) {
  const server = createServer();
  server.listen(port, "127.0.0.1", () => {
    console.log(`hash256 gpu miner: http://127.0.0.1:${port}`);
    console.log(`contract: ${CONTRACT}`);
    console.log(`tx rpc: ${TX_RPC}`);
    console.log(`broadcast: ${BROADCAST_RPCS.join(", ")}`);
    console.log(`tip: ${TIP_GWEI} gwei`);
  });
  return server;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  startServer(PORT);
}
