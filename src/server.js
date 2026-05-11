import crypto from "node:crypto";
import { ethers } from "ethers";
import { loadDotEnv } from "./env.js";

loadDotEnv();

const CONTRACT = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";
const READ_RPC = process.env.ETH_RPC_URL || "https://ethereum.publicnode.com";
const TX_RPC = process.env.ETH_TX_RPC_URL || "https://rpc.mevblocker.io/fast";
const BROADCAST_RPCS = (process.env.BROADCAST_RPCS ||
  "https://rpc.mevblocker.io/fast,https://rpc.flashbots.net/fast")
  .split(",")
  .map(item => item.trim())
  .filter(Boolean);
const TIP_GWEI = process.env.TIP_GWEI || "25";

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
