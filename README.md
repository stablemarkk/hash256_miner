# DEV: https://x.com/stablemark_

# HASH256 CLI GPU Miner

HASH256 CLI GPU Miner is a high-performance native miner for HASH256 proof-of-work. It is designed to be more efficient than mining through the HASH256 website by using CUDA/OpenCL GPU kernels, and it automatically submits the mint transaction as soon as a valid nonce is found.

## What This Project Does

- Reads mining state from an Ethereum RPC endpoint.
- Runs a native GPU miner locally.
- Submits a found nonce through your own wallet.
- Keeps your private key in local Node.js code only, not in the browser miner.

## Requirements

- Windows 10/11 x64.
- NVIDIA RTX 20 series or newer for CUDA mining.
- Node.js 20 or newer.
- ETH on the mining wallet for gas.
- An Ethereum RPC URL. Public RPCs can work, but a private RPC is usually more reliable.

## GPU Support

The included CUDA binary is built for RTX 20+ class NVIDIA GPUs:

```text
RTX 20 series: sm_75
RTX 30 series: sm_86
RTX 40 series: sm_89
Newer NVIDIA GPUs: sm_120 plus PTX fallback
```

Older NVIDIA cards are not the target for the CUDA build. If CUDA does not work on an old card, try the OpenCL backend instead:

```powershell
npm run native:opencl
```

## Quick Start

Install dependencies:

```powershell
npm install
```

Create your local environment file:

```powershell
copy .env.example .env
notepad .env
```

Edit `.env` and set at least:

```text
PRIVATE_KEY=0xYOUR_PRIVATE_KEY_WITH_ETH_FOR_GAS
ETH_RPC_URL=https://your-read-rpc.example
ETH_TX_RPC_URL=https://your-tx-rpc.example
MINER_BACKEND=cuda
```

Never commit `.env`. It is ignored by git.

Run the CUDA miner:

```powershell
npm run native:cuda
```

Run the OpenCL miner:

```powershell
npm run native:opencl
```

Run the local browser/WebGPU UI:

```powershell
npm start
```

Then open:

```text
http://127.0.0.1:8787
```

## CUDA Test And Benchmark

Check correctness without network access:

```powershell
npm run cuda:test
```

Run a 10 second CUDA benchmark without network access:

```powershell
npm run cuda:bench
```

Direct binary usage:

```powershell
native\bin\hash256-cuda.exe --test
native\bin\hash256-cuda.exe --bench 10 6144 384 96
native\bin\hash256-cuda.exe <challenge32> <difficulty32> [blocks] [threads] [hashesPerThread]
```

The current CUDA defaults are:

```text
CUDA_BLOCKS=0
CUDA_THREADS=384
CUDA_HASHES_PER_THREAD=96
```

`CUDA_BLOCKS=0` lets the miner choose an SM-based block count automatically.

## Environment Variables

```text
PRIVATE_KEY              Wallet private key used to submit mine transactions.
TIP_GWEI                 Priority fee for transactions.
ETH_RPC_URL              RPC used to read contract state.
ETH_TX_RPC_URL           Main RPC used for nonce and gas data.
BROADCAST_RPCS           Comma-separated RPC list for transaction broadcast.
MINER_BACKEND            cuda or opencl.
MINER_WORKERS            Number of native miner processes. Use 1 for one GPU.
CUDA_BLOCKS              CUDA block count. 0 means auto.
CUDA_THREADS             CUDA threads per block.
CUDA_HASHES_PER_THREAD   Nonces scanned by each CUDA thread per launch.
OPENCL_GLOBAL            OpenCL global work items.
PORT                     Local web server port.
```

## Build CUDA Binary From Source

The repository includes a ready-to-run Windows binary in `native/bin/hash256-cuda.exe`. It is built as a fat binary for RTX 20+ GPUs. To rebuild it, install CUDA Toolkit and Visual Studio Build Tools, then run:

```powershell
cmd /c """C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && ""C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"" -O3 -Xptxas=-v,-O3 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120 native\hash256_cuda.cu -o native\bin\hash256-cuda.exe"
```

## Project Layout

```text
src/                         Node.js server, CLI, and transaction submission code.
src/public/                  Browser WebGPU UI.
native/hash256_cuda.cu       CUDA miner source.
native/hash256_opencl.cpp    OpenCL miner source.
native/bin/                  Ready-to-run native Windows binaries.
PERFORMANCE_AUDIT.md         CUDA performance audit notes and benchmark data.
.env.example                 Safe environment template.
```

## Safety Notes

- Do not upload `.env`.
- Do not paste your private key into issues, logs, screenshots, or chats.
- Test the CUDA binary with `npm run cuda:test` before mining.
- Use one miner worker per GPU unless you know exactly why you need more.
