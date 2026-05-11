# HASH256 CUDA performance audit

## Pipeline

- Network/state path: `src/native-cli.js` fetches `challenge` and `difficulty`, then starts `native/bin/hash256-cuda.exe`.
- CUDA input: `native/hash256_cuda.cu` parses two 32-byte hex values once on process startup.
- Protocol message per nonce: `challenge32 || zero24 || uint64_be(nonce)`.
- Nonce position: bytes 56..63 in a 64-byte message.
- Hash: Keccak-256 padding suffix `0x01`, not SHA3 `0x06`.
- Cost per nonce: one Keccak-f[1600] permutation.
- Target compare: on GPU, big-endian 256-bit digest against difficulty.
- Hot loop CPU/GPU copies: only 48-byte result clear/read per kernel launch; no digest buffer per nonce.

## Main findings

- No sprintf/itoa/hex/decimal formatting was present in the CUDA kernel.
- No full message buffer is built per nonce in global memory.
- No digest is written to global memory for every nonce.
- Current CUDA path was already close to the hardware-bound single-permutation Keccak case; a multi-x gain is not realistic without a different algorithm/protocol or hand-tuned SASS.
- The useful gain found here is from larger per-launch nonce batches and a lower-register dynamic grid-stride kernel.

## Changes

- Added `--bench [seconds] [blocks] [threads] [hashesPerThread]`.
- Added `--test` with Keccak-256 vectors, boundary lengths 135/136/271/272, and CUDA-vs-CPU protocol hash checks.
- Added dynamic grid-stride kernel where each thread can scan a configurable number of nonces.
- Added early digest compare without materializing all eight hash words unless a result is found.
- Changed CUDA CLI defaults to `CUDA_BLOCKS=0`, `CUDA_THREADS=384`, `CUDA_HASHES_PER_THREAD=96`.

## Measured on RTX 4060 Ti

Sequential mining-loop A/B, zero target, config 6144 blocks x 384 threads:

- Baseline binary: avg last 8 samples 1166.70 MH/s, max 1262.60 MH/s.
- Optimized binary, 96 nonces/thread dynamic grid-stride: avg last 8 samples 1307.16 MH/s, max 1342.45 MH/s.
- Final 30s validation run: avg after first 4 samples 1259.26 MH/s, avg last 12 samples 1265.23 MH/s, max 1305.26 MH/s, at 150.53 W reported by nvidia-smi.

Build check:

```text
nvcc -O3 -arch=sm_89 -Xptxas=-v,-O3
mine_kernel:       111 registers, 0 spill stores, 0 spill loads
mine_kernel_iters: 72 registers, 0 spill stores, 0 spill loads
```

SASS inspection of `native/bin/hash256-cuda.sass` shows LOP3 and SHF are used heavily, with no LDL/STL local-memory spill instructions found by text scan.

## Remaining work

- Inspect generated SASS for chi/rotate instruction mix and LOP3 usage.
- Try a uint2/funnelshift Keccak layout only if it beats native uint64 on Ada.
- Power-normalized MH/W requires a controlled nvidia-smi sampling run with no competing GPU processes.
- A binary protocol/header format is already effectively used by the CUDA miner; changing external network/protocol behavior is not required.
