#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  const char* s = cudaGetErrorString(e); \
  std::fprintf(stderr, "CUDA error %s:%d code=%d: %s\n", __FILE__, __LINE__, (int)e, s ? s : "(null)"); std::exit(2); } } while (0)

#ifndef HASHES_PER_THREAD
#define HASHES_PER_THREAD 64
#endif

static constexpr uint32_t kDefaultHashesPerThread = 96;
static constexpr int kMessageLen = 64;
static constexpr int kNonceOffset = 56;
static constexpr int kNonceBytes = 8;
static constexpr int kPermutationsPerNonce = 1;

__constant__ uint32_t c_ch[8];
__constant__ uint32_t c_diff[8];
__constant__ uint64_t RC[24] = {
  0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
  0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
  0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
  0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
  0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
  0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t rol64(uint64_t x, uint32_t n) {
  return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ uint32_t bswap32d(uint32_t v) {
  return __byte_perm(v, 0, 0x0123);
}

__device__ __forceinline__ void keccak_f(uint64_t s[25]) {
#pragma unroll 24
  for (uint32_t r = 0; r < 24; r++) {
    uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
    uint64_t D0 = C4 ^ rol64(C1, 1), D1 = C0 ^ rol64(C2, 1), D2 = C1 ^ rol64(C3, 1), D3 = C2 ^ rol64(C4, 1), D4 = C3 ^ rol64(C0, 1);
    uint64_t b00=s[0]^D0, b10=rol64(s[1]^D1,1), b20=rol64(s[2]^D2,62), b05=rol64(s[3]^D3,28), b15=rol64(s[4]^D4,27);
    uint64_t b16=rol64(s[5]^D0,36), b01=rol64(s[6]^D1,44), b11=rol64(s[7]^D2,6), b21=rol64(s[8]^D3,55), b06=rol64(s[9]^D4,20);
    uint64_t b07=rol64(s[10]^D0,3), b17=rol64(s[11]^D1,10), b02=rol64(s[12]^D2,43), b12=rol64(s[13]^D3,25), b22=rol64(s[14]^D4,39);
    uint64_t b23=rol64(s[15]^D0,41), b08=rol64(s[16]^D1,45), b18=rol64(s[17]^D2,15), b03=rol64(s[18]^D3,21), b13=rol64(s[19]^D4,8);
    uint64_t b14=rol64(s[20]^D0,18), b24=rol64(s[21]^D1,2), b09=rol64(s[22]^D2,61), b19=rol64(s[23]^D3,56), b04=rol64(s[24]^D4,14);
    s[0]=b00^((~b01)&b02); s[1]=b01^((~b02)&b03); s[2]=b02^((~b03)&b04); s[3]=b03^((~b04)&b00); s[4]=b04^((~b00)&b01);
    s[5]=b05^((~b06)&b07); s[6]=b06^((~b07)&b08); s[7]=b07^((~b08)&b09); s[8]=b08^((~b09)&b05); s[9]=b09^((~b05)&b06);
    s[10]=b10^((~b11)&b12); s[11]=b11^((~b12)&b13); s[12]=b12^((~b13)&b14); s[13]=b13^((~b14)&b10); s[14]=b14^((~b10)&b11);
    s[15]=b15^((~b16)&b17); s[16]=b16^((~b17)&b18); s[17]=b17^((~b18)&b19); s[18]=b18^((~b19)&b15); s[19]=b19^((~b15)&b16);
    s[20]=b20^((~b21)&b22); s[21]=b21^((~b22)&b23); s[22]=b22^((~b23)&b24); s[23]=b23^((~b24)&b20); s[24]=b24^((~b20)&b21);
    s[0] ^= RC[r];
  }
}

__device__ __forceinline__ void store_result(const uint64_t st[25], uint32_t nlo, uint32_t nhi, uint32_t* out) {
  out[1] = nlo;
  out[2] = nhi;
  out[4] = bswap32d((uint32_t)st[0]);
  out[5] = bswap32d((uint32_t)(st[0] >> 32));
  out[6] = bswap32d((uint32_t)st[1]);
  out[7] = bswap32d((uint32_t)(st[1] >> 32));
  out[8] = bswap32d((uint32_t)st[2]);
  out[9] = bswap32d((uint32_t)(st[2] >> 32));
  out[10] = bswap32d((uint32_t)st[3]);
  out[11] = bswap32d((uint32_t)(st[3] >> 32));
}

__device__ __forceinline__ bool digest_less_than_target(const uint64_t st[25]) {
  uint32_t h = bswap32d((uint32_t)st[0]);
  if (h != c_diff[0]) return h < c_diff[0];
  h = bswap32d((uint32_t)(st[0] >> 32));
  if (h != c_diff[1]) return h < c_diff[1];
  h = bswap32d((uint32_t)st[1]);
  if (h != c_diff[2]) return h < c_diff[2];
  h = bswap32d((uint32_t)(st[1] >> 32));
  if (h != c_diff[3]) return h < c_diff[3];
  h = bswap32d((uint32_t)st[2]);
  if (h != c_diff[4]) return h < c_diff[4];
  h = bswap32d((uint32_t)(st[2] >> 32));
  if (h != c_diff[5]) return h < c_diff[5];
  h = bswap32d((uint32_t)st[3]);
  if (h != c_diff[6]) return h < c_diff[6];
  h = bswap32d((uint32_t)(st[3] >> 32));
  return h < c_diff[7];
}

__device__ __forceinline__ void mine_one_nonce(uint64_t nonce, uint32_t* out) {
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; i++) st[i] = 0;

  st[0] = ((uint64_t)c_ch[1] << 32) | c_ch[0];
  st[1] = ((uint64_t)c_ch[3] << 32) | c_ch[2];
  st[2] = ((uint64_t)c_ch[5] << 32) | c_ch[4];
  st[3] = ((uint64_t)c_ch[7] << 32) | c_ch[6];
  uint32_t nlo = (uint32_t)nonce;
  uint32_t nhi = (uint32_t)(nonce >> 32);

  // Protocol message is challenge32 || zero24 || uint64_be(nonce), Keccak padding 0x01.
  st[7] = ((uint64_t)bswap32d(nlo) << 32) | bswap32d(nhi);
  st[8] = 1ULL;
  st[16] = 0x8000000000000000ULL;

  keccak_f(st);
  if (digest_less_than_target(st) && atomicCAS(&out[0], 0u, 1u) == 0u) {
    store_result(st, nlo, nhi, out);
  }
}

extern "C" __global__ void mine_kernel(uint64_t base, uint32_t* out) {
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t nonce = base + gid;
#pragma unroll HASHES_PER_THREAD
  for (uint32_t k = 0; k < HASHES_PER_THREAD; k++) {
    mine_one_nonce(nonce, out);
    nonce += stride;
  }
}

extern "C" __global__ void mine_kernel_iters(uint64_t base, uint32_t iterations, uint32_t* out) {
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t nonce = base + gid;
  for (uint32_t k = 0; k < iterations; k++) {
    mine_one_nonce(nonce, out);
    nonce += stride;
  }
}

static const uint64_t kCpuRC[24] = {
  0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
  0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
  0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
  0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
  0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
  0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
};

static inline uint64_t rol64h(uint64_t x, uint32_t n) {
  return (x << n) | (x >> (64u - n));
}

void keccak_f_cpu(uint64_t s[25]) {
  for (uint32_t r = 0; r < 24; r++) {
    uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
    uint64_t D0 = C4 ^ rol64h(C1, 1), D1 = C0 ^ rol64h(C2, 1), D2 = C1 ^ rol64h(C3, 1), D3 = C2 ^ rol64h(C4, 1), D4 = C3 ^ rol64h(C0, 1);
    uint64_t b00=s[0]^D0, b10=rol64h(s[1]^D1,1), b20=rol64h(s[2]^D2,62), b05=rol64h(s[3]^D3,28), b15=rol64h(s[4]^D4,27);
    uint64_t b16=rol64h(s[5]^D0,36), b01=rol64h(s[6]^D1,44), b11=rol64h(s[7]^D2,6), b21=rol64h(s[8]^D3,55), b06=rol64h(s[9]^D4,20);
    uint64_t b07=rol64h(s[10]^D0,3), b17=rol64h(s[11]^D1,10), b02=rol64h(s[12]^D2,43), b12=rol64h(s[13]^D3,25), b22=rol64h(s[14]^D4,39);
    uint64_t b23=rol64h(s[15]^D0,41), b08=rol64h(s[16]^D1,45), b18=rol64h(s[17]^D2,15), b03=rol64h(s[18]^D3,21), b13=rol64h(s[19]^D4,8);
    uint64_t b14=rol64h(s[20]^D0,18), b24=rol64h(s[21]^D1,2), b09=rol64h(s[22]^D2,61), b19=rol64h(s[23]^D3,56), b04=rol64h(s[24]^D4,14);
    s[0]=b00^((~b01)&b02); s[1]=b01^((~b02)&b03); s[2]=b02^((~b03)&b04); s[3]=b03^((~b04)&b00); s[4]=b04^((~b00)&b01);
    s[5]=b05^((~b06)&b07); s[6]=b06^((~b07)&b08); s[7]=b07^((~b08)&b09); s[8]=b08^((~b09)&b05); s[9]=b09^((~b05)&b06);
    s[10]=b10^((~b11)&b12); s[11]=b11^((~b12)&b13); s[12]=b12^((~b13)&b14); s[13]=b13^((~b14)&b10); s[14]=b14^((~b10)&b11);
    s[15]=b15^((~b16)&b17); s[16]=b16^((~b17)&b18); s[17]=b17^((~b18)&b19); s[18]=b18^((~b19)&b15); s[19]=b19^((~b15)&b16);
    s[20]=b20^((~b21)&b22); s[21]=b21^((~b22)&b23); s[22]=b22^((~b23)&b24); s[23]=b23^((~b24)&b20); s[24]=b24^((~b20)&b21);
    s[0] ^= kCpuRC[r];
  }
}

std::array<uint8_t, 32> keccak256_cpu(const uint8_t* msg, size_t len, uint8_t suffix = 0x01) {
  constexpr size_t rate = 136;
  uint64_t st[25] = {};
  size_t offset = 0;
  while (len - offset >= rate) {
    for (size_t i = 0; i < rate / 8; i++) {
      uint64_t lane = 0;
      for (int b = 0; b < 8; b++) lane |= (uint64_t)msg[offset + i * 8 + b] << (8 * b);
      st[i] ^= lane;
    }
    keccak_f_cpu(st);
    offset += rate;
  }

  uint8_t block[rate] = {};
  size_t tail = len - offset;
  if (tail) std::memcpy(block, msg + offset, tail);
  block[tail] ^= suffix;
  block[rate - 1] ^= 0x80;
  for (size_t i = 0; i < rate / 8; i++) {
    uint64_t lane = 0;
    for (int b = 0; b < 8; b++) lane |= (uint64_t)block[i * 8 + b] << (8 * b);
    st[i] ^= lane;
  }
  keccak_f_cpu(st);

  std::array<uint8_t, 32> out{};
  for (int i = 0; i < 32; i++) out[i] = (uint8_t)(st[i / 8] >> (8 * (i & 7)));
  return out;
}

std::array<uint8_t, 32> protocol_hash_cpu(const uint8_t challenge[32], uint64_t nonce) {
  uint8_t msg[kMessageLen] = {};
  std::memcpy(msg, challenge, 32);
  for (int i = 0; i < 8; i++) msg[kNonceOffset + i] = (uint8_t)(nonce >> (56 - 8 * i));
  return keccak256_cpu(msg, sizeof(msg), 0x01);
}

int hexNibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::vector<uint8_t> parse32(const char* in) {
  std::string s(in);
  if (s.rfind("0x", 0) == 0) s = s.substr(2);
  if (s.size() != 64) { std::fprintf(stderr, "expected 32-byte hex\n"); std::exit(2); }
  std::vector<uint8_t> out(32);
  for (int i = 0; i < 32; i++) {
    int hi = hexNibble(s[2*i]);
    int lo = hexNibble(s[2*i+1]);
    if (hi < 0 || lo < 0) { std::fprintf(stderr, "invalid hex\n"); std::exit(2); }
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return out;
}

std::string hex32(const std::array<uint8_t, 32>& b) {
  static const char* h = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint8_t v : b) {
    out.push_back(h[v >> 4]);
    out.push_back(h[v & 15]);
  }
  return out;
}

void toChallengeWords(const std::vector<uint8_t>& b, uint32_t out[8]) {
  for (int i = 0; i < 8; i++) out[i] = uint32_t(b[4*i]) | (uint32_t(b[4*i+1]) << 8) | (uint32_t(b[4*i+2]) << 16) | (uint32_t(b[4*i+3]) << 24);
}

void toDifficultyWords(const std::vector<uint8_t>& b, uint32_t out[8]) {
  for (int i = 0; i < 8; i++) out[i] = (uint32_t(b[4*i]) << 24) | (uint32_t(b[4*i+1]) << 16) | (uint32_t(b[4*i+2]) << 8) | uint32_t(b[4*i+3]);
}

std::vector<uint8_t> plusOneTarget(const std::array<uint8_t, 32>& hash) {
  std::vector<uint8_t> target(hash.begin(), hash.end());
  for (int i = 31; i >= 0; i--) {
    target[i]++;
    if (target[i] != 0) return target;
  }
  std::fprintf(stderr, "test hash overflowed target\n");
  std::exit(2);
}

int defaultBlocksForDevice() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.multiProcessorCount > 0 ? prop.multiProcessorCount * 192 : 6144;
}

void printUsage() {
  std::fprintf(stderr,
    "usage: hash256-cuda.exe <challenge32> <difficulty32> [blocks] [threads] [hashesPerThread]\n"
    "       hash256-cuda.exe --bench [seconds] [blocks] [threads] [hashesPerThread]\n"
    "       hash256-cuda.exe --test\n");
}

void launchMine(int blocks, int threads, uint64_t base, uint32_t hashesPerThread, uint32_t* out) {
  if (hashesPerThread == HASHES_PER_THREAD) {
    mine_kernel<<<blocks, threads>>>(base, out);
  } else {
    mine_kernel_iters<<<blocks, threads>>>(base, hashesPerThread, out);
  }
  CUDA_CHECK(cudaGetLastError());
}

int runBenchmark(int argc, char** argv) {
  double seconds = argc >= 3 ? std::atof(argv[2]) : 10.0;
  int blocks = argc >= 4 ? std::atoi(argv[3]) : 0;
  int threads = argc >= 5 ? std::atoi(argv[4]) : 384;
  uint32_t hashesPerThread = argc >= 6 ? (uint32_t)std::strtoul(argv[5], nullptr, 10) : kDefaultHashesPerThread;
  if (seconds <= 0.0) seconds = 10.0;
  if (threads <= 0) threads = 384;
  if (hashesPerThread == 0) hashesPerThread = HASHES_PER_THREAD;

  CUDA_CHECK(cudaSetDevice(0));
  if (blocks <= 0) blocks = defaultBlocksForDevice();
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint32_t ch[8] = {};
  uint32_t diff[8] = {};
  CUDA_CHECK(cudaMemcpyToSymbol(c_ch, ch, sizeof(ch)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_diff, diff, sizeof(diff)));

  uint32_t* out = nullptr;
  uint32_t zero[12] = {};
  CUDA_CHECK(cudaMalloc(&out, sizeof(zero)));
  CUDA_CHECK(cudaMemcpy(out, zero, sizeof(zero), cudaMemcpyHostToDevice));

  cudaEvent_t evStart{}, evStop{};
  CUDA_CHECK(cudaEventCreate(&evStart));
  CUDA_CHECK(cudaEventCreate(&evStop));

  std::mt19937_64 rng(0x48415348323536ULL);
  uint64_t base = rng();
  uint64_t batch = (uint64_t)blocks * (uint64_t)threads * (uint64_t)hashesPerThread;
  uint64_t total = 0;
  auto wallStart = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(evStart));
  while (true) {
    launchMine(blocks, threads, base, hashesPerThread, out);
    CUDA_CHECK(cudaDeviceSynchronize());
    total += batch;
    base += batch;
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - wallStart).count() >= seconds) break;
  }
  CUDA_CHECK(cudaEventRecord(evStop));
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
  double elapsed = ms / 1000.0;
  double mhs = elapsed > 0.0 ? (double)total / elapsed / 1e6 : 0.0;

  std::printf("BENCH backend=CUDA device=\"%s\" sm=%d.%d\n", prop.name, prop.major, prop.minor);
  std::printf("BENCH total_hashes=%llu elapsed_seconds=%.6f MH/s=%.2f\n", (unsigned long long)total, elapsed, mhs);
  std::printf("BENCH config blocks=%d threads=%d hashes_per_thread=%u fixed_unroll=%s\n", blocks, threads, hashesPerThread, hashesPerThread == HASHES_PER_THREAD ? "yes" : "no");
  std::printf("BENCH message_length=%d nonce_offset=%d nonce_bytes=%d permutations_per_nonce=%d padding=Keccak-0x01\n", kMessageLen, kNonceOffset, kNonceBytes, kPermutationsPerNonce);
  std::printf("BENCH compiler_spills=runtime_unknown build_check=\"nvcc -O3 -arch=sm_89 -Xptxas=-v,-O3\"\n");
  std::fflush(stdout);

  CUDA_CHECK(cudaEventDestroy(evStart));
  CUDA_CHECK(cudaEventDestroy(evStop));
  CUDA_CHECK(cudaFree(out));
  return 0;
}

int runTests() {
  auto empty = keccak256_cpu(nullptr, 0, 0x01);
  if (hex32(empty) != "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470") {
    std::fprintf(stderr, "Keccak-256 empty vector failed: %s\n", hex32(empty).c_str());
    return 2;
  }
  const uint8_t abc[] = {'a','b','c'};
  auto abcHash = keccak256_cpu(abc, sizeof(abc), 0x01);
  if (hex32(abcHash) != "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45") {
    std::fprintf(stderr, "Keccak-256 abc vector failed: %s\n", hex32(abcHash).c_str());
    return 2;
  }

  for (size_t len : {size_t(135), size_t(136), size_t(271), size_t(272)}) {
    std::vector<uint8_t> msg(len);
    for (size_t i = 0; i < len; i++) msg[i] = (uint8_t)(i * 131u + 17u);
    auto a = keccak256_cpu(msg.data(), msg.size(), 0x01);
    auto b = keccak256_cpu(msg.data(), msg.size(), 0x01);
    if (a != b) {
      std::fprintf(stderr, "Keccak boundary deterministic test failed len=%zu\n", len);
      return 2;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  uint8_t chBytes[32] = {};
  for (int i = 0; i < 32; i++) chBytes[i] = (uint8_t)(i * 7 + 3);
  std::vector<uint8_t> chVec(chBytes, chBytes + 32);
  uint32_t ch[8], diff[8];
  toChallengeWords(chVec, ch);
  uint64_t nonce = 42;
  auto ref = protocol_hash_cpu(chBytes, nonce);
  auto target = plusOneTarget(ref);
  toDifficultyWords(target, diff);
  CUDA_CHECK(cudaMemcpyToSymbol(c_ch, ch, sizeof(ch)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_diff, diff, sizeof(diff)));

  uint32_t* out = nullptr;
  uint32_t host[12] = {};
  uint32_t zero[12] = {};
  CUDA_CHECK(cudaMalloc(&out, sizeof(host)));

  CUDA_CHECK(cudaMemcpy(out, zero, sizeof(host), cudaMemcpyHostToDevice));
  mine_kernel_iters<<<1, 1>>>(nonce, 1, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  if (!host[0] || host[1] != (uint32_t)nonce || host[2] != (uint32_t)(nonce >> 32)) {
    std::fprintf(stderr, "CUDA dynamic kernel nonce test failed\n");
    return 2;
  }
  for (int i = 0; i < 8; i++) {
    uint32_t got = host[4 + i];
    uint32_t want = (uint32_t(ref[4*i]) << 24) | (uint32_t(ref[4*i+1]) << 16) | (uint32_t(ref[4*i+2]) << 8) | ref[4*i+3];
    if (got != want) {
      std::fprintf(stderr, "CUDA dynamic kernel hash test failed word=%d got=%08x want=%08x\n", i, got, want);
      return 2;
    }
  }

  CUDA_CHECK(cudaMemcpy(out, zero, sizeof(host), cudaMemcpyHostToDevice));
  mine_kernel<<<1, 1>>>(nonce, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  if (!host[0] || host[1] != (uint32_t)nonce || host[2] != (uint32_t)(nonce >> 32)) {
    std::fprintf(stderr, "CUDA fixed kernel nonce test failed\n");
    return 2;
  }

  CUDA_CHECK(cudaFree(out));
  std::printf("TEST ok keccak_vectors=2 boundary_lengths=4 cuda_protocol=2 message_length=%d permutations_per_nonce=%d padding=Keccak-0x01\n", kMessageLen, kPermutationsPerNonce);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) return runBenchmark(argc, argv);
  if (argc >= 2 && std::strcmp(argv[1], "--test") == 0) return runTests();
  if (argc < 3) {
    printUsage();
    return 2;
  }

  uint32_t ch[8], diff[8];
  toChallengeWords(parse32(argv[1]), ch);
  toDifficultyWords(parse32(argv[2]), diff);
  int blocks = argc >= 4 ? std::atoi(argv[3]) : 0;
  int threads = argc >= 5 ? std::atoi(argv[4]) : 384;
  uint32_t hashesPerThread = argc >= 6 ? (uint32_t)std::strtoul(argv[5], nullptr, 10) : kDefaultHashesPerThread;
  if (threads <= 0) threads = 384;
  if (hashesPerThread == 0) hashesPerThread = HASHES_PER_THREAD;

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 1) {
    std::fprintf(stderr, "no CUDA devices\n");
    return 2;
  }
  CUDA_CHECK(cudaSetDevice(0));
  if (blocks <= 0) blocks = defaultBlocksForDevice();
  CUDA_CHECK(cudaMemcpyToSymbol(c_ch, ch, sizeof(ch)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_diff, diff, sizeof(diff)));

  uint32_t zero[12] = {};
  uint32_t host[12] = {};
  uint32_t* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, sizeof(host)));
  std::mt19937_64 rng(std::random_device{}());
  uint64_t base = rng();
  uint64_t batch = (uint64_t)blocks * (uint64_t)threads * (uint64_t)hashesPerThread;
  uint64_t total = 0;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    CUDA_CHECK(cudaMemcpy(out, zero, sizeof(host), cudaMemcpyHostToDevice));
    launchMine(blocks, threads, base, hashesPerThread, out);
    CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    total += batch;
    if (host[0]) {
      std::printf("FOUND 0x%048llx%08x%08x 0x", 0ULL, host[2], host[1]);
      for (int i = 0; i < 8; i++) std::printf("%08x", host[4+i]);
      std::printf("\n");
      std::fflush(stdout);
      return 0;
    }
    base += batch;
    auto now = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(now - start).count();
    if (sec >= 1.0) {
      std::printf("PROGRESS %.2f MH/s total=%llu\n", double(total) / sec / 1e6, (unsigned long long)total);
      std::fflush(stdout);
      total = 0;
      start = now;
    }
  }
}
