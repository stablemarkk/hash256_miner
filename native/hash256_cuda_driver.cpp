#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUdeviceptr = unsigned long long;
using CUresult = int;

#define CUDA_SUCCESS 0

struct Driver {
  HMODULE lib{};
  CUresult (*cuInit)(unsigned int){};
  CUresult (*cuDeviceGet)(CUdevice*, int){};
  CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice){};
  CUresult (*cuModuleLoad)(CUmodule*, const char*){};
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*){};
  CUresult (*cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, const char*){};
  CUresult (*cuMemAlloc)(CUdeviceptr*, size_t){};
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t){};
  CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t){};
  CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**){};
  CUresult (*cuCtxSynchronize)(){};
  CUresult (*cuGetErrorName)(CUresult, const char**){};
  CUresult (*cuGetErrorString)(CUresult, const char**){};
};

template <typename T> void sym(HMODULE lib, T& out, const char* name) {
  out = reinterpret_cast<T>(GetProcAddress(lib, name));
  if (!out) {
    std::fprintf(stderr, "missing nvcuda symbol %s\n", name);
    std::exit(2);
  }
}

Driver loadDriver() {
  Driver d;
  d.lib = LoadLibraryA("nvcuda.dll");
  if (!d.lib) {
    std::fprintf(stderr, "nvcuda.dll not found\n");
    std::exit(2);
  }
  sym(d.lib, d.cuInit, "cuInit");
  sym(d.lib, d.cuDeviceGet, "cuDeviceGet");
  sym(d.lib, d.cuCtxCreate, "cuCtxCreate_v2");
  sym(d.lib, d.cuModuleLoad, "cuModuleLoad");
  sym(d.lib, d.cuModuleGetFunction, "cuModuleGetFunction");
  sym(d.lib, d.cuModuleGetGlobal, "cuModuleGetGlobal_v2");
  sym(d.lib, d.cuMemAlloc, "cuMemAlloc_v2");
  sym(d.lib, d.cuMemcpyHtoD, "cuMemcpyHtoD_v2");
  sym(d.lib, d.cuMemcpyDtoH, "cuMemcpyDtoH_v2");
  sym(d.lib, d.cuLaunchKernel, "cuLaunchKernel");
  sym(d.lib, d.cuCtxSynchronize, "cuCtxSynchronize");
  sym(d.lib, d.cuGetErrorName, "cuGetErrorName");
  sym(d.lib, d.cuGetErrorString, "cuGetErrorString");
  return d;
}

void check(Driver& d, CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* msg = nullptr;
  d.cuGetErrorName(r, &name);
  d.cuGetErrorString(r, &msg);
  std::fprintf(stderr, "CUDA driver error at %s: %d %s %s\n", what, r, name ? name : "", msg ? msg : "");
  std::exit(2);
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
  if (s.size() != 64) {
    std::fprintf(stderr, "expected 32-byte hex\n");
    std::exit(2);
  }
  std::vector<uint8_t> out(32);
  for (int i = 0; i < 32; i++) out[i] = (hexNibble(s[2*i]) << 4) | hexNibble(s[2*i+1]);
  return out;
}

void toChallengeWords(const std::vector<uint8_t>& b, uint32_t out[8]) {
  for (int i = 0; i < 8; i++) out[i] = uint32_t(b[4*i]) | (uint32_t(b[4*i+1]) << 8) | (uint32_t(b[4*i+2]) << 16) | (uint32_t(b[4*i+3]) << 24);
}

void toDifficultyWords(const std::vector<uint8_t>& b, uint32_t out[8]) {
  for (int i = 0; i < 8; i++) out[i] = (uint32_t(b[4*i]) << 24) | (uint32_t(b[4*i+1]) << 16) | (uint32_t(b[4*i+2]) << 8) | uint32_t(b[4*i+3]);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: hash256-cuda-driver.exe <challenge32> <difficulty32> [blocks] [threads] [cubin]\n");
    return 2;
  }
  uint32_t ch[8], diff[8];
  toChallengeWords(parse32(argv[1]), ch);
  toDifficultyWords(parse32(argv[2]), diff);
  unsigned int blocks = argc >= 4 ? std::strtoul(argv[3], nullptr, 10) : 8192;
  unsigned int threads = argc >= 5 ? std::strtoul(argv[4], nullptr, 10) : 256;
  const char* cubin = argc >= 6 ? argv[5] : "native\\bin\\hash256-cuda.cubin";

  Driver d = loadDriver();
  check(d, d.cuInit(0), "cuInit");
  CUdevice dev;
  check(d, d.cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  check(d, d.cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
  CUmodule mod;
  check(d, d.cuModuleLoad(&mod, cubin), "cuModuleLoad");
  CUfunction fn;
  check(d, d.cuModuleGetFunction(&fn, mod, "mine_kernel"), "cuModuleGetFunction");

  CUdeviceptr chSym, diffSym, outDev;
  size_t symSize = 0;
  check(d, d.cuModuleGetGlobal(&chSym, &symSize, mod, "c_ch"), "cuModuleGetGlobal(c_ch)");
  check(d, d.cuMemcpyHtoD(chSym, ch, sizeof(ch)), "cuMemcpyHtoD(c_ch)");
  check(d, d.cuModuleGetGlobal(&diffSym, &symSize, mod, "c_diff"), "cuModuleGetGlobal(c_diff)");
  check(d, d.cuMemcpyHtoD(diffSym, diff, sizeof(diff)), "cuMemcpyHtoD(c_diff)");

  uint32_t zero[12] = {};
  uint32_t host[12] = {};
  check(d, d.cuMemAlloc(&outDev, sizeof(host)), "cuMemAlloc");

  std::mt19937_64 rng(std::random_device{}());
  uint64_t base = rng();
  uint64_t total = 0;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    std::memset(host, 0, sizeof(host));
    check(d, d.cuMemcpyHtoD(outDev, zero, sizeof(host)), "clear out");
    void* args[] = { &base, &outDev };
    check(d, d.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel");
    check(d, d.cuCtxSynchronize(), "cuCtxSynchronize");
    check(d, d.cuMemcpyDtoH(host, outDev, sizeof(host)), "read out");
    uint64_t batch = uint64_t(blocks) * uint64_t(threads) * 64ULL;
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
