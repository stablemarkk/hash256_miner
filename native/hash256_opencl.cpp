#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

using cl_int = int;
using cl_uint = unsigned int;
using cl_ulong = unsigned long long;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;
using cl_event = struct _cl_event*;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_ONLY (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROGRAM_BUILD_LOG 0x1183

using p_clGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using p_clGetDeviceIDs = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using p_clCreateContext = cl_context (*)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
using p_clCreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id, cl_bitfield, cl_int*);
using p_clCreateProgramWithSource = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using p_clBuildProgram = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
using p_clGetProgramBuildInfo = cl_int (*)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
using p_clCreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
using p_clCreateBuffer = cl_mem (*)(cl_context, cl_bitfield, size_t, void*, cl_int*);
using p_clSetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using p_clEnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
using p_clEnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
using p_clEnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
using p_clFinish = cl_int (*)(cl_command_queue);

struct CL {
  HMODULE lib{};
  p_clGetPlatformIDs clGetPlatformIDs{};
  p_clGetDeviceIDs clGetDeviceIDs{};
  p_clCreateContext clCreateContext{};
  p_clCreateCommandQueue clCreateCommandQueue{};
  p_clCreateProgramWithSource clCreateProgramWithSource{};
  p_clBuildProgram clBuildProgram{};
  p_clGetProgramBuildInfo clGetProgramBuildInfo{};
  p_clCreateKernel clCreateKernel{};
  p_clCreateBuffer clCreateBuffer{};
  p_clSetKernelArg clSetKernelArg{};
  p_clEnqueueWriteBuffer clEnqueueWriteBuffer{};
  p_clEnqueueReadBuffer clEnqueueReadBuffer{};
  p_clEnqueueNDRangeKernel clEnqueueNDRangeKernel{};
  p_clFinish clFinish{};
};

template <typename T> void loadSym(HMODULE lib, T& out, const char* name) {
  out = reinterpret_cast<T>(GetProcAddress(lib, name));
  if (!out) {
    std::fprintf(stderr, "missing OpenCL symbol %s\n", name);
    std::exit(2);
  }
}

CL loadOpenCL() {
  CL cl;
  cl.lib = LoadLibraryA("OpenCL.dll");
  if (!cl.lib) {
    std::fprintf(stderr, "OpenCL.dll not found\n");
    std::exit(2);
  }
  loadSym(cl.lib, cl.clGetPlatformIDs, "clGetPlatformIDs");
  loadSym(cl.lib, cl.clGetDeviceIDs, "clGetDeviceIDs");
  loadSym(cl.lib, cl.clCreateContext, "clCreateContext");
  loadSym(cl.lib, cl.clCreateCommandQueue, "clCreateCommandQueue");
  loadSym(cl.lib, cl.clCreateProgramWithSource, "clCreateProgramWithSource");
  loadSym(cl.lib, cl.clBuildProgram, "clBuildProgram");
  loadSym(cl.lib, cl.clGetProgramBuildInfo, "clGetProgramBuildInfo");
  loadSym(cl.lib, cl.clCreateKernel, "clCreateKernel");
  loadSym(cl.lib, cl.clCreateBuffer, "clCreateBuffer");
  loadSym(cl.lib, cl.clSetKernelArg, "clSetKernelArg");
  loadSym(cl.lib, cl.clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
  loadSym(cl.lib, cl.clEnqueueReadBuffer, "clEnqueueReadBuffer");
  loadSym(cl.lib, cl.clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
  loadSym(cl.lib, cl.clFinish, "clFinish");
  return cl;
}

const char* KERNEL = R"CLC(
typedef unsigned int uint;
typedef unsigned long ulong;

inline ulong rol64(ulong x, uint n) { return (x << n) | (x >> (64u - n)); }
inline uint bswap32(uint v) {
  return ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
         ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
}

__constant ulong RC[24] = {
  0x0000000000000001UL,0x0000000000008082UL,0x800000000000808aUL,0x8000000080008000UL,
  0x000000000000808bUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
  0x000000000000008aUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000aUL,
  0x000000008000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,
  0x8000000000008002UL,0x8000000000000080UL,0x000000000000800aUL,0x800000008000000aUL,
  0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL
};

inline void keccak_f(ulong s[25]) {
  for (uint r = 0; r < 24; r++) {
    ulong C[5], D[5], B[25];
    for (uint x = 0; x < 5; x++) C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
    for (uint x = 0; x < 5; x++) D[x] = C[(x+4)%5] ^ rol64(C[(x+1)%5], 1);
    for (uint x = 0; x < 5; x++) for (uint y = 0; y < 5; y++) s[x+5*y] ^= D[x];
    B[0]=s[0]; B[10]=rol64(s[1],1); B[20]=rol64(s[2],62); B[5]=rol64(s[3],28); B[15]=rol64(s[4],27);
    B[16]=rol64(s[5],36); B[1]=rol64(s[6],44); B[11]=rol64(s[7],6); B[21]=rol64(s[8],55); B[6]=rol64(s[9],20);
    B[7]=rol64(s[10],3); B[17]=rol64(s[11],10); B[2]=rol64(s[12],43); B[12]=rol64(s[13],25); B[22]=rol64(s[14],39);
    B[23]=rol64(s[15],41); B[8]=rol64(s[16],45); B[18]=rol64(s[17],15); B[3]=rol64(s[18],21); B[13]=rol64(s[19],8);
    B[14]=rol64(s[20],18); B[24]=rol64(s[21],2); B[9]=rol64(s[22],61); B[19]=rol64(s[23],56); B[4]=rol64(s[24],14);
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) s[x+5*y] = B[x+5*y] ^ ((~B[((x+1)%5)+5*y]) & B[((x+2)%5)+5*y]);
    s[0] ^= RC[r];
  }
}

__kernel void mine_kernel(__constant uint* ch, __constant uint* diff, ulong base, __global uint* out) {
  ulong gid = get_global_id(0);
  for (uint k = 0; k < 64; k++) {
    ulong nonce = base + gid * 64UL + (ulong)k;
    ulong st[25];
    for (uint i = 0; i < 25; i++) st[i] = 0UL;
    st[0] = ((ulong)ch[1] << 32) | ch[0];
    st[1] = ((ulong)ch[3] << 32) | ch[2];
    st[2] = ((ulong)ch[5] << 32) | ch[4];
    st[3] = ((ulong)ch[7] << 32) | ch[6];
    uint nlo = (uint)(nonce & 0xffffffffUL);
    uint nhi = (uint)(nonce >> 32);
    st[7] = ((ulong)bswap32(nlo) << 32) | bswap32(nhi);
    st[8] = 0x0000000000000001UL;
    st[16] = 0x8000000000000000UL;
    keccak_f(st);
    uint h[8] = {
      bswap32((uint)st[0]), bswap32((uint)(st[0] >> 32)),
      bswap32((uint)st[1]), bswap32((uint)(st[1] >> 32)),
      bswap32((uint)st[2]), bswap32((uint)(st[2] >> 32)),
      bswap32((uint)st[3]), bswap32((uint)(st[3] >> 32))
    };
    int lt = 0, done = 0;
    for (uint i = 0; i < 8; i++) {
      if (!done && h[i] < diff[i]) { lt = 1; done = 1; }
      else if (!done && h[i] > diff[i]) { done = 1; }
    }
    if (lt && atomic_cmpxchg((volatile __global unsigned int*)&out[0], 0u, 1u) == 0u) {
      out[1] = nlo; out[2] = nhi;
      for (uint i = 0; i < 8; i++) out[4+i] = h[i];
    }
  }
}
)CLC";

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
  for (size_t i = 0; i < 32; i++) out[i] = (hexNibble(s[2*i]) << 4) | hexNibble(s[2*i+1]);
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
    std::fprintf(stderr, "usage: hash256-opencl.exe <challenge32> <difficulty32> [globalWorkItems]\n");
    return 2;
  }
  uint32_t ch[8], diff[8];
  toChallengeWords(parse32(argv[1]), ch);
  toDifficultyWords(parse32(argv[2]), diff);
  size_t global = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : (1u << 22);

  CL cl = loadOpenCL();
  cl_uint platformCount = 0;
  cl.clGetPlatformIDs(0, nullptr, &platformCount);
  std::vector<cl_platform_id> platforms(platformCount);
  cl.clGetPlatformIDs(platformCount, platforms.data(), nullptr);
  cl_device_id dev = nullptr;
  for (auto p : platforms) {
    if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) == CL_SUCCESS && dev) break;
  }
  if (!dev) { std::fprintf(stderr, "no OpenCL GPU device\n"); return 2; }
  cl_int err = 0;
  cl_context ctx = cl.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue q = cl.clCreateCommandQueue(ctx, dev, 0, &err);
  const char* src = KERNEL;
  size_t len = std::strlen(KERNEL);
  cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &src, &len, &err);
  err = cl.clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::string log(logSize, '\0');
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    std::fprintf(stderr, "%s\n", log.c_str());
    return 2;
  }
  cl_kernel kernel = cl.clCreateKernel(prog, "mine_kernel", &err);
  cl_mem chBuf = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(ch), ch, &err);
  cl_mem diffBuf = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(diff), diff, &err);
  uint32_t out[12] = {};
  cl_mem outBuf = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(out), out, &err);
  cl.clSetKernelArg(kernel, 0, sizeof(chBuf), &chBuf);
  cl.clSetKernelArg(kernel, 1, sizeof(diffBuf), &diffBuf);
  cl.clSetKernelArg(kernel, 3, sizeof(outBuf), &outBuf);

  std::mt19937_64 rng(std::random_device{}());
  uint64_t base = rng();
  uint64_t total = 0;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    std::memset(out, 0, sizeof(out));
    cl.clEnqueueWriteBuffer(q, outBuf, 1, 0, sizeof(out), out, 0, nullptr, nullptr);
    cl.clSetKernelArg(kernel, 2, sizeof(base), &base);
    cl.clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    cl.clEnqueueReadBuffer(q, outBuf, 1, 0, sizeof(out), out, 0, nullptr, nullptr);
    total += uint64_t(global) * 64ull;
    if (out[0]) {
      std::printf("FOUND 0x%048llx%08x%08x 0x", 0ull, out[2], out[1]);
      for (int i = 0; i < 8; i++) std::printf("%08x", out[4+i]);
      std::printf("\n");
      std::fflush(stdout);
      return 0;
    }
    base += uint64_t(global) * 64ull;
    auto now = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(now - start).count();
    if (sec > 1.0) {
      std::printf("PROGRESS %.2f MH/s total=%llu\n", double(total) / sec / 1e6, (unsigned long long)total);
      std::fflush(stdout);
      start = now;
      total = 0;
    }
  }
}
