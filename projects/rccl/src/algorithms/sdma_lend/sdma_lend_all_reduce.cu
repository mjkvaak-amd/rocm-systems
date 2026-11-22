/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SDMA mesh-lending AllReduce, two ranks, sum of fp32 / fp16 / bf16.
 *
 * On a full xGMI mesh a two-rank communicator talks over one of each GPU's
 * links. The links to the GPUs outside the communicator are idle. An SDMA
 * engine on such a GPU (a lender) can copy from one rank's memory into the
 * other rank's memory: the read crosses the lender's link to the source and
 * the write crosses its link to the destination. No compute unit moves a byte
 * across a link, nothing is staged on the lender and no kernel runs there.
 *
 * Each rank owns private SDMA queues: one on its own GPU and one on every
 * lender. Per call the host writes COPY_LINEAR + FENCE + FENCE into them. The
 * whole input goes to the peer's inbox in 1 + L equal pieces, one per path.
 * The first fence stamps an arrival flag at the receiver, the second a
 * departure flag at the sender. One kernel on the caller's stream waits for
 * the flags of the pieces it reads and adds, so everything queued behind it
 * is ordered after the transport. The peer's next call can start landing
 * while this rank still adds, so two inboxes alternate by call parity.
 *
 * Differences from the other AllReduce backends, all of which a reviewer
 * should weigh:
 *   - The host calls hipStreamSynchronize(stream) before it posts the copies,
 *     because the engines read sendbuff and are not ordered with the stream.
 *     The call therefore blocks until earlier work on the stream is done, and
 *     the path is not selected while the stream is being captured.
 *   - The queues are host-written KFD queues of this file, not Anvil queues.
 *     Anvil's device handles are written by kernels of the GPU that owns the
 *     queue; a lender's queue is on a GPU where this rank runs no kernel.
 *   - sendbuff has to be mapped on the lenders: hsa_amd_agents_allow_access
 *     is extended on the caller's allocation.
 *
 * The transport is the one of ROCm/aiter#8444 (csrc/kernels/mesh_lend_all_reduce.cu).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/sdma_lend/sdma_lend_all_reduce.h"

#include "algorithms/sdma_lend/kfd_sdma_queue.h"
#include "algorithms/sdma_lend/sdma_lend_policy.h"
#include "archinfo.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

RCCL_PARAM(SdmaLendEnable, "SDMA_LEND_ENABLE", 0);
RCCL_PARAM(SdmaLendMinBytes, "SDMA_LEND_MIN_BYTES", static_cast<int64_t>(kSdmaLendDefaultMinBytes));
// Also the size of each of a rank's two inboxes.
RCCL_PARAM(SdmaLendMaxBytes, "SDMA_LEND_MAX_BYTES", static_cast<int64_t>(kSdmaLendDefaultMaxBytes));

namespace {

namespace kfd = rccl_sdma_lend;

constexpr long kSpinLimit = 2000000000L;
constexpr size_t kPage = 4096;
constexpr int kThreads = 256;
constexpr int kMaxCandidates = 16;
constexpr uint32_t kMagic = 0x4c454e44; // "LEND"
constexpr uint32_t kVersion = 1;

enum ElemKind : int {
  kElemF32 = 0,
  kElemF16 = 1,
  kElemBf16 = 2
};

// ---- device ------------------------------------------------------------------

// A caller's buffer is only guaranteed the alignment of its element.
typedef uint64_t __attribute__((aligned(1), may_alias)) u64_any_t;

__device__ __forceinline__ float bf2f(uint16_t x) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  u += 0x7FFFu + ((u >> 16) & 1u); // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}

// Element-wise sum of the elements packed in two 64-bit words.
template <int KIND>
__device__ __forceinline__ uint64_t addWord(uint64_t a, uint64_t b) {
  uint64_t r = 0;
  if (KIND == kElemF32) {
    for (int j = 0; j < 2; ++j) {
      const int sh = 32 * j;
      const float s = __builtin_bit_cast(float, static_cast<uint32_t>(a >> sh)) +
                      __builtin_bit_cast(float, static_cast<uint32_t>(b >> sh));
      r |= static_cast<uint64_t>(__builtin_bit_cast(uint32_t, s)) << sh;
    }
  } else if (KIND == kElemF16) {
    for (int j = 0; j < 4; ++j) {
      const int sh = 16 * j;
      const _Float16 s = __builtin_bit_cast(_Float16, static_cast<uint16_t>(a >> sh)) +
                         __builtin_bit_cast(_Float16, static_cast<uint16_t>(b >> sh));
      r |= static_cast<uint64_t>(__builtin_bit_cast(uint16_t, s)) << sh;
    }
  } else {
    for (int j = 0; j < 4; ++j) {
      const int sh = 16 * j;
      r |= static_cast<uint64_t>(f2bf(bf2f(static_cast<uint16_t>(a >> sh)) + bf2f(static_cast<uint16_t>(b >> sh))))
           << sh;
    }
  }
  return r;
}

// Flags only grow, and a later fence in the same queue implies the earlier
// copy: compare with >=, not ==.
__device__ __forceinline__ bool waitFlag(const int* flag, int epoch) {
  long spins = 0;
  while (__atomic_load_n(flag, __ATOMIC_RELAXED) - epoch < 0) {
    if (++spins > kSpinLimit) return false;
  }
  return true;
}

// Waits for the pieces that overlap vectors [lo, hi).
__device__ __forceinline__ bool waitPieces(const int* flags, int dir, const SdmaLendSplit& s, size_t lo, size_t hi,
                                           int epoch) {
  for (int p = 0; p < s.nPaths; ++p) {
    if (s.bound[p + 1] <= lo || s.bound[p] >= hi) continue;
    if (!waitFlag(flags + sdmaLendFlagIndex(dir, p), epoch)) return false;
  }
  return true;
}

struct ReduceArgs {
  const char* inp;   // this rank's sendbuff
  const char* inbox; // the peer's sendbuff, as the engines delivered it
  char* out;         // recvbuff; may be inp
  const int* flags;
  int* err;
  SdmaLendSplit split; // in 16-byte vectors
  size_t bytes;
  int epoch;
};

template <int KIND>
__global__ void sdmaLendReducePairKernel(ReduceArgs a) {
  const size_t nvec = sdmaLendVecs(a.bytes);
  const size_t per = (nvec + gridDim.x - 1) / gridDim.x;
  size_t lo = static_cast<size_t>(blockIdx.x) * per;
  if (lo > nvec) lo = nvec;
  const size_t hi = lo + per < nvec ? lo + per : nvec;
  // Arrival: the peer's piece is in the inbox. Departure: the engines have
  // read this rank's piece, so writing out may overwrite inp (in place) and
  // the caller may reuse sendbuff once the kernel is done. Both ranks split
  // the same size over the same paths, so piece p covers the same vectors in
  // both directions.
  if (!waitPieces(a.flags, kSdmaLendArrival, a.split, lo, hi, a.epoch)) {
    a.err[0] = 1;
    return;
  }
  if (!waitPieces(a.flags, kSdmaLendDeparture, a.split, lo, hi, a.epoch)) {
    a.err[0] = 2;
    return;
  }
  for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const size_t off = kSdmaLendVecBytes * i;
    if (off + kSdmaLendVecBytes <= a.bytes) {
      const u64_any_t* x = reinterpret_cast<const u64_any_t*>(a.inp + off);
      const uint64_t* q = reinterpret_cast<const uint64_t*>(a.inbox + off);
      const uint64_t x0 = x[0], x1 = x[1];
      const uint64_t w0 = __atomic_load_n(q, __ATOMIC_RELAXED);
      const uint64_t w1 = __atomic_load_n(q + 1, __ATOMIC_RELAXED);
      u64_any_t* o = reinterpret_cast<u64_any_t*>(a.out + off);
      o[0] = addWord<KIND>(x0, w0);
      o[1] = addWord<KIND>(x1, w1);
    } else {
      // The last, partial vector: a whole number of elements below 16 bytes.
      uint64_t x[2] = {0, 0}, w[2] = {0, 0};
      const size_t rem = a.bytes - off;
      const volatile char* q = a.inbox + off;
      for (size_t k = 0; k < rem; ++k) {
        reinterpret_cast<char*>(x)[k] = a.inp[off + k];
        reinterpret_cast<char*>(w)[k] = q[k];
      }
      const uint64_t r[2] = {addWord<KIND>(x[0], w[0]), addWord<KIND>(x[1], w[1])};
      for (size_t k = 0; k < rem; ++k) a.out[off + k] = reinterpret_cast<const char*>(r)[k];
    }
  }
}

// ---- host --------------------------------------------------------------------

struct Export {
  uint32_t magic, version, ok, rank;
  uint64_t maxBytes;
  uint32_t gpuId, nCandidates;
  uint32_t directOnly;                 // RCCL_SDMA_LEND_LENDERS=none on this rank
  uint32_t candidates[kMaxCandidates]; // gpu_ids this rank can lend from
  struct {
    hsa_amd_ipc_memory_t handle;
    uint64_t len, off, addr;
  } buf[3]; // inbox 0, inbox 1, flags
};

struct Via {
  kfd::Device dev;
  int node = -1;
};

} // namespace

struct rcclSdmaLendState {
  int cudaDev = -1, rank = 0;
  size_t minBytes = 0, maxBytes = 0;
  kfd::GpuNode self{};
  std::vector<kfd::GpuNode> candidates;
  Export exported{};

  char* inbox[2] = {nullptr, nullptr};
  int* flags = nullptr;
  int* err = nullptr;

  bool peerSameProcess = false;
  char* peerInbox[2] = {nullptr, nullptr};
  int* peerFlags = nullptr;
  void* attached[3] = {nullptr, nullptr, nullptr};

  int nVia = 0;
  Via via[kSdmaLendMaxPaths]; // 0: own GPU, then the lenders
  kfd::Queue* route[kSdmaLendMaxPaths] = {};
  std::vector<std::unique_ptr<kfd::Queue>> queues;
  std::vector<hsa_agent_t> engineAgents; // agents whose engines touch our buffers
  std::vector<hsa_queue_t*> hsaQueues;   // keep ROCr's doorbell frame on a lender mapped

  bool hsaUp = false, connected = false, closed = false;
  uint32_t epoch = 0;
  uint64_t calls = 0, grants = 0;
};

namespace {

size_t pageUp(size_t n) {
  return (n + kPage - 1) & ~(kPage - 1);
}

bool pciOfHipDevice(int dev, uint32_t* domain, uint32_t* loc) {
  char bus[32] = {};
  unsigned d = 0, b = 0, s = 0, f = 0;
  if (hipDeviceGetPCIBusId(bus, sizeof(bus), dev) != hipSuccess || sscanf(bus, "%x:%x:%x.%x", &d, &b, &s, &f) != 4)
    return false;
  *domain = d;
  *loc = sdmaLendLocationId(b, s, f);
  return true;
}

struct AgentFind {
  uint32_t bdf, domain;
  hsa_agent_t out;
  bool found;
};

hsa_status_t agentCb(hsa_agent_t agent, void* data) {
  auto* f = static_cast<AgentFind*>(data);
  hsa_device_type_t type;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS || type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;
  uint32_t bdf = 0, domain = 0;
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdf);
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain);
  if (bdf == f->bdf && domain == f->domain) {
    f->out = agent;
    f->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// The ROCr agent of a GPU. A GPU hidden by HIP_VISIBLE_DEVICES still has one;
// a GPU hidden by ROCR_VISIBLE_DEVICES does not.
bool agentOf(const kfd::GpuNode& g, hsa_agent_t* out) {
  AgentFind f{g.location_id, g.domain, {}, false};
  hsa_iterate_agents(agentCb, &f);
  if (f.found) *out = f.out;
  return f.found;
}

const kfd::GpuNode* findGpu(const std::vector<kfd::GpuNode>& all, uint32_t gpuId) {
  for (const auto& g : all)
    if (g.gpu_id == gpuId) return &g;
  return nullptr;
}

// RCCL_SDMA_LEND_LENDERS: "auto" (default; every GPU ROCr shows this process
// except the rank's own), a comma list of PCI addresses, or "none". Without a
// lender the path is slower than the default AllReduce (one engine against the
// compute units of the GPU), so it is turned off; "none" on both ranks keeps
// it on over the direct link alone, for measurements.
bool parseLenders(const std::string& spec, const std::vector<kfd::GpuNode>& all, uint32_t selfId,
                  std::vector<kfd::GpuNode>* out, std::string* why) {
  hsa_agent_t agent;
  if (spec.empty() || spec == "auto") {
    for (const auto& g : all)
      if (g.gpu_id != selfId && agentOf(g, &agent)) out->push_back(g);
    return true;
  }
  if (spec == "none") return true;
  size_t pos = 0;
  while (pos <= spec.size()) {
    size_t end = spec.find(',', pos);
    if (end == std::string::npos) end = spec.size();
    const std::string tok = spec.substr(pos, end - pos);
    pos = end + 1;
    if (tok.empty()) continue;
    unsigned d = 0, b = 0, s = 0, f = 0;
    if (sscanf(tok.c_str(), "%x:%x:%x.%x", &d, &b, &s, &f) != 4) {
      *why = "lender '" + tok + "' is not a PCI address (dddd:bb:dd.f)";
      return false;
    }
    const uint32_t loc = sdmaLendLocationId(b, s, f);
    bool found = false;
    for (const auto& g : all) {
      if (g.domain != d || g.location_id != loc) continue;
      found = true;
      if (g.gpu_id == selfId) continue; // a list shared by both ranks names each rank's GPU for the other
      if (!agentOf(g, &agent)) {
        *why = "lender '" + tok + "' has no ROCr agent (ROCR_VISIBLE_DEVICES?)";
        return false;
      }
      out->push_back(g);
    }
    if (!found) {
      *why = "lender '" + tok + "' is not in the KFD topology of this process";
      return false;
    }
  }
  return true;
}

// hsa_amd_agents_allow_access replaces the access list, so a grant is a
// read-modify-write. Ranks that share a process grant on each other's buffers.
std::mutex& grantMutex() {
  static std::mutex m;
  return m;
}

// The allocation holding ptr must be mapped on every engine GPU: an engine
// that touches an unmapped address faults. Asked on every call and never
// remembered, because a freed block's address can come back as a new
// allocation.
bool ensureMapped(rcclSdmaLendState* st, const void* ptr, std::string* why) {
  std::lock_guard<std::mutex> lock(grantMutex());
  hsa_amd_pointer_info_t info;
  memset(&info, 0, sizeof(info));
  info.size = sizeof(info);
  uint32_t n = 0;
  hsa_agent_t* acc = nullptr;
  if (hsa_amd_pointer_info(ptr, &info, malloc, &n, &acc) != HSA_STATUS_SUCCESS ||
      info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
    free(acc);
    *why = "not a ROCr allocation";
    return false;
  }
  std::vector<hsa_agent_t> grant(acc, acc + n);
  free(acc);
  bool missing = false;
  for (hsa_agent_t a : st->engineAgents) {
    bool ok = a.handle == info.agentOwner.handle;
    for (uint32_t i = 0; i < n && !ok; ++i) ok = grant[i].handle == a.handle;
    if (!ok) {
      grant.push_back(a);
      missing = true;
    }
  }
  if (!missing) return true;
  const hsa_status_t s =
    hsa_amd_agents_allow_access(static_cast<uint32_t>(grant.size()), grant.data(), nullptr, info.agentBaseAddress);
  if (s != HSA_STATUS_SUCCESS) {
    *why = "cannot be mapped on every engine GPU (hsa status " + std::to_string(static_cast<int>(s)) + ")";
    return false;
  }
  ++st->grants;
  return true;
}

bool isRocrAllocation(const void* ptr) {
  hsa_amd_pointer_info_t info;
  memset(&info, 0, sizeof(info));
  info.size = sizeof(info);
  return hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS &&
         info.type != HSA_EXT_POINTER_TYPE_UNKNOWN;
}

// Buffers the peer's engines write: uncached, so a kernel reads what landed.
bool deviceAllocUncached(void** p, size_t bytes) {
  if (hipExtMallocWithFlags(p, bytes, hipDeviceMallocUncached) != hipSuccess) {
    *p = nullptr;
    return false;
  }
  return hipMemset(*p, 0, bytes) == hipSuccess;
}

bool exportBuffer(rcclSdmaLendState* st, void* ptr, int i) {
  hsa_amd_pointer_info_t info;
  memset(&info, 0, sizeof(info));
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
      info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
    return false;
  auto& b = st->exported.buf[i];
  if (hsa_amd_ipc_memory_create(info.agentBaseAddress, info.sizeInBytes, &b.handle) != HSA_STATUS_SUCCESS) return false;
  b.len = info.sizeInBytes;
  b.off = static_cast<uint64_t>(static_cast<char*>(ptr) - static_cast<char*>(info.agentBaseAddress));
  b.addr = reinterpret_cast<uint64_t>(ptr);
  return true;
}

bool commEligible(const ncclComm* comm) {
  const bool arch =
    comm->archName != nullptr && (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"));
  return comm->nRanks == kSdmaLendRanks && comm->nNodes == 1 && comm->bootstrap != nullptr && !comm->MNNVL && arch;
}

// This rank's own half: topology, inboxes, flag page, the KFD handle of its
// GPU and the handles the peer attaches. Runs with comm->cudaDev current.
bool localSetup(ncclComm* comm, rcclSdmaLendState* st, std::string* why) {
  st->cudaDev = comm->cudaDev;
  st->rank = comm->rank;
  st->minBytes = static_cast<size_t>(std::max<int64_t>(rcclParamSdmaLendMinBytes(), 0));
  st->maxBytes = pageUp(static_cast<size_t>(std::max<int64_t>(rcclParamSdmaLendMaxBytes(), kPage)));
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    *why = "hsa_init failed";
    return false;
  }
  st->hsaUp = true;

  const std::vector<kfd::GpuNode> all = kfd::gpu_nodes();
  uint32_t domain = 0, loc = 0;
  if (!pciOfHipDevice(st->cudaDev, &domain, &loc)) {
    *why = "no PCI address for HIP device " + std::to_string(st->cudaDev);
    return false;
  }
  bool found = false;
  for (const auto& g : all) {
    if (g.domain == domain && g.location_id == loc) {
      st->self = g;
      found = true;
    }
  }
  hsa_agent_t agent;
  if (!found || !agentOf(st->self, &agent)) {
    *why = "HIP device " + std::to_string(st->cudaDev) + " is not in the KFD topology";
    return false;
  }
  st->engineAgents.push_back(agent);
  const char* env = ncclGetEnv("RCCL_SDMA_LEND_LENDERS");
  const std::string lenders = env ? env : "auto";
  if (!parseLenders(lenders, all, st->self.gpu_id, &st->candidates, why)) return false;
  if (st->candidates.size() > static_cast<size_t>(kMaxCandidates)) st->candidates.resize(kMaxCandidates);

  for (int i = 0; i < 2; ++i) {
    if (!deviceAllocUncached(reinterpret_cast<void**>(&st->inbox[i]), st->maxBytes)) {
      *why = "cannot allocate a " + std::to_string(st->maxBytes) + "-byte inbox";
      return false;
    }
  }
  if (!deviceAllocUncached(reinterpret_cast<void**>(&st->flags), pageUp(kSdmaLendFlagBytes)) ||
      hipHostMalloc(reinterpret_cast<void**>(&st->err), kPage, 0) != hipSuccess) {
    *why = "cannot allocate the flag pages";
    return false;
  }
  memset(st->err, 0, kPage);
  if (hipDeviceSynchronize() != hipSuccess) {
    *why = "hipDeviceSynchronize failed";
    return false;
  }

  // HIP has run on this GPU by now, so its doorbell frame is mapped.
  *why = kfd::open_device(st->self.gpu_id, &st->via[0].dev);
  if (!why->empty()) return false;
  st->via[0].node = st->self.node;
  st->nVia = 1;

  Export& e = st->exported;
  e.magic = kMagic;
  e.version = kVersion;
  e.rank = static_cast<uint32_t>(st->rank);
  e.maxBytes = st->maxBytes;
  e.gpuId = st->self.gpu_id;
  e.directOnly = lenders == "none" ? 1 : 0;
  for (const auto& g : st->candidates) e.candidates[e.nCandidates++] = g.gpu_id;
  if (!exportBuffer(st, st->inbox[0], 0) || !exportBuffer(st, st->inbox[1], 1) || !exportBuffer(st, st->flags, 2)) {
    *why = "cannot export the inboxes (hsa_amd_ipc_memory_create)";
    return false;
  }
  e.ok = 1;
  return true;
}

// The peer's half: lenders, the peer's buffers and the queues.
bool connectPeer(ncclComm* comm, rcclSdmaLendState* st, const Export* ex, std::string* why) {
  const std::vector<kfd::GpuNode> all = kfd::gpu_nodes();
  const kfd::GpuNode* member[kSdmaLendRanks];
  for (int r = 0; r < kSdmaLendRanks; ++r) {
    if (ex[r].magic != kMagic || ex[r].version != kVersion || ex[r].rank != static_cast<uint32_t>(r) ||
        ex[r].maxBytes != st->maxBytes) {
      *why = "rank " + std::to_string(r) + " was set up with different parameters";
      return false;
    }
    member[r] = findGpu(all, ex[r].gpuId);
    if (!member[r]) {
      *why = "the GPU of rank " + std::to_string(r) + " is not visible to this process's KFD";
      return false;
    }
  }
  if (kfd::link_between(member[0]->node, member[1]->node).type != kfd::kIoLinkXgmi ||
      kfd::link_between(member[1]->node, member[0]->node).type != kfd::kIoLinkXgmi) {
    *why = "the two ranks have no direct xGMI link";
    return false;
  }

  // The lenders: GPUs both ranks can open, outside the communicator, with an
  // xGMI link to both ranks. Both ranks derive the same list in the same order.
  for (const auto& g : all) {
    bool ok = st->nVia < kSdmaLendMaxPaths;
    for (int r = 0; r < kSdmaLendRanks && ok; ++r) {
      ok = g.gpu_id != ex[r].gpuId &&
           std::find(ex[r].candidates, ex[r].candidates + ex[r].nCandidates, g.gpu_id) !=
             ex[r].candidates + ex[r].nCandidates &&
           kfd::link_between(g.node, member[r]->node).type == kfd::kIoLinkXgmi;
    }
    if (!ok) continue;
    hsa_agent_t agent;
    if (!agentOf(g, &agent)) {
      *why = "no ROCr agent for lender gpu_id " + std::to_string(g.gpu_id);
      return false;
    }
    // HIP may not drive this GPU in this process. KFD maps a GPU's doorbell
    // frame with the process's first queue there, so have ROCr create one.
    uint64_t base = 0, size = 0;
    if (!kfd::find_doorbell_frame(g.gpu_id, &base, &size)) {
      hsa_queue_t* q = nullptr;
      if (hsa_queue_create(agent, 4096, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q) !=
          HSA_STATUS_SUCCESS) {
        *why = "hsa_queue_create on lender gpu_id " + std::to_string(g.gpu_id);
        return false;
      }
      st->hsaQueues.push_back(q);
    }
    Via& v = st->via[st->nVia];
    *why = kfd::open_device(g.gpu_id, &v.dev);
    if (!why->empty()) return false;
    v.node = g.node;
    ++st->nVia;
    st->engineAgents.push_back(agent);
  }
  if (st->nVia == 1 && !(ex[0].directOnly && ex[1].directOnly)) {
    *why = "no GPU outside the communicator can lend: a lender needs an xGMI link to both ranks and a ROCr agent in "
           "both processes";
    return false;
  }

  const int peer = 1 - st->rank;
  st->peerSameProcess = comm->peerInfo[peer].pidHash == comm->peerInfo[st->rank].pidHash;
  void* ptrs[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    if (st->peerSameProcess) {
      // Ranks of one process share an address space: grant instead of attach.
      ptrs[i] = reinterpret_cast<void*>(ex[peer].buf[i].addr);
      if (!ensureMapped(st, ptrs[i], why)) {
        *why = "a buffer of rank " + std::to_string(peer) + " " + *why;
        return false;
      }
    } else {
      // Mapped on exactly the GPUs whose engines write it: ours and the lenders.
      hsa_amd_ipc_memory_t handle = ex[peer].buf[i].handle;
      if (hsa_amd_ipc_memory_attach(&handle, ex[peer].buf[i].len, static_cast<uint32_t>(st->engineAgents.size()),
                                    st->engineAgents.data(), &st->attached[i]) != HSA_STATUS_SUCCESS) {
        *why = "hsa_amd_ipc_memory_attach of rank " + std::to_string(peer);
        return false;
      }
      ptrs[i] = static_cast<char*>(st->attached[i]) + ex[peer].buf[i].off;
    }
  }
  st->peerInbox[0] = static_cast<char*>(ptrs[0]);
  st->peerInbox[1] = static_cast<char*>(ptrs[1]);
  st->peerFlags = static_cast<int*>(ptrs[2]);

  // Own GPU: the engine KFD recommends for the link to the peer. Lender: the
  // engine KFD recommends for the lender's link to this rank, so the two ranks
  // do not share an engine there.
  for (int v = 0; v < st->nVia; ++v) {
    const int far = v == 0 ? member[peer]->node : st->self.node;
    const int engine = sdmaLendPickEngine(kfd::link_between(st->via[v].node, far).engine_mask, st->rank);
    if (engine < 0) {
      *why = "KFD recommends no SDMA engine for a link of the communicator";
      return false;
    }
    auto fresh = std::make_unique<kfd::Queue>();
    *why = kfd::create_queue(st->via[v].dev, static_cast<uint32_t>(engine), fresh.get());
    st->queues.push_back(std::move(fresh));
    if (!why->empty()) return false;
    st->route[v] = st->queues.back().get();
  }

  // The departure fences of every path land in this rank's flag page.
  if (!ensureMapped(st, st->flags, why)) {
    *why = "the flag page " + *why;
    return false;
  }
  st->grants = 0;
  st->connected = true;
  return true;
}

// Queues first (the engines must be idle), then the peer's mappings.
void closePeer(rcclSdmaLendState* st) {
  if (st->closed) return;
  st->closed = true;
  st->connected = false;
  for (auto& q : st->queues) kfd::destroy_queue(q.get());
  st->queues.clear();
  for (void*& a : st->attached) {
    if (a) hsa_amd_ipc_memory_detach(a);
    a = nullptr;
  }
}

void destroyState(rcclSdmaLendState* st) {
  if (st == nullptr) return;
  closePeer(st);
  for (hsa_queue_t* q : st->hsaQueues) hsa_queue_destroy(q);
  for (int v = 0; v < kSdmaLendMaxPaths; ++v) kfd::close_device(&st->via[v].dev);
  for (void* p : {static_cast<void*>(st->inbox[0]), static_cast<void*>(st->inbox[1]), static_cast<void*>(st->flags)})
    if (p) (void)hipFree(p);
  if (st->err) (void)hipHostFree(st->err);
  if (st->hsaUp) hsa_shut_down();
  delete st;
}

bool elemKindOf(ncclDataType_t datatype, int* kind) {
  switch (datatype) {
  case ncclFloat32:
    *kind = kElemF32;
    return true;
  case ncclFloat16:
    *kind = kElemF16;
    return true;
  case ncclBfloat16:
    *kind = kElemBf16;
    return true;
  default:
    return false;
  }
}

int gridFor(size_t vecs) {
  const size_t want = vecs / 1024;
  return static_cast<int>(want < 64 ? 64 : want > 1024 ? 1024 : want);
}

template <int KIND>
void launchReduce(const ReduceArgs& a, int blocks, cudaStream_t stream, hipEvent_t stopEvent) {
  hipExtLaunchKernelGGL((sdmaLendReducePairKernel<KIND>), dim3(blocks), dim3(kThreads), 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, a);
}

} // namespace

ncclResult_t rcclSdmaLendCommInit(ncclComm* comm) {
  if (comm == nullptr || rcclParamSdmaLendEnable() == 0) return ncclSuccess;
  // Every input of this gate is the same on both ranks, so either both run the
  // two all-gathers below or neither does.
  if (!commEligible(comm)) {
    INFO(NCCL_INIT, "SDMA lend: not a two-rank single-node gfx942/gfx950 communicator, off");
    return ncclSuccess;
  }

  std::string why;
  auto* st = new (std::nothrow) rcclSdmaLendState();
  bool ok = st != nullptr && localSetup(comm, st, &why);

  Export all[kSdmaLendRanks];
  memset(all, 0, sizeof(all));
  if (ok) all[comm->rank] = st->exported;
  if (bootstrapAllGather(comm->bootstrap, all, static_cast<int>(sizeof(Export))) != ncclSuccess) {
    ok = false;
    why = "bootstrap all-gather of the exports failed";
  }
  if (ok && !(all[0].ok && all[1].ok)) {
    ok = false;
    why = "the peer could not set the path up";
  }
  if (ok) ok = connectPeer(comm, st, all, &why);

  // Rank-uniform outcome, else the ranks pick different backends and hang.
  int ready[kSdmaLendRanks] = {0, 0};
  ready[comm->rank] = ok ? 1 : 0;
  if (bootstrapAllGather(comm->bootstrap, ready, static_cast<int>(sizeof(int))) != ncclSuccess) {
    ready[0] = ready[1] = 0;
    if (ok) why = "bootstrap all-gather of the outcome failed";
  }
  if (!(ready[0] && ready[1])) {
    if (ok) why = "the peer could not connect";
    WARN("SDMA lend: off for this communicator: %s", why.c_str());
    destroyState(st);
    return ncclSuccess;
  }

  comm->sdmaLend = st;
  INFO(NCCL_INIT, "SDMA lend: rank %d gpu_id %u, %d lender(s), messages of %zu to %zu bytes", comm->rank,
       st->self.gpu_id, st->nVia - 1, st->minBytes, st->maxBytes);
  return ncclSuccess;
}

ncclResult_t rcclSdmaLendCommClosePeerMappings(ncclComm* comm) {
  if (comm == nullptr || comm->sdmaLend == nullptr) return ncclSuccess;
  // The reduce kernels run on the user stream and wait for every departure
  // flag, and doneEvent is recorded after every launch: once it has fired no
  // engine of this rank is reading or writing.
  if (comm->doneEvent != nullptr) {
    cudaError_t syncErr = cudaEventSynchronize(comm->doneEvent);
    if (syncErr != cudaSuccess) {
      WARN("rcclSdmaLendCommClosePeerMappings: doneEvent sync failed (%s); continuing cleanup",
           cudaGetErrorString(syncErr));
    }
  }
  closePeer(comm->sdmaLend);
  return ncclSuccess;
}

ncclResult_t rcclSdmaLendCommFini(ncclComm* comm) {
  if (comm == nullptr || comm->sdmaLend == nullptr) return ncclSuccess;
  ncclResult_t closeResult = ncclSuccess;
  NCCLCHECKIGNORE(rcclSdmaLendCommClosePeerMappings(comm), closeResult);
  destroyState(comm->sdmaLend);
  comm->sdmaLend = nullptr;
  return ncclSuccess;
}

int rcclSdmaLendNumPaths(ncclComm* comm) {
  return comm != nullptr && comm->sdmaLend != nullptr && comm->sdmaLend->connected ? comm->sdmaLend->nVia : 0;
}

bool rcclAllReduceSdmaLendEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->sdmaLend == nullptr || !comm->sdmaLend->connected) return false;
  if (op != ncclSum) return false;
  int kind = 0;
  if (!elemKindOf(datatype, &kind)) return false;
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));
  if (!sdmaLendSizeEligible(bytes, comm->sdmaLend->minBytes, comm->sdmaLend->maxBytes)) return false;
  if (!sdmaLendBuffersOk(sendbuff, recvbuff, bytes)) return false;
  // The engines read sendbuff by its address in this process's GPU VMs.
  return isRocrAllocation(sendbuff);
}

uint32_t rcclAllReduceSdmaLendBlocks(size_t count, ncclDataType_t datatype) {
  return static_cast<uint32_t>(gridFor(sdmaLendVecs(count * static_cast<size_t>(ncclTypeSize(datatype)))));
}

ncclResult_t rcclAllReduceSdmaLend(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  rcclSdmaLendState* st = comm->sdmaLend;
  int kind = 0;
  if (st == nullptr || !st->connected || !elemKindOf(datatype, &kind)) return ncclInvalidUsage;
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));

  // The engines read sendbuff: everything that produces it must have run.
  CUDACHECK(hipStreamSynchronize(stream));
  if (st->err[0]) {
    WARN("SDMA lend: a kernel timed out waiting for transport flag %d", st->err[0]);
    return ncclInternalError;
  }
  std::string why;
  if (!ensureMapped(st, sendbuff, &why)) {
    WARN("SDMA lend: sendbuff %s", why.c_str());
    return ncclInvalidUsage;
  }

  const uint32_t epoch = ++st->epoch;
  const int parity = static_cast<int>(st->calls++ & 1);
  ReduceArgs a = {};
  a.split = sdmaLendMakeSplit(sdmaLendVecs(bytes), st->nVia);
  const char* src = static_cast<const char*>(sendbuff);
  char* dst = st->peerInbox[parity];
  // Lenders first: their path is longer. An empty piece still gets its
  // fences, so the kernel may wait for every flag.
  for (int i = 1; i <= a.split.nPaths; ++i) {
    const int p = i % a.split.nPaths;
    size_t off = a.split.bound[p] * kSdmaLendVecBytes, end = a.split.bound[p + 1] * kSdmaLendVecBytes;
    end = end < bytes ? end : bytes;
    off = off < end ? off : end;
    kfd::Queue& q = *st->route[p];
    if (end > off) kfd::put_copy(q, src + off, dst + off, end - off);
    kfd::put_fence(q, st->peerFlags + sdmaLendFlagIndex(kSdmaLendArrival, p), epoch);
    kfd::put_fence(q, st->flags + sdmaLendFlagIndex(kSdmaLendDeparture, p), epoch);
  }
  for (auto& q : st->queues) kfd::ring_doorbell(*q);

  a.inp = src;
  a.inbox = st->inbox[parity];
  a.out = static_cast<char*>(recvbuff);
  a.flags = st->flags;
  a.err = st->err;
  a.bytes = bytes;
  a.epoch = static_cast<int>(epoch);
  const int blocks = gridFor(sdmaLendVecs(bytes));
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (kind) {
  case kElemF32:
    launchReduce<kElemF32>(a, blocks, stream, stopEvent);
    break;
  case kElemF16:
    launchReduce<kElemF16>(a, blocks, stream, stopEvent);
    break;
  default:
    launchReduce<kElemBf16>(a, blocks, stream, stopEvent);
    break;
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
