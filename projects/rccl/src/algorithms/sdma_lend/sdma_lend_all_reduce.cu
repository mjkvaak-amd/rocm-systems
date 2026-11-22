/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SDMA mesh-lending AllReduce: two ranks, sum of float32 / float16 / bfloat16.
 *
 * A two-rank communicator on a full xGMI mesh uses one link of each GPU; the
 * links to the GPUs outside it idle. An SDMA engine of such a GPU (a lender)
 * copies from one rank's memory into the other's over its own two links.
 * Nothing is staged on the lender and no kernel runs there.
 *
 * Each rank has a private SDMA queue on its own GPU and on every lender. Per
 * call the host cuts sendbuff into one piece per queue and writes copy + fence
 * + fence: the copy lands the piece in the peer's inbox, the first fence
 * stamps an arrival flag at the peer and the second a departure flag at the
 * sender. One kernel on the caller's stream waits for the flags of the pieces
 * it reads, then adds. The peer's next call can land while this rank still
 * adds, so two inboxes alternate.
 *
 * The engines are not ordered with the stream, so the host synchronizes the
 * stream before it posts: the call blocks, and it cannot be captured.
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
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

RCCL_PARAM(SdmaLendEnable, "SDMA_LEND_ENABLE", 0);
RCCL_PARAM(SdmaLendMinBytes, "SDMA_LEND_MIN_BYTES", 4LL << 20);
// Also the size of each of a rank's two inboxes.
RCCL_PARAM(SdmaLendMaxBytes, "SDMA_LEND_MAX_BYTES", 256LL << 20);
// 0 keeps the path on over the direct link alone, which is slower than the
// default AllReduce: for measurements.
RCCL_PARAM(SdmaLendMaxLenders, "SDMA_LEND_MAX_LENDERS", kSdmaLendMaxPaths - 1);

namespace {

namespace kfd = rccl_sdma_lend;

constexpr long kSpinLimit = 2000000000L;
// A rank's block: a flag page, then two inboxes. Flag p is the arrival of the
// peer's piece p, flag kSdmaLendMaxPaths + p the departure of this rank's.
constexpr size_t kFlagBytes = kfd::kPage;

// ---- device ------------------------------------------------------------------

// Flags only grow, and a later fence of a queue implies the earlier ones: >=.
__device__ __forceinline__ bool waitFlag(const int* flag, int epoch) {
  for (long spins = 0; __atomic_load_n(flag, __ATOMIC_RELAXED) - epoch < 0;) {
    if (++spins > kSpinLimit) return false;
  }
  return true;
}

// Element-wise sum of two vectors. Both are loaded whole before the store: out may be inp.
template <typename T>
__device__ __forceinline__ void addVec(char* out, const char* inp, const char* inbox) {
  T x[kSdmaLendVecBytes / sizeof(T)], w[kSdmaLendVecBytes / sizeof(T)];
  __builtin_memcpy(x, inp, kSdmaLendVecBytes);
  __builtin_memcpy(w, inbox, kSdmaLendVecBytes);
  for (size_t j = 0; j < kSdmaLendVecBytes / sizeof(T); ++j)
    x[j] = static_cast<T>(static_cast<float>(x[j]) + static_cast<float>(w[j]));
  __builtin_memcpy(out, x, kSdmaLendVecBytes);
}

struct ReduceArgs {
  const char* inp;   // sendbuff
  const char* inbox; // the peer's sendbuff, as the engines delivered it
  char* out;         // recvbuff; may be inp
  const int* flags;
  int* err;
  SdmaLendSplit split;
  size_t bytes;
  int epoch;
};

template <typename T>
__global__ void sdmaLendReduceKernel(ReduceArgs a) {
  const size_t nvec = sdmaLendVecs(a.bytes), per = (nvec + gridDim.x - 1) / gridDim.x;
  const size_t lo = std::min(blockIdx.x * per, nvec), hi = std::min(lo + per, nvec);
  // A block needs its pieces to have arrived, and to have departed before it
  // writes over them in place. Once every block is past this, sendbuff is free.
  for (int p = 0; p < a.split.nPaths; ++p) {
    if (a.split.bound[p + 1] <= lo || a.split.bound[p] >= hi) continue;
    if (!waitFlag(a.flags + p, a.epoch) || !waitFlag(a.flags + kSdmaLendMaxPaths + p, a.epoch)) {
      *a.err = 1;
      return;
    }
  }
  for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const size_t off = kSdmaLendVecBytes * i;
    if (off + kSdmaLendVecBytes <= a.bytes) {
      addVec<T>(a.out + off, a.inp + off, a.inbox + off);
    } else {
      // The last vector of a message can be short: add zero-padded copies.
      char x[kSdmaLendVecBytes] = {}, w[kSdmaLendVecBytes] = {};
      for (size_t k = 0; off + k < a.bytes; ++k) {
        x[k] = a.inp[off + k];
        w[k] = a.inbox[off + k];
      }
      addVec<T>(x, x, w);
      for (size_t k = 0; off + k < a.bytes; ++k) a.out[off + k] = x[k];
    }
  }
}

// ---- host --------------------------------------------------------------------

struct Gpu {
  hsa_agent_t agent;
  uint32_t gpuId; // KFD's
  int node;       // KFD topology node
};

// What a rank tells the other.
struct Export {
  uint32_t ok;
  int32_t node;
  uint64_t lendable; // nodes of the GPUs this rank can open, as a bit mask
  int64_t minBytes, maxBytes, maxLenders;
  hsa_amd_ipc_memory_t handle; // the rank's block
  uint64_t len, off, addr;
};

struct Path {
  Gpu gpu;
  hsa_queue_t* rocrQueue = nullptr; // keeps the doorbell frame of a GPU HIP does not drive mapped
  kfd::Queue queue;
};

} // namespace

struct rcclSdmaLendState {
  size_t minBytes = 0, maxBytes = 0;
  char* block = nullptr;
  char* peerBlock = nullptr;
  void* attached = nullptr; // the IPC mapping behind peerBlock; null for a peer in this process
  int* err = nullptr;       // set by a kernel that gave up waiting, read by the host
  int kfdFd = -1;
  int nPaths = 0;
  Path path[kSdmaLendMaxPaths]; // 0: this rank's GPU, then the lenders
  uint32_t epoch = 0;

  char* inbox(char* ofBlock) const {
    return ofBlock + kFlagBytes + (epoch & 1) * maxBytes;
  }
};

namespace {

hsa_status_t addGpu(hsa_agent_t agent, void* out) {
  hsa_device_type_t type;
  uint32_t gpuId = 0;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU &&
      hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_UID), &gpuId) ==
        HSA_STATUS_SUCCESS)
    static_cast<std::vector<Gpu>*>(out)->push_back({agent, gpuId, kfd::node_of(gpuId, 64)});
  return HSA_STATUS_SUCCESS;
}

bool rocrInfo(const void* ptr, hsa_amd_pointer_info_t* info, uint32_t* n = nullptr, hsa_agent_t** acc = nullptr) {
  *info = {};
  info->size = sizeof(*info);
  return hsa_amd_pointer_info(ptr, info, acc ? malloc : nullptr, n, acc) == HSA_STATUS_SUCCESS &&
         info->type != HSA_EXT_POINTER_TYPE_UNKNOWN;
}

// Maps the allocation holding ptr on every engine GPU; an engine faults on an
// unmapped address. Asked on every call and never remembered: a freed block's
// address can come back as a new allocation. hsa_amd_agents_allow_access
// replaces the list, and ranks of one process grant on each other's blocks.
bool mapOnEngines(const rcclSdmaLendState* st, const void* ptr) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  hsa_amd_pointer_info_t info;
  uint32_t n = 0;
  hsa_agent_t* acc = nullptr;
  const bool known = rocrInfo(ptr, &info, &n, &acc);
  std::vector<hsa_agent_t> grant(acc, acc + n);
  free(acc);
  if (!known) return false;
  for (int p = 0; p < st->nPaths; ++p) {
    const uint64_t h = st->path[p].gpu.agent.handle;
    if (h != info.agentOwner.handle &&
        std::none_of(grant.begin(), grant.begin() + n, [h](hsa_agent_t a) { return a.handle == h; }))
      grant.push_back(st->path[p].gpu.agent);
  }
  return grant.size() == n || hsa_amd_agents_allow_access(static_cast<uint32_t>(grant.size()), grant.data(), nullptr,
                                                          info.agentBaseAddress) == HSA_STATUS_SUCCESS;
}

// This rank's half: its block, its GPU and what the peer needs to know. Runs
// with comm->cudaDev current.
bool localSetup(rcclSdmaLendState* st, const std::vector<Gpu>& gpus, Export* ex, std::string* why) {
  ex->minBytes = std::max<int64_t>(rcclParamSdmaLendMinBytes(), 1);
  ex->maxBytes = (std::max(rcclParamSdmaLendMaxBytes(), ex->minBytes) + kfd::kPage - 1) & ~(kfd::kPage - 1);
  ex->maxLenders = std::clamp<int64_t>(rcclParamSdmaLendMaxLenders(), 0, kSdmaLendMaxPaths - 1);
  st->minBytes = ex->minBytes;
  st->maxBytes = ex->maxBytes;

  // Uncached: a kernel reads what the engines wrote.
  hsa_amd_pointer_info_t info;
  if (hipExtMallocWithFlags(reinterpret_cast<void**>(&st->block), kFlagBytes + 2 * st->maxBytes,
                            hipDeviceMallocUncached) != hipSuccess ||
      hipMemset(st->block, 0, kFlagBytes) != hipSuccess ||
      hipHostMalloc(reinterpret_cast<void**>(&st->err), sizeof(int), 0) != hipSuccess || !rocrInfo(st->block, &info) ||
      hsa_amd_ipc_memory_create(info.agentBaseAddress, info.sizeInBytes, &ex->handle) != HSA_STATUS_SUCCESS) {
    *why = "cannot allocate and export two inboxes of " + std::to_string(st->maxBytes) + " bytes";
    return false;
  }
  *st->err = 0;
  ex->len = info.sizeInBytes;
  ex->off = static_cast<uint64_t>(st->block - static_cast<char*>(info.agentBaseAddress));
  ex->addr = reinterpret_cast<uint64_t>(st->block);

  for (const Gpu& g : gpus) {
    if (g.node < 0) continue;
    ex->lendable |= 1ull << g.node;
    if (g.agent.handle == info.agentOwner.handle) st->path[st->nPaths++].gpu = g;
  }
  if (st->nPaths != 1) {
    *why = "the rank's GPU is not in the KFD topology";
    return false;
  }
  ex->node = st->path[0].gpu.node;
  *why = kfd::open_kfd(&st->kfdFd);
  ex->ok = why->empty();
  return ex->ok;
}

// The peer's half: the lenders, the peer's block and the queues.
bool connectPeer(ncclComm* comm, rcclSdmaLendState* st, const std::vector<Gpu>& gpus, const Export* ex,
                 std::string* why) {
  const Export& me = ex[comm->rank];
  const Export& peer = ex[1 - comm->rank];
  if (me.minBytes != peer.minBytes || me.maxBytes != peer.maxBytes || me.maxLenders != peer.maxLenders) {
    *why = "the ranks have different RCCL_SDMA_LEND_* settings";
    return false;
  }
  const auto xgmi = [](int from, int to) { return kfd::link_between(from, to).type == kfd::kIoLinkXgmi; };
  if (!xgmi(me.node, peer.node) || !xgmi(peer.node, me.node)) {
    *why = "the two ranks have no direct xGMI link";
    return false;
  }

  // Lenders: GPUs outside the communicator that both ranks can open, with an
  // xGMI link to both. gpus is in node order, so both ranks build the same list.
  const uint64_t both = me.lendable & peer.lendable & ~(1ull << me.node | 1ull << peer.node);
  for (const Gpu& g : gpus) {
    if (st->nPaths > me.maxLenders) break;
    if (g.node >= 0 && ((both >> g.node) & 1) && xgmi(g.node, me.node) && xgmi(g.node, peer.node))
      st->path[st->nPaths++].gpu = g;
  }
  if (st->nPaths == 1 && me.maxLenders > 0) {
    *why = "no GPU outside the communicator has an xGMI link to both ranks and a ROCr agent in both processes";
    return false;
  }

  // The engines write the peer's block, and stamp departures into this rank's.
  bool mapped = false;
  if (comm->peerInfo[0].pidHash == comm->peerInfo[1].pidHash) {
    // One address space: grant instead of attach.
    st->peerBlock = reinterpret_cast<char*>(peer.addr);
    mapped = mapOnEngines(st, st->peerBlock);
  } else {
    hsa_agent_t agents[kSdmaLendMaxPaths];
    for (int p = 0; p < st->nPaths; ++p) agents[p] = st->path[p].gpu.agent;
    hsa_amd_ipc_memory_t handle = peer.handle;
    mapped = hsa_amd_ipc_memory_attach(&handle, peer.len, static_cast<uint32_t>(st->nPaths), agents, &st->attached) ==
             HSA_STATUS_SUCCESS;
    st->peerBlock = static_cast<char*>(st->attached) + peer.off;
  }
  if (!mapped || !mapOnEngines(st, st->block)) {
    *why = "cannot map the inboxes on the engine GPUs";
    return false;
  }

  for (int p = 0; p < st->nPaths; ++p) {
    Path& path = st->path[p];
    // KFD maps a GPU's doorbell frame with the process's first queue there.
    uint64_t base = 0, size = 0;
    if (!kfd::find_doorbell_frame(path.gpu.gpuId, &base, &size) &&
        hsa_queue_create(path.gpu.agent, 4096, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                         &path.rocrQueue) != HSA_STATUS_SUCCESS) {
      *why = "hsa_queue_create on gpu_id " + std::to_string(path.gpu.gpuId);
      return false;
    }
    // The engine KFD recommends for the link the copy starts on: to the peer
    // from this rank's GPU, to this rank from a lender.
    const int engine =
      sdmaLendPickEngine(kfd::link_between(path.gpu.node, p == 0 ? peer.node : me.node).engine_mask, comm->rank);
    *why = engine < 0 ? "KFD recommends no SDMA engine for a link of the communicator" :
                        kfd::create_queue(st->kfdFd, path.gpu.gpuId, static_cast<uint32_t>(engine), &path.queue);
    if (!why->empty()) return false;
  }
  return true;
}

// Queues first (the engines must be idle), then the peer's mapping.
void closePeer(rcclSdmaLendState* st) {
  for (Path& path : st->path) kfd::destroy_queue(&path.queue);
  if (st->attached) hsa_amd_ipc_memory_detach(st->attached);
  st->attached = nullptr;
}

void destroyState(rcclSdmaLendState* st) {
  closePeer(st);
  for (Path& path : st->path) {
    if (path.rocrQueue) hsa_queue_destroy(path.rocrQueue);
  }
  if (st->kfdFd >= 0) close(st->kfdFd);
  if (st->block) (void)hipFree(st->block);
  if (st->err) (void)hipHostFree(st->err);
  delete st;
}

template <typename T>
void launchReduce(const ReduceArgs& a, cudaStream_t stream, hipEvent_t stopEvent) {
  const auto blocks = static_cast<uint32_t>(std::clamp<size_t>(sdmaLendVecs(a.bytes) / 1024, 64, 1024));
  hipExtLaunchKernelGGL((sdmaLendReduceKernel<T>), dim3(blocks), dim3(256), 0, stream, /*startEvent=*/nullptr,
                        stopEvent, /*flags=*/0, a);
}

} // namespace

ncclResult_t rcclSdmaLendCommInit(ncclComm* comm) {
  // Rank-uniform gate: both ranks run the two all-gathers below, or neither.
  if (rcclParamSdmaLendEnable() == 0) return ncclSuccess;
  if (comm->nRanks != 2 || comm->nNodes != 1 || comm->MNNVL ||
      !(IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"))) {
    INFO(NCCL_INIT, "SDMA lend: not a two-rank single-node gfx942/gfx950 communicator, off");
    return ncclSuccess;
  }

  std::vector<Gpu> gpus;
  hsa_iterate_agents(addGpu, &gpus);
  std::sort(gpus.begin(), gpus.end(), [](const Gpu& a, const Gpu& b) { return a.node < b.node; });

  auto* st = new rcclSdmaLendState();
  std::string why;
  Export ex[2] = {};
  bool ok = localSetup(st, gpus, &ex[comm->rank], &why);
  if (bootstrapAllGather(comm->bootstrap, ex, sizeof(Export)) != ncclSuccess || !ex[1 - comm->rank].ok) {
    if (ok) why = "the peer could not set the path up";
    ok = false;
  }
  ok = ok && connectPeer(comm, st, gpus, ex, &why);

  // Rank-uniform outcome, else the ranks pick different backends and hang.
  int ready[2] = {0, 0};
  ready[comm->rank] = ok;
  if (bootstrapAllGather(comm->bootstrap, ready, sizeof(int)) != ncclSuccess || !ready[0] || !ready[1]) {
    WARN("SDMA lend: off for this communicator: %s", ok ? "the peer could not connect" : why.c_str());
    destroyState(st);
    return ncclSuccess;
  }
  comm->sdmaLend = st;
  INFO(NCCL_INIT, "SDMA lend: rank %d gpu_id %u, %d lender(s), messages of %zu to %zu bytes", comm->rank,
       st->path[0].gpu.gpuId, st->nPaths - 1, st->minBytes, st->maxBytes);
  return ncclSuccess;
}

ncclResult_t rcclSdmaLendCommClosePeerMappings(ncclComm* comm) {
  if (comm->sdmaLend == nullptr) return ncclSuccess;
  // doneEvent is recorded after every launch and the reduce kernel waits for
  // every departure: once it has fired, no engine of this rank is at work.
  if (comm->doneEvent != nullptr && cudaEventSynchronize(comm->doneEvent) != cudaSuccess)
    WARN("SDMA lend: doneEvent sync failed; continuing cleanup");
  closePeer(comm->sdmaLend);
  return ncclSuccess;
}

ncclResult_t rcclSdmaLendCommFini(ncclComm* comm) {
  if (comm->sdmaLend == nullptr) return ncclSuccess;
  (void)rcclSdmaLendCommClosePeerMappings(comm);
  destroyState(comm->sdmaLend);
  comm->sdmaLend = nullptr;
  return ncclSuccess;
}

bool rcclAllReduceSdmaLendEligible(ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  const rcclSdmaLendState* st = comm->sdmaLend;
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));
  return st != nullptr && op == ncclSum &&
         (datatype == ncclFloat32 || datatype == ncclFloat16 || datatype == ncclBfloat16) && bytes >= st->minBytes &&
         bytes <= st->maxBytes;
}

ncclResult_t rcclAllReduceSdmaLend(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream) {
  rcclSdmaLendState* st = comm->sdmaLend;
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));

  // The engines read sendbuff: everything that produces it must have run.
  CUDACHECK(hipStreamSynchronize(stream));
  if (*st->err) {
    WARN("SDMA lend: a kernel gave up waiting for the engines");
    return ncclInternalError;
  }
  if (!mapOnEngines(st, sendbuff)) {
    WARN("SDMA lend: sendbuff is not a ROCr allocation, or cannot be mapped on the engine GPUs");
    return ncclInvalidUsage;
  }

  const uint32_t epoch = ++st->epoch;
  int* flags = reinterpret_cast<int*>(st->block);
  const ReduceArgs a = {static_cast<const char*>(sendbuff),
                        st->inbox(st->block),
                        static_cast<char*>(recvbuff),
                        flags,
                        st->err,
                        sdmaLendMakeSplit(sdmaLendVecs(bytes), st->nPaths),
                        bytes,
                        static_cast<int>(epoch)};
  // Lenders first: their path is longer. An empty piece still gets its
  // fences, so the kernel can wait for every flag.
  for (int p = st->nPaths; p-- > 0;) {
    const size_t off = std::min(a.split.bound[p] * kSdmaLendVecBytes, bytes);
    const size_t end = std::min(a.split.bound[p + 1] * kSdmaLendVecBytes, bytes);
    kfd::Queue& q = st->path[p].queue;
    if (end > off) kfd::put_copy(q, a.inp + off, st->inbox(st->peerBlock) + off, end - off);
    kfd::put_fence(q, reinterpret_cast<int*>(st->peerBlock) + p, epoch);
    kfd::put_fence(q, flags + kSdmaLendMaxPaths + p, epoch);
    kfd::ring_doorbell(q);
  }

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (datatype) {
  case ncclFloat32:
    launchReduce<float>(a, stream, stopEvent);
    break;
  case ncclFloat16:
    launchReduce<_Float16>(a, stream, stopEvent);
    break;
  default:
    launchReduce<hip_bfloat16>(a, stream, stopEvent);
    break;
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
