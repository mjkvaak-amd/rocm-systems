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
 * Each rank has an SDMA queue on its own GPU and on every lender. Per call
 * the host cuts sendbuff into one piece per queue and writes copy + fence +
 * fence: the copy lands the piece in the peer's inbox, the first fence stamps
 * an arrival flag at the peer and the second a departure flag at the sender.
 * One kernel on the caller's stream waits for the flags of the pieces it
 * reads, then adds. The peer's next call can land while this rank still adds,
 * so two inboxes alternate.
 *
 * The engines are not ordered with the stream, so the host synchronizes the
 * stream before it posts: the call blocks, and it cannot be captured.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/sdma_lend/sdma_lend_all_reduce.h"

#include "algorithms/sdma_lend/sdma_lend_queue.h"
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

namespace {

namespace sdma = rccl_sdma_lend;

constexpr int kMaxPaths = 8;       // the direct link and up to seven lenders
constexpr uint32_t kMaxNodes = 64; // KFD topology nodes, as the bits of a mask
constexpr size_t kMinBytes = 4 << 20;
constexpr size_t kMaxBytes = 256 << 20;
constexpr size_t kVecBytes = 16; // the kernel adds 16-byte vectors
constexpr size_t kPieceAlign = 4096;
// A rank's block: a flag page, then two inboxes of kMaxBytes. Flag p is the
// arrival of the peer's piece p, flag kMaxPaths + p the departure of this rank's.
constexpr size_t kFlagBytes = 4096;
constexpr size_t kBlockBytes = kFlagBytes + 2 * kMaxBytes;

// Piece p of a message covers bytes [bound[p], bound[p + 1]).
struct Split {
  size_t bound[kMaxPaths + 1];
  int nPaths;
};

// Equal shares. Both ranks compute the same split, so piece p covers the same
// bytes in both directions.
Split makeSplit(size_t bytes, int nPaths) {
  Split s = {};
  s.nPaths = nPaths;
  for (int p = 1; p < nPaths; ++p) s.bound[p] = bytes * p / nPaths / kPieceAlign * kPieceAlign;
  s.bound[nPaths] = bytes;
  return s;
}
// No piece is empty, and one fits the 30-bit count of a copy packet.
static_assert(kMinBytes / kMaxPaths >= kPieceAlign && kMaxBytes <= size_t(1) << 30, "piece size");

// ---- device ------------------------------------------------------------------

// Flags only grow, and a later fence of a queue implies the earlier ones: >=.
// The load races with the engine's write on purpose. Relaxed: the engine wrote
// the piece before the flag, and an acquire load measured 14 us slower per
// call from 16 MiB. False if the communicator aborts first.
__device__ __forceinline__ bool waitFlag(const int* flag, int epoch, const uint32_t* abort) {
  for (uint32_t spins = 0; __atomic_load_n(flag, __ATOMIC_RELAXED) - epoch < 0;) {
    if ((++spins & 0xffff) == 0 && __atomic_load_n(abort, __ATOMIC_RELAXED)) return false;
  }
  return true;
}

// Element-wise sum of two vectors. Both are loaded whole before the store: out may be inp.
template <typename T>
__device__ __forceinline__ void addVec(char* out, const char* inp, const char* inbox) {
  T x[kVecBytes / sizeof(T)], w[kVecBytes / sizeof(T)];
  __builtin_memcpy(x, inp, kVecBytes);
  __builtin_memcpy(w, inbox, kVecBytes);
  for (size_t j = 0; j < kVecBytes / sizeof(T); ++j)
    x[j] = static_cast<T>(static_cast<float>(x[j]) + static_cast<float>(w[j]));
  __builtin_memcpy(out, x, kVecBytes);
}

struct ReduceArgs {
  const char* inp;   // sendbuff
  const char* inbox; // the peer's sendbuff, as the engines delivered it
  char* out;         // recvbuff; may be inp
  const int* flags;
  const uint32_t* abort;
  Split split;
  int epoch;
};

template <typename T>
__global__ void sdmaLendReduceKernel(ReduceArgs a) {
  const size_t bytes = a.split.bound[a.split.nPaths];
  const size_t per = (bytes / kVecBytes + gridDim.x - 1) / gridDim.x * kVecBytes;
  const size_t lo = std::min(blockIdx.x * per, bytes), hi = std::min(lo + per, bytes);
  // A block needs its pieces to have arrived, and to have departed before it
  // writes over them in place. One thread polls for the block.
  __shared__ bool ok;
  if (threadIdx.x == 0) {
    ok = true;
    for (int p = 0; p < a.split.nPaths && ok; ++p) {
      if (a.split.bound[p + 1] > lo && a.split.bound[p] < hi)
        ok = waitFlag(a.flags + p, a.epoch, a.abort) && waitFlag(a.flags + kMaxPaths + p, a.epoch, a.abort);
    }
  }
  __syncthreads();
  if (!ok) return;
  for (size_t off = lo + threadIdx.x * kVecBytes; off < hi; off += blockDim.x * kVecBytes)
    addVec<T>(a.out + off, a.inp + off, a.inbox + off);
}

// ---- host --------------------------------------------------------------------

struct Path {
  hsa_agent_t agent;
  uint32_t node; // KFD topology node
  sdma::Queue queue;
};

// What a rank tells the other.
struct Export {
  uint32_t ok;
  uint32_t node;               // of the rank's GPU
  uint64_t lendable;           // nodes of the GPUs the rank's ROCr sees, as a bit mask
  hsa_amd_ipc_memory_t handle; // the rank's block
};

} // namespace

struct rcclSdmaLendState {
  char* block = nullptr;
  char* peerBlock = nullptr; // the peer's block, attached
  int nPaths = 0;
  Path path[kMaxPaths]; // 0: this rank's GPU, then the lenders
  uint32_t epoch = 0;
};

namespace {

// Fills an array of the process's GPU agents, indexed by KFD topology node.
hsa_status_t addGpu(hsa_agent_t agent, void* byNode) {
  hsa_device_type_t type;
  uint32_t node = 0;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU &&
      hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node) == HSA_STATUS_SUCCESS && node < kMaxNodes)
    static_cast<hsa_agent_t*>(byNode)[node] = agent;
  return HSA_STATUS_SUCCESS;
}

// The SDMA engine of `gpu` that ROCr recommends for copies over its link to
// `to`; -1 if there is none. ROCr recommends one for xGMI links only.
int engineFor(hsa_agent_t gpu, hsa_agent_t to) {
  uint32_t mask = 0;
  return gpu.handle != 0 && to.handle != 0 &&
             hsa_amd_memory_get_preferred_copy_engine(to, gpu, &mask) == HSA_STATUS_SUCCESS && mask != 0 ?
           __builtin_ctz(mask) :
           -1;
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
// replaces the list, so communicators of one process take turns.
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
    const uint64_t h = st->path[p].agent.handle;
    if (h != info.agentOwner.handle &&
        std::none_of(grant.begin(), grant.begin() + n, [h](hsa_agent_t a) { return a.handle == h; }))
      grant.push_back(st->path[p].agent);
  }
  return grant.size() == n || hsa_amd_agents_allow_access(static_cast<uint32_t>(grant.size()), grant.data(), nullptr,
                                                          info.agentBaseAddress) == HSA_STATUS_SUCCESS;
}

// This rank's half: its block and what the peer needs to know. Runs with
// comm->cudaDev current.
bool localSetup(rcclSdmaLendState* st, const hsa_agent_t* gpu, Export* ex, std::string* why) {
  // Uncached: a kernel reads what the engines wrote.
  hsa_amd_pointer_info_t info;
  if (hipExtMallocWithFlags(reinterpret_cast<void**>(&st->block), kBlockBytes, hipDeviceMallocUncached) != hipSuccess ||
      hipMemset(st->block, 0, kFlagBytes) != hipSuccess || !rocrInfo(st->block, &info) ||
      hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_NODE, &ex->node) != HSA_STATUS_SUCCESS ||
      hsa_amd_ipc_memory_create(st->block, kBlockBytes, &ex->handle) != HSA_STATUS_SUCCESS) {
    *why = "cannot allocate and export the inboxes";
    return false;
  }
  st->path[st->nPaths++] = {info.agentOwner, ex->node};
  for (uint32_t n = 0; n < kMaxNodes; ++n) ex->lendable |= static_cast<uint64_t>(gpu[n].handle != 0) << n;
  ex->ok = 1;
  return true;
}

// The peer's half: the lenders, the peer's block and the queues.
bool connectPeer(ncclComm* comm, rcclSdmaLendState* st, const hsa_agent_t* gpu, const Export* ex, std::string* why) {
  const Export& me = ex[comm->rank];
  const Export& peer = ex[1 - comm->rank];
  const hsa_agent_t self = st->path[0].agent;
  const hsa_agent_t other = peer.node < kMaxNodes ? gpu[peer.node] : hsa_agent_t{};
  if (engineFor(self, other) < 0 || engineFor(other, self) < 0) {
    *why = other.handle == 0 ? "ROCr does not see the peer's GPU in this process" :
                               "ROCr recommends no SDMA engine for a direct xGMI link between the two ranks";
    return false;
  }

  // Lenders: GPUs outside the communicator that the ROCr of both ranks sees,
  // with an engine for the link to each rank.
  for (uint32_t n = 0; n < kMaxNodes && st->nPaths < kMaxPaths; ++n) {
    if (((me.lendable & peer.lendable) >> n & 1) && n != me.node && n != peer.node && engineFor(gpu[n], self) >= 0 &&
        engineFor(gpu[n], other) >= 0)
      st->path[st->nPaths++] = {gpu[n], n};
  }
  if (st->nPaths == 1) {
    *why = "no GPU outside the communicator has an xGMI link to both ranks and a ROCr agent in both processes";
    return false;
  }

  // The engines write the peer's block, and stamp departures into this rank's.
  hsa_agent_t agents[kMaxPaths];
  for (int p = 0; p < st->nPaths; ++p) agents[p] = st->path[p].agent;
  hsa_amd_ipc_memory_t handle = peer.handle;
  if (hsa_amd_ipc_memory_attach(&handle, kBlockBytes, static_cast<uint32_t>(st->nPaths), agents,
                                reinterpret_cast<void**>(&st->peerBlock)) != HSA_STATUS_SUCCESS ||
      !mapOnEngines(st, st->block)) {
    *why = "cannot map the inboxes on the engine GPUs";
    return false;
  }

  for (int p = 0; p < st->nPaths; ++p) {
    // The engine for the link the copy starts on: to the peer from this
    // rank's GPU, to this rank from a lender.
    Path& path = st->path[p];
    const int engine = engineFor(path.agent, p == 0 ? other : self);
    if (!sdma::create_queue(path.node, static_cast<uint32_t>(engine), &path.queue)) {
      *why = "no SDMA queue on engine " + std::to_string(engine) + " of node " + std::to_string(path.node);
      return false;
    }
  }
  return true;
}

template <typename T>
void launchReduce(const ReduceArgs& a, size_t bytes, cudaStream_t stream, hipEvent_t stopEvent) {
  const auto blocks = static_cast<uint32_t>(std::clamp<size_t>(bytes / kVecBytes / 1024, 64, 1024));
  hipExtLaunchKernelGGL((sdmaLendReduceKernel<T>), dim3(blocks), dim3(256), 0, stream, /*startEvent=*/nullptr,
                        stopEvent, /*flags=*/0, a);
}

} // namespace

ncclResult_t rcclSdmaLendCommInit(ncclComm* comm) {
  // Rank-uniform gate: both ranks run the two all-gathers below, or neither.
  if (rcclParamSdmaLendEnable() == 0) return ncclSuccess;
  if (comm->nRanks != 2 || comm->nNodes != 1 || comm->MNNVL || !IsArchMatch(comm->archName, "gfx950")) {
    INFO(NCCL_INIT, "SDMA lend: not a two-rank single-node gfx950 communicator, off");
    return ncclSuccess;
  }

  hsa_agent_t gpu[kMaxNodes] = {};
  hsa_iterate_agents(addGpu, gpu);
  // In comm from here on: rcclSdmaLendCommFini frees it, also when init fails.
  rcclSdmaLendState* st = comm->sdmaLend = new rcclSdmaLendState();
  std::string why = "the ranks did not both set it up, with the same number of lenders";
  Export ex[2] = {};
  bool ok = localSetup(st, gpu, &ex[comm->rank], &why);
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, ex, sizeof(Export)));
  ok = ok && ex[1 - comm->rank].ok && connectPeer(comm, st, gpu, ex, &why);

  // Rank-uniform outcome, else the ranks pick different backends and hang.
  int paths[2] = {0, 0};
  paths[comm->rank] = ok ? st->nPaths : 0;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, paths, sizeof(int)));
  if (paths[0] == 0 || paths[0] != paths[1]) {
    WARN("SDMA lend: off for this communicator: %s", why.c_str());
    return rcclSdmaLendCommFini(comm);
  }
  INFO(NCCL_INIT, "SDMA lend: rank %d on node %u, %d lender(s)", comm->rank, st->path[0].node, st->nPaths - 1);
  return ncclSuccess;
}

ncclResult_t rcclSdmaLendCommFini(ncclComm* comm) {
  rcclSdmaLendState* st = comm->sdmaLend;
  if (st == nullptr) return ncclSuccess;
  // doneEvent is recorded after every launch: once it has fired, no kernel
  // reads the block. After an abort the engines can still be at work:
  // destroy_queue waits for this rank's, and the peer's write through the
  // peer's own attachment, which keeps the block's memory until it detaches.
  if (comm->doneEvent != nullptr && cudaEventSynchronize(comm->doneEvent) != cudaSuccess)
    WARN("SDMA lend: doneEvent sync failed; continuing cleanup");
  for (Path& path : st->path) sdma::destroy_queue(&path.queue);
  if (st->peerBlock) hsa_amd_ipc_memory_detach(st->peerBlock);
  if (st->block) (void)hipFree(st->block);
  delete st;
  comm->sdmaLend = nullptr;
  return ncclSuccess;
}

bool rcclAllReduceSdmaLendEligible(ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));
  return comm->sdmaLend != nullptr && op == ncclSum &&
         (datatype == ncclFloat32 || datatype == ncclFloat16 || datatype == ncclBfloat16) && bytes >= kMinBytes &&
         bytes <= kMaxBytes && bytes % kVecBytes == 0;
}

ncclResult_t rcclAllReduceSdmaLend(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream) {
  rcclSdmaLendState* st = comm->sdmaLend;
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(datatype));

  // The engines read sendbuff: everything that produces it must have run.
  CUDACHECK(hipStreamSynchronize(stream));
  if (!mapOnEngines(st, sendbuff)) {
    // No fallback: the other rank cannot know, and is lending.
    WARN("SDMA lend: sendbuff is not a ROCr allocation, or cannot be mapped on the engine GPUs");
    return ncclInvalidUsage;
  }

  const uint32_t epoch = ++st->epoch;
  const size_t inbox = kFlagBytes + (epoch & 1) * kMaxBytes;
  int* flags = reinterpret_cast<int*>(st->block);
  const ReduceArgs a = {
    static_cast<const char*>(sendbuff), st->block + inbox,      static_cast<char*>(recvbuff), flags, comm->abortFlagDev,
    makeSplit(bytes, st->nPaths),       static_cast<int>(epoch)
  };
  // Lenders first: their path is longer.
  for (int p = st->nPaths; p-- > 0;) {
    const size_t off = a.split.bound[p];
    sdma::Queue& q = st->path[p].queue;
    sdma::put_copy(q, a.inp + off, st->peerBlock + inbox + off, a.split.bound[p + 1] - off);
    sdma::put_fence(q, reinterpret_cast<int*>(st->peerBlock) + p, epoch);
    sdma::put_fence(q, flags + kMaxPaths + p, epoch);
    sdma::ring_doorbell(q);
  }

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (datatype) {
  case ncclFloat32:
    launchReduce<float>(a, bytes, stream, stopEvent);
    break;
  case ncclFloat16:
    launchReduce<_Float16>(a, bytes, stream, stopEvent);
    break;
  default:
    launchReduce<hip_bfloat16>(a, bytes, stream, stopEvent);
    break;
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
