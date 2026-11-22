/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Size policy, path split and flag layout of the SDMA mesh-lending AllReduce.
 * No HIP and no librccl dependency: the host unit test in
 * test/sdma_lend/sdma_lend_policy_test.cpp compiles this header alone, and
 * the kernel and the host code in sdma_lend_all_reduce.cu use the same
 * functions.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_SDMA_LEND_POLICY_H_
#define RCCL_SDMA_LEND_POLICY_H_

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define SDMA_LEND_HD __host__ __device__ __forceinline__
#else
#define SDMA_LEND_HD inline
#endif

// Two ranks only: one shot, each rank's whole input goes to the peer.
constexpr int kSdmaLendRanks = 2;
// Paths of one flow: the direct link (an engine of the sender) and one per lender.
constexpr int kSdmaLendMaxPaths = 8;
// One int flag per 64 bytes, so two engines never write the same cache line.
constexpr int kSdmaLendFlagStride = 16;
// The payload moves in 16-byte vectors; a piece starts on a 4 KiB boundary.
constexpr size_t kSdmaLendVecBytes = 16;
constexpr size_t kSdmaLendVecAlign = 256;

// Defaults of RCCL_SDMA_LEND_MIN_BYTES and RCCL_SDMA_LEND_MAX_BYTES.
constexpr size_t kSdmaLendDefaultMinBytes = 4ULL * 1024 * 1024;
constexpr size_t kSdmaLendDefaultMaxBytes = 256ULL * 1024 * 1024;

enum SdmaLendFlagDir : int {
  kSdmaLendArrival = 0,   // stamped at the receiver once a piece has landed
  kSdmaLendDeparture = 1, // stamped at the sender once a piece has been read
};

constexpr size_t kSdmaLendFlagBytes = 2 * kSdmaLendMaxPaths * kSdmaLendFlagStride * sizeof(int);

// Flag of path `path`, in ints from the start of a rank's flag page.
SDMA_LEND_HD int sdmaLendFlagIndex(int dir, int path) {
  return (dir * kSdmaLendMaxPaths + path) * kSdmaLendFlagStride;
}

// Piece p of a flow covers vectors [bound[p], bound[p + 1]).
struct SdmaLendSplit {
  size_t bound[kSdmaLendMaxPaths + 1];
  int nPaths;
};

// Equal shares for every path. The last path takes the remainder, so the
// pieces always cover [0, vecs). Both ranks compute the same split from the
// same (vecs, nPaths), which is what lets a kernel wait on the arrival and
// the departure flag of the same piece index.
SDMA_LEND_HD SdmaLendSplit sdmaLendMakeSplit(size_t vecs, int nPaths) {
  SdmaLendSplit s = {};
  if (nPaths < 1) nPaths = 1;
  if (nPaths > kSdmaLendMaxPaths) nPaths = kSdmaLendMaxPaths;
  s.nPaths = nPaths;
  for (int p = 0; p < nPaths; ++p) {
    s.bound[p + 1] = p == nPaths - 1 ? vecs :
                                       vecs * static_cast<size_t>(p + 1) / static_cast<size_t>(nPaths) /
                                         kSdmaLendVecAlign * kSdmaLendVecAlign;
  }
  return s;
}

SDMA_LEND_HD size_t sdmaLendVecs(size_t bytes) {
  return (bytes + kSdmaLendVecBytes - 1) / kSdmaLendVecBytes;
}

// The message sizes this path takes. maxBytes is also the size of an inbox.
inline bool sdmaLendSizeEligible(size_t bytes, size_t minBytes, size_t maxBytes) {
  return bytes > 0 && bytes >= minBytes && bytes <= maxBytes;
}

// In place (the same pointer) and disjoint buffers are supported. A partial
// overlap is not: the engines read sendbuff while the kernel writes recvbuff.
inline bool sdmaLendBuffersOk(const void* sendbuff, const void* recvbuff, size_t bytes) {
  const uintptr_t s = reinterpret_cast<uintptr_t>(sendbuff);
  const uintptr_t r = reinterpret_cast<uintptr_t>(recvbuff);
  if (s == 0 || r == 0) return false;
  if (s == r) return true;
  return s + bytes <= r || r + bytes <= s;
}

// The engine of a link for this rank, out of KFD's recommended mask. Even
// engines are preferred (a private queue on an odd engine was seen to stall on
// gfx942, and Anvil has the same rule). Ranks take different engines when the
// mask has more than one, so two ranks never share an engine of a lender.
// Returns -1 when the mask is empty.
inline int sdmaLendPickEngine(uint32_t mask, int rank) {
  uint32_t even = mask & 0x55555555u;
  uint32_t use = even != 0 ? even : mask;
  int n = __builtin_popcount(use);
  if (n == 0) return -1;
  int want = (rank < 0 ? 0 : rank) % n;
  for (int b = 0; b < 32; ++b) {
    if ((use >> b) & 1u) {
      if (want == 0) return b;
      --want;
    }
  }
  return -1;
}

// KFD's location_id of a PCI bus/device/function.
inline uint32_t sdmaLendLocationId(unsigned bus, unsigned dev, unsigned fn) {
  return ((bus & 0xFFu) << 8) | ((dev & 0x1Fu) << 3) | (fn & 0x7u);
}

#undef SDMA_LEND_HD

#endif
