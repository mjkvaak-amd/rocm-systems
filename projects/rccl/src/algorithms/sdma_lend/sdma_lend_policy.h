/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Path split and engine choice of the SDMA mesh-lending AllReduce. No HIP and
 * no librccl dependency, so test/sdma_lend/sdma_lend_policy_test.cpp compiles
 * this header alone.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_SDMA_LEND_POLICY_H_
#define RCCL_SDMA_LEND_POLICY_H_

#include <cstddef>
#include <cstdint>

// Paths of one flow: the direct link and one per lender.
constexpr int kSdmaLendMaxPaths = 8;
// The kernel adds 16-byte vectors; a piece starts on a 4 KiB boundary.
constexpr size_t kSdmaLendVecBytes = 16;
constexpr size_t kSdmaLendVecAlign = 256;

constexpr size_t sdmaLendVecs(size_t bytes) {
  return (bytes + kSdmaLendVecBytes - 1) / kSdmaLendVecBytes;
}

// Piece p of a flow covers vectors [bound[p], bound[p + 1]).
struct SdmaLendSplit {
  size_t bound[kSdmaLendMaxPaths + 1];
  int nPaths;
};

// Equal shares; the last path takes the remainder. Both ranks compute the same
// split, so piece p covers the same vectors in both directions.
constexpr SdmaLendSplit sdmaLendMakeSplit(size_t vecs, int nPaths) {
  SdmaLendSplit s = {};
  s.nPaths = nPaths;
  const size_t n = static_cast<size_t>(nPaths);
  for (size_t p = 1; p < n; ++p) s.bound[p] = vecs * p / n / kSdmaLendVecAlign * kSdmaLendVecAlign;
  s.bound[n] = vecs;
  return s;
}

// The engine of a link for this rank, out of KFD's recommended mask; -1 if the
// mask is empty. Even engines first: a private queue on an odd engine was seen
// to stall on gfx942, and Anvil has the same rule. The ranks take different
// engines when they can, so they do not share an engine of a lender.
inline int sdmaLendPickEngine(uint32_t mask, int rank) {
  if (mask & 0x55555555u) mask &= 0x55555555u;
  if (mask == 0) return -1;
  for (int skip = rank % __builtin_popcount(mask); skip > 0; --skip) mask &= mask - 1;
  return __builtin_ctz(mask);
}

#endif
