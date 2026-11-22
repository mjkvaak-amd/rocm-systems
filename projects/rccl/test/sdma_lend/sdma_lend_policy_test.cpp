/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for src/algorithms/sdma_lend/sdma_lend_policy.h. No GPU, no librccl.

#include <gtest/gtest.h>

#include <cstddef>

#include "algorithms/sdma_lend/sdma_lend_policy.h"

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;

TEST(SdmaLendPolicy, VecsRoundsUp) {
  EXPECT_EQ(sdmaLendVecs(0), 0u);
  EXPECT_EQ(sdmaLendVecs(16), 1u);
  EXPECT_EQ(sdmaLendVecs(17), 2u);
}

TEST(SdmaLendPolicy, SplitCoversEveryVectorOnce) {
  for (size_t vecs : {size_t{0}, size_t{1}, size_t{257}, 4 * kMiB / 16, 80 * kMiB / 16 + 3, 256 * kMiB / 16}) {
    for (int paths = 1; paths <= kSdmaLendMaxPaths; ++paths) {
      const SdmaLendSplit s = sdmaLendMakeSplit(vecs, paths);
      ASSERT_EQ(s.nPaths, paths);
      EXPECT_EQ(s.bound[0], 0u);
      EXPECT_EQ(s.bound[paths], vecs);
      for (int p = 0; p < paths; ++p) {
        EXPECT_LE(s.bound[p], s.bound[p + 1]) << "vecs=" << vecs << " paths=" << paths << " p=" << p;
        EXPECT_EQ(s.bound[p] % kSdmaLendVecAlign, 0u); // a piece starts on a 4 KiB boundary
      }
    }
  }
}

TEST(SdmaLendPolicy, SplitSharesAreEqualWithinOneAlignment) {
  const size_t vecs = 80 * kMiB / 16;
  for (int paths = 2; paths <= kSdmaLendMaxPaths; ++paths) {
    const SdmaLendSplit s = sdmaLendMakeSplit(vecs, paths);
    for (int p = 0; p < paths; ++p) {
      const size_t share = s.bound[p + 1] - s.bound[p];
      EXPECT_GE(share + kSdmaLendVecAlign, vecs / paths);
      EXPECT_LE(share, vecs / paths + static_cast<size_t>(paths) * kSdmaLendVecAlign);
    }
  }
}

TEST(SdmaLendPolicy, EnginePrefersEvenAndSeparatesRanks) {
  EXPECT_EQ(sdmaLendPickEngine(0, 0), -1);
  // Only an odd engine recommended: take it.
  EXPECT_EQ(sdmaLendPickEngine(0x2, 0), 1);
  EXPECT_EQ(sdmaLendPickEngine(0x2, 1), 1);
  // Engines 2 and 3: the even one for both ranks.
  EXPECT_EQ(sdmaLendPickEngine(0xC, 0), 2);
  EXPECT_EQ(sdmaLendPickEngine(0xC, 1), 2);
  // Engines 0, 2, 4, 5: the ranks take different even engines.
  EXPECT_EQ(sdmaLendPickEngine(0x35, 0), 0);
  EXPECT_EQ(sdmaLendPickEngine(0x35, 1), 2);
}

} // namespace
