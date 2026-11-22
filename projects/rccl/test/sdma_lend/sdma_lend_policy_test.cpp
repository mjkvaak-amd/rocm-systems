/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for src/algorithms/sdma_lend/sdma_lend_policy.h. No GPU, no
// librccl: the header is the code the reduce kernel and the host posting loop
// of the SDMA mesh-lending AllReduce share.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <set>

#include "algorithms/sdma_lend/sdma_lend_policy.h"

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;

TEST(SdmaLendPolicy, DefaultsMatchDocumentedBand) {
  EXPECT_EQ(kSdmaLendRanks, 2);
  EXPECT_EQ(kSdmaLendDefaultMinBytes, 4 * kMiB);
  EXPECT_EQ(kSdmaLendDefaultMaxBytes, 256 * kMiB);
  EXPECT_EQ(kSdmaLendVecBytes * kSdmaLendVecAlign, 4096u);
}

TEST(SdmaLendPolicy, SizeBandIsInclusive) {
  EXPECT_FALSE(sdmaLendSizeEligible(0, 0, 256 * kMiB));
  EXPECT_FALSE(sdmaLendSizeEligible(4 * kMiB - 1, 4 * kMiB, 256 * kMiB));
  EXPECT_TRUE(sdmaLendSizeEligible(4 * kMiB, 4 * kMiB, 256 * kMiB));
  EXPECT_TRUE(sdmaLendSizeEligible(256 * kMiB, 4 * kMiB, 256 * kMiB));
  EXPECT_FALSE(sdmaLendSizeEligible(256 * kMiB + 1, 4 * kMiB, 256 * kMiB));
}

TEST(SdmaLendPolicy, VecsRoundsUp) {
  EXPECT_EQ(sdmaLendVecs(0), 0u);
  EXPECT_EQ(sdmaLendVecs(1), 1u);
  EXPECT_EQ(sdmaLendVecs(16), 1u);
  EXPECT_EQ(sdmaLendVecs(17), 2u);
  EXPECT_EQ(sdmaLendVecs(80 * kMiB), 80 * kMiB / 16);
}

TEST(SdmaLendPolicy, SplitCoversEveryVectorOnce) {
  const size_t sizes[] = {0, 1, 255, 256, 257, 4097, 4 * kMiB / 16, 80 * kMiB / 16 + 3, 256 * kMiB / 16};
  for (size_t vecs : sizes) {
    for (int paths = 1; paths <= kSdmaLendMaxPaths; ++paths) {
      const SdmaLendSplit s = sdmaLendMakeSplit(vecs, paths);
      ASSERT_EQ(s.nPaths, paths);
      EXPECT_EQ(s.bound[0], 0u);
      EXPECT_EQ(s.bound[paths], vecs);
      for (int p = 0; p < paths; ++p) {
        EXPECT_LE(s.bound[p], s.bound[p + 1]) << "vecs=" << vecs << " paths=" << paths << " p=" << p;
        // Every piece but the first starts on a 4 KiB boundary of the message.
        if (p > 0) {
          EXPECT_EQ(s.bound[p] % kSdmaLendVecAlign, 0u);
        }
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

TEST(SdmaLendPolicy, SplitClampsThePathCount) {
  EXPECT_EQ(sdmaLendMakeSplit(1000, 0).nPaths, 1);
  EXPECT_EQ(sdmaLendMakeSplit(1000, -3).nPaths, 1);
  EXPECT_EQ(sdmaLendMakeSplit(1000, kSdmaLendMaxPaths + 5).nPaths, kSdmaLendMaxPaths);
}

TEST(SdmaLendPolicy, FlagsAreDistinctAndOneCacheLineApart) {
  std::set<int> seen;
  for (int dir : {static_cast<int>(kSdmaLendArrival), static_cast<int>(kSdmaLendDeparture)}) {
    for (int p = 0; p < kSdmaLendMaxPaths; ++p) {
      const int idx = sdmaLendFlagIndex(dir, p);
      EXPECT_EQ(idx % kSdmaLendFlagStride, 0);
      EXPECT_LT(static_cast<size_t>(idx) * sizeof(int), kSdmaLendFlagBytes);
      EXPECT_TRUE(seen.insert(idx).second);
    }
  }
  EXPECT_EQ(static_cast<size_t>(kSdmaLendFlagStride) * sizeof(int), 64u);
}

TEST(SdmaLendPolicy, BuffersInPlaceOrDisjoint) {
  char* base = reinterpret_cast<char*>(uintptr_t{0x10000});
  EXPECT_TRUE(sdmaLendBuffersOk(base, base, 4096));
  EXPECT_TRUE(sdmaLendBuffersOk(base, base + 4096, 4096));
  EXPECT_TRUE(sdmaLendBuffersOk(base + 4096, base, 4096));
  EXPECT_FALSE(sdmaLendBuffersOk(base, base + 4095, 4096));
  EXPECT_FALSE(sdmaLendBuffersOk(base + 1, base, 4096));
  EXPECT_FALSE(sdmaLendBuffersOk(nullptr, base, 4096));
  EXPECT_FALSE(sdmaLendBuffersOk(base, nullptr, 4096));
}

TEST(SdmaLendPolicy, EnginePrefersEvenAndSeparatesRanks) {
  EXPECT_EQ(sdmaLendPickEngine(0, 0), -1);
  // Only odd engines recommended: take one of them.
  EXPECT_EQ(sdmaLendPickEngine(0x2, 0), 1);
  EXPECT_EQ(sdmaLendPickEngine(0x2, 1), 1);
  // Engines 2 and 3: the even one for both ranks.
  EXPECT_EQ(sdmaLendPickEngine(0xC, 0), 2);
  EXPECT_EQ(sdmaLendPickEngine(0xC, 1), 2);
  // Engines 0, 2, 4, 5: the ranks take different even engines.
  EXPECT_EQ(sdmaLendPickEngine(0x35, 0), 0);
  EXPECT_EQ(sdmaLendPickEngine(0x35, 1), 2);
  EXPECT_EQ(sdmaLendPickEngine(0x35, 2), 4);
  EXPECT_EQ(sdmaLendPickEngine(0x35, 3), 0);
  EXPECT_EQ(sdmaLendPickEngine(0x35, -1), 0);
}

TEST(SdmaLendPolicy, LocationIdIsKfdBdf) {
  EXPECT_EQ(sdmaLendLocationId(0x00, 0x00, 0), 0u);
  EXPECT_EQ(sdmaLendLocationId(0x05, 0x00, 0), 0x0500u);
  EXPECT_EQ(sdmaLendLocationId(0xf5, 0x1f, 7), 0xf5ffu);
}

} // namespace
