/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_SDMA_LEND_KFD_QUEUE_H_
#define RCCL_SDMA_LEND_KFD_QUEUE_H_

// Private SDMA queues on any GPU of the process, written from the host.
//
// KFD keeps one process object per mm, so a second /dev/kfd fd reaches the
// process ROCr set up. A queue then needs a ring and read/write pointer pages
// registered as userptr BOs at GPU VA == CPU VA, and the doorbell frame ROCr
// mapped with its first queue on that GPU.
//
// Needs KFD ioctl 1.17 (queue type SDMA_BY_ENG_ID). A host-written doorbell
// is prompt on bare metal; in an SR-IOV guest it can take milliseconds.
//
// Queue creation follows Nico Holmberg's kfdq.hpp.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace rccl_sdma_lend {

// uapi/linux/kfd_ioctl.h, spelled out: an installed header can predate 1.17
// and lack sdma_engine_id.
struct GetVersionArgs {
  uint32_t major_version, minor_version;
};
struct CreateQueueArgs {
  uint64_t ring_base_address, write_pointer_address, read_pointer_address, doorbell_offset;
  uint32_t ring_size, gpu_id, queue_type, queue_percentage, queue_priority, queue_id;
  uint64_t eop_buffer_address, eop_buffer_size, ctx_save_restore_address;
  uint32_t ctx_save_restore_size, ctl_stack_size, sdma_engine_id, pad;
};
struct DestroyQueueArgs {
  uint32_t queue_id, pad;
};
struct AllocArgs {
  uint64_t va_addr, size, handle, mmap_offset;
  uint32_t gpu_id, flags;
};
struct FreeArgs {
  uint64_t handle;
};
struct MapArgs {
  uint64_t handle, device_ids_array_ptr;
  uint32_t n_devices, n_success;
};

constexpr unsigned long kIocGetVersion = _IOR('K', 0x01, GetVersionArgs);
constexpr unsigned long kIocCreateQueue = _IOWR('K', 0x02, CreateQueueArgs);
constexpr unsigned long kIocDestroyQueue = _IOWR('K', 0x03, DestroyQueueArgs);
constexpr unsigned long kIocAlloc = _IOWR('K', 0x16, AllocArgs);
constexpr unsigned long kIocFree = _IOW('K', 0x17, FreeArgs);
constexpr unsigned long kIocMap = _IOWR('K', 0x18, MapArgs);
constexpr unsigned long kIocUnmap = _IOWR('K', 0x19, MapArgs);

constexpr uint32_t kQueueTypeSdmaByEngId = 0x4;
constexpr uint32_t kMemUserptr = 1u << 2;
constexpr uint32_t kMemUncached = 1u << 25;
constexpr uint32_t kMemCoherent = 1u << 26;
constexpr uint32_t kMemExecutable = 1u << 30;
constexpr uint32_t kMemWritable = 1u << 31;

constexpr uint32_t kIoLinkXgmi = 11; // HSA_IOLINKTYPE_XGMI

// SDMA packets. COPY_LINEAR: header, count - 1, parameters, source, destination.
// FENCE: header, address, value.
constexpr uint32_t kOpCopy = 1;
constexpr uint32_t kOpFence = 5;
constexpr size_t kMaxCopyBytes = size_t(1) << 29; // the count field holds 30 bits
constexpr uint32_t kRingBytes = 256 * 1024;
constexpr uint64_t kPage = 4096;

inline std::string node_dir(int node) {
  return "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node);
}

/// The topology node of a GPU, below `limit`; -1 if there is none.
inline int node_of(uint32_t gpu_id, int limit) {
  for (int n = 0; n < limit; ++n) {
    std::ifstream f(node_dir(n) + "/gpu_id");
    uint32_t id = 0;
    if (!f.is_open()) break;
    if (f >> id && id == gpu_id) return n;
  }
  return -1;
}

struct Link {
  uint32_t type = 0;        // 0: no direct link
  uint32_t engine_mask = 0; // recommended_sdma_engine_id_mask
};

/// The direct io_link from one topology node to another.
inline Link link_between(int node_from, int node_to) {
  for (int l = 0; l < 64; ++l) {
    std::ifstream p(node_dir(node_from) + "/io_links/" + std::to_string(l) + "/properties");
    if (!p.is_open()) break;
    std::string key;
    long long value, to = -1, type = 0, mask = 0;
    while (p >> key >> value) {
      if (key == "node_to") to = value;
      if (key == "type") type = value;
      if (key == "recommended_sdma_engine_id_mask") mask = value;
    }
    if (to == node_to) return {static_cast<uint32_t>(type), static_cast<uint32_t>(mask)};
  }
  return {};
}

/// ROCr's doorbell frame for gpu_id: the /dev/kfd mapping whose file offset is
/// KFD_MMAP_TYPE_DOORBELL (3 << 62) | gpu_id << 46. It exists once the process
/// has any queue on that GPU.
inline bool find_doorbell_frame(uint32_t gpu_id, uint64_t* base, uint64_t* size) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    unsigned long lo = 0, hi = 0, off = 0;
    char perms[8] = {};
    if (line.find("kfd") == std::string::npos || sscanf(line.c_str(), "%lx-%lx %7s %lx", &lo, &hi, perms, &off) != 4)
      continue;
    if ((off >> 62) == 3 && ((off >> 46) & 0xFFFF) == (gpu_id & 0xFFFF)) {
      *base = lo;
      *size = hi - lo;
      return true;
    }
  }
  return false;
}

/// A /dev/kfd fd of this process in *fd. Returns "" or the reason.
inline std::string open_kfd(int* fd) {
  *fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (*fd < 0) return std::string("open /dev/kfd: ") + strerror(errno);
  GetVersionArgs v{};
  if (ioctl(*fd, kIocGetVersion, &v) || v.major_version != 1 || v.minor_version < 17)
    return "KFD ioctl " + std::to_string(v.major_version) + "." + std::to_string(v.minor_version) +
           " < 1.17: no SDMA_BY_ENG_ID queues";
  return "";
}

/// Zeroed, locked host pages registered on a GPU at their own address.
struct HostBuffer {
  char* ptr = nullptr;
  uint64_t size = 0, handle = 0;
};

inline void free_host_buffer(int fd, uint32_t gpu_id, HostBuffer* b) {
  if (!b->ptr) return;
  MapArgs m{b->handle, reinterpret_cast<uint64_t>(&gpu_id), 1, 0};
  ioctl(fd, kIocUnmap, &m);
  FreeArgs f{b->handle};
  ioctl(fd, kIocFree, &f);
  munmap(b->ptr, b->size);
  b->ptr = nullptr;
}

inline bool alloc_host_buffer(int fd, uint32_t gpu_id, uint64_t size, uint32_t extra_flags, HostBuffer* b) {
  // A fresh anonymous mapping, aligned to its size: heap pages can sit in a
  // range ROCr once registered as a userptr, which KFD refuses with EADDRINUSE.
  void* r = mmap(nullptr, 2 * size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (r == MAP_FAILED) return false;
  const uint64_t lo = reinterpret_cast<uint64_t>(r), va = (lo + size - 1) & ~(size - 1);
  if (va > lo) munmap(r, va - lo);
  if (va < lo + size) munmap(reinterpret_cast<void*>(va + size), lo + size - va);
  mlock(reinterpret_cast<void*>(va), size); // fresh anonymous pages read as zero
  AllocArgs a{va, size, 0, va, gpu_id, kMemUserptr | kMemWritable | kMemCoherent | kMemUncached | extra_flags};
  if (ioctl(fd, kIocAlloc, &a)) {
    munmap(reinterpret_cast<void*>(va), size);
    return false;
  }
  *b = {reinterpret_cast<char*>(va), size, a.handle};
  MapArgs m{a.handle, reinterpret_cast<uint64_t>(&gpu_id), 1, 0};
  if (ioctl(fd, kIocMap, &m) == 0 && m.n_success == 1) return true;
  free_host_buffer(fd, gpu_id, b);
  return false;
}

struct Queue {
  int fd = -1;
  uint32_t gpu_id = 0, queue_id = 0;
  bool created = false;
  HostBuffer ring, rptr, wptr;
  uint64_t wpos = 0;
  uint64_t* doorbell = nullptr;
};

/// One SDMA queue on `engine` of a GPU whose doorbell frame is mapped.
/// Returns "" or the reason.
inline std::string create_queue(int fd, uint32_t gpu_id, uint32_t engine, Queue* q) {
  const std::string where = " on gpu_id " + std::to_string(gpu_id);
  uint64_t bell_base = 0, bell_size = 0;
  if (!find_doorbell_frame(gpu_id, &bell_base, &bell_size) || (bell_size & (bell_size - 1)))
    return "no doorbell frame" + where;
  q->fd = fd;
  q->gpu_id = gpu_id;
  if (!alloc_host_buffer(fd, gpu_id, kRingBytes, kMemExecutable, &q->ring) ||
      !alloc_host_buffer(fd, gpu_id, kPage, 0, &q->rptr) || !alloc_host_buffer(fd, gpu_id, kPage, 0, &q->wptr))
    return "queue buffers" + where + ": " + strerror(errno);
  CreateQueueArgs a{};
  a.ring_base_address = reinterpret_cast<uint64_t>(q->ring.ptr);
  a.ring_size = kRingBytes;
  a.read_pointer_address = reinterpret_cast<uint64_t>(q->rptr.ptr);
  a.write_pointer_address = reinterpret_cast<uint64_t>(q->wptr.ptr);
  a.gpu_id = gpu_id;
  a.queue_type = kQueueTypeSdmaByEngId;
  a.queue_percentage = 100;
  a.queue_priority = 15;
  a.sdma_engine_id = engine;
  if (ioctl(fd, kIocCreateQueue, &a))
    return "AMDKFD_IOC_CREATE_QUEUE" + where + " engine " + std::to_string(engine) + ": " + strerror(errno);
  q->queue_id = a.queue_id;
  q->created = true;
  q->doorbell = reinterpret_cast<uint64_t*>(bell_base + (a.doorbell_offset & (bell_size - 1)));
  return "";
}

/// Idempotent.
inline void destroy_queue(Queue* q) {
  if (q->created) {
    DestroyQueueArgs a{q->queue_id, 0};
    ioctl(q->fd, kIocDestroyQueue, &a);
    q->created = false;
  }
  for (HostBuffer* b : {&q->ring, &q->rptr, &q->wptr}) free_host_buffer(q->fd, q->gpu_id, b);
}

inline void put(Queue& q, const uint32_t* words, size_t n) {
  const size_t bytes = n * sizeof(uint32_t);
  uint64_t pos = q.wpos % kRingBytes;
  if (pos + bytes > kRingBytes) {
    memset(q.ring.ptr + pos, 0, kRingBytes - pos); // NOPs to the end of the ring
    q.wpos += kRingBytes - pos;
    pos = 0;
  }
  memcpy(q.ring.ptr + pos, words, bytes);
  q.wpos += bytes;
}

inline void put_copy(Queue& q, const void* src, void* dst, size_t bytes) {
  auto s = reinterpret_cast<uintptr_t>(src), d = reinterpret_cast<uintptr_t>(dst);
  while (bytes) {
    const size_t n = bytes < kMaxCopyBytes ? bytes : kMaxCopyBytes;
    const uint32_t w[7] = {kOpCopy,
                           static_cast<uint32_t>(n - 1),
                           0,
                           static_cast<uint32_t>(s),
                           static_cast<uint32_t>(s >> 32),
                           static_cast<uint32_t>(d),
                           static_cast<uint32_t>(d >> 32)};
    put(q, w, 7);
    s += n;
    d += n;
    bytes -= n;
  }
}

/// The engine writes `value` to `addr` once everything queued before has run.
inline void put_fence(Queue& q, void* addr, uint32_t value) {
  const auto a = reinterpret_cast<uintptr_t>(addr);
  const uint32_t w[4] = {kOpFence, static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), value};
  put(q, w, 4);
}

inline void ring_doorbell(Queue& q) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(q.wptr.ptr), q.wpos, __ATOMIC_SEQ_CST);
  __atomic_store_n(q.doorbell, q.wpos, __ATOMIC_SEQ_CST);
}

} // namespace rccl_sdma_lend

#endif
