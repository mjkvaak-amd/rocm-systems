/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SDMA mesh-lending AllReduce for a two-rank communicator on a node whose
 * xGMI mesh has more GPUs than the communicator.
 *
 * Opt-in: RCCL_SDMA_LEND_ENABLE=1. Plain user buffers, no window
 * registration, default RCCL build. See sdma_lend_all_reduce.cu.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_SDMA_LEND_ALL_REDUCE_H_
#define RCCL_SDMA_LEND_ALL_REDUCE_H_

#include "nccl.h"

#include <cstddef>
#include <cstdint>

struct ncclComm;
struct rcclSdmaLendState;

// RCCL_SDMA_LEND_ENABLE. Defined in sdma_lend_all_reduce.cu.
int64_t rcclParamSdmaLendEnable();

// Collective at comm init: every rank of an eligible comm runs the same two
// bootstrap all-gathers, so the outcome (path on or off) is the same on every
// rank. Never fails the comm: a rank that cannot set the path up turns it off
// for all.
ncclResult_t rcclSdmaLendCommInit(struct ncclComm* comm);
// Destroys this rank's queues and unmaps the peer's buffers. Idempotent.
ncclResult_t rcclSdmaLendCommClosePeerMappings(struct ncclComm* comm);
ncclResult_t rcclSdmaLendCommFini(struct ncclComm* comm);

bool rcclAllReduceSdmaLendEligible(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op);
uint32_t rcclAllReduceSdmaLendBlocks(size_t count, ncclDataType_t datatype);
// Paths of one flow: 1 + the number of lenders. 0 when the path is off.
int rcclSdmaLendNumPaths(struct ncclComm* comm);

ncclResult_t rcclAllReduceSdmaLend(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream);

#endif
