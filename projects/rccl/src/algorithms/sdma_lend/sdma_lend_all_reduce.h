/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SDMA mesh-lending AllReduce (RCCL_SDMA_LEND_ENABLE=1): a two-rank
 * communicator borrows the SDMA engines of the GPUs outside it. See
 * sdma_lend_all_reduce.cu.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_SDMA_LEND_ALL_REDUCE_H_
#define RCCL_SDMA_LEND_ALL_REDUCE_H_

#include "nccl.h"

#include <cstddef>

struct ncclComm;

// Collective at comm init. A rank that cannot set the path up turns it off for
// both; only a bootstrap error fails the comm.
ncclResult_t rcclSdmaLendCommInit(struct ncclComm* comm);
ncclResult_t rcclSdmaLendCommFini(struct ncclComm* comm);

bool rcclAllReduceSdmaLendEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op);
ncclResult_t rcclAllReduceSdmaLend(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   struct ncclComm* comm, cudaStream_t stream);

#endif
