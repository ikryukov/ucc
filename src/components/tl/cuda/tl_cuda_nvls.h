/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_H_
#define UCC_TL_CUDA_NVLS_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>  // For CU_MEM_CREATE_USAGE_MULTICAST
#include "components/base/ucc_base_iface.h"  // For ucc_base_context_t

// Forward declaration to avoid circular dependency
struct ucc_tl_cuda_team;

#define UCC_TL_CUDA_NVLS_TEAM_SIZE 72 // TODO: make this dynamic?
#define UCC_TL_CUDA_NVLS_MAX_BLOCKS_PER_GPU 32

typedef struct ucc_tl_cuda_nvls {
    CUmemGenericAllocationHandle mc_handle;    // Multicast handle for NVLS
    CUmemGenericAllocationHandle mc_memhandle; // Multicast memory handle for NVLS
    CUdeviceptr                  mc_va;        // Device pointer for multicast memory
    CUdeviceptr                  uc_va;        // Device pointer for unicast memory
    size_t                       mc_size;      // Size of multicast memory
    size_t                       mc_offset;    // Offset of the multicast memory
    size_t                      *coll_ids;     // Coll id for the each task in flight slot, needed for barrier
} ucc_tl_cuda_nvls_t;

/*
 * Ensure that frequently-updated fields used by different block groups do not
 * share an L2 cache line across GPUs. Place count and sense on separate
 * 128-byte cache lines within each barrier element, and align each element to
 * 128 bytes. This makes sizeof(barrier) == 256 bytes with count at offset 0
 * and sense at offset 128.
 */
#define UCC_TL_CUDA_NVLS_CACHELINE 128
typedef struct ucc_tl_cuda_nvls_barrier {
    /* line 0: counter */
    ucc_rank_t count;
    char       _pad_count[UCC_TL_CUDA_NVLS_CACHELINE - sizeof(ucc_rank_t)];

    /* line 1: sense flag */
    int        sense;
    char       _pad_sense[UCC_TL_CUDA_NVLS_CACHELINE - sizeof(int)];
} __attribute__((aligned(UCC_TL_CUDA_NVLS_CACHELINE))) ucc_tl_cuda_nvls_barrier_t;

typedef struct ucc_tl_cuda_nvls_control {
    ucc_tl_cuda_nvls_barrier_t barriers[UCC_TL_CUDA_NVLS_MAX_BLOCKS_PER_GPU];
} ucc_tl_cuda_nvls_control_t;

ucc_status_t ucc_tl_cuda_nvls_init(struct ucc_tl_cuda_team *self, ucc_base_context_t *tl_context);

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *self, ucc_base_context_t *tl_context);

#endif // UCC_TL_CUDA_NVLS_H_
