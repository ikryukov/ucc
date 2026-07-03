/**
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_H_
#define UCC_TL_CUDA_NVLS_H_

#include <cuda.h>
#include "ucc/api/ucc_status.h"

// Forward declaration to avoid circular dependency
struct ucc_tl_cuda_lib;
struct ucc_tl_cuda_team;
struct ucc_base_context;

typedef enum {
    UCC_TL_CUDA_NVLS_HANDLE_TYPE_POSIX,
    UCC_TL_CUDA_NVLS_HANDLE_TYPE_FABRIC
} ucc_tl_cuda_nvls_handle_type_t;

typedef struct {
    ucc_tl_cuda_nvls_handle_type_t type;
    /* Rank 0 sets this to UCC_ERR_NOT_SUPPORTED when cuMulticastCreate fails.
     * Non-root ranks read share_data[0].status to propagate the error instead
     * of relying on garbage-handle import failure. */
    ucc_status_t status;
    union {
        struct {
            pid_t pid;
            int   handle;
        } posix;
        CUmemFabricHandle fabric;
    } data;
} ucc_tl_cuda_nvls_handle_t;

typedef enum {
    UCC_TL_CUDA_NVLS_STATE_INIT,
    UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES,
    UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE,
    UCC_TL_CUDA_NVLS_STATE_SYNC_STATUS,
    UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE,
    /* Final OOB barrier to sync all ranks */
    UCC_TL_CUDA_NVLS_STATE_BARRIER,
} ucc_tl_cuda_nvls_state_t;

#define UCC_TL_CUDA_NVLS_REG_CACHE_SIZE 16

/* One cached user-buffer NVLS registration: a per-buffer multicast object
 * bound over the user VA, plus its multicast alias used for zero-copy AR. */
typedef struct ucc_tl_cuda_nvls_reg {
    void                        *ptr;      /* user buffer base */
    size_t                       size;     /* registered (granularity-rounded) */
    CUmemGenericAllocationHandle mc_handle;/* per-buffer multicast object */
    CUdeviceptr                  mc_va;    /* multicast alias of the user buffer */
} ucc_tl_cuda_nvls_reg_t;

typedef struct ucc_tl_cuda_nvls {
    // Multicast handle
    CUmemGenericAllocationHandle mc_handle;
    // Multicast memory handle
    CUmemGenericAllocationHandle mc_memhandle;
    // Device pointer for multicast
    CUdeviceptr                  mc_va;
    // Device pointer for unicast
    CUdeviceptr                  uc_va;
    // Size of multicast memory
    size_t                       mc_size;
    // Offset of multicast memory
    size_t                       mc_offset;
    // Coll id for each task
    size_t                      *coll_ids;
    // Whether the team is multi-node
    int                          is_multinode;
    // Temporary buffer for allgather
    ucc_tl_cuda_nvls_handle_t   *share_data;
    // State variables for re-entrant initialization
    ucc_status_t                 status_supported;
    // Handle for export (POSIX or fabric)
    ucc_tl_cuda_nvls_handle_t    local_handle;
    // CUDA device ID
    int                          device;
    // Minimum granularity
    size_t                       minGran;
    // Granularity
    size_t                       gran;
    /* temporary buffer for STATE_BARRIER */
    char                        *barrier_data;
    /* Whether this rank locally succeeded in importing the multicast handle.
     * Exchanged across ranks in STATE_SYNC_STATUS so a per-rank import failure
     * (e.g. pidfd_getfd EPERM) disables NVLS on the whole team instead of
     * deadlocking the ranks that succeeded in cuMulticastBindAddr. */
    int                          init_ready;
    /* temporary buffer for STATE_SYNC_STATUS allgather */
    char                        *init_sync_data;
    /* Set to 1 only when NVLS initialization fully succeeded for this team.
     * Gates advertising the NVLS collectives: the hardware may support
     * multicast while NVLS init fell back (e.g. peer fd import denied), and in
     * that case collectives must not be routed to the NVLS algorithms. */
    int                          enabled;
    /* User-buffer registration cache for zero-copy allreduce. */
    ucc_tl_cuda_nvls_reg_t       reg_cache[UCC_TL_CUDA_NVLS_REG_CACHE_SIZE];
    int                          reg_count;
} ucc_tl_cuda_nvls_t;

typedef struct ucc_tl_cuda_nvls_control {
    uint32_t base;
    uint32_t counter;
    /* lightweight grid sync for 4-CTA kernels */
    uint32_t grid_barrier;
    /* padding for 16-byte alignment */
    uint32_t _pad;
} ucc_tl_cuda_nvls_control_t;

ucc_status_t ucc_tl_cuda_nvls_check_support(
    struct ucc_tl_cuda_lib *lib, int device, int is_multinode);

ucc_status_t ucc_tl_cuda_nvls_init(
    struct ucc_tl_cuda_team *team, struct ucc_base_context *tl_context);

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *team);

/* Add the local device to a multicast object and bind a device VA range into
 * it (collective; cuMulticastBindAddr barriers across binding ranks). */
ucc_status_t ucc_tl_cuda_nvls_bind_va(CUmemGenericAllocationHandle mc_handle,
                                      int device, CUdeviceptr va, size_t size);

/* Reserve a device VA and map a multicast handle into it (R/W). */
ucc_status_t ucc_tl_cuda_nvls_map_mc(CUmemGenericAllocationHandle mc_handle,
                                     size_t size, size_t gran, int device,
                                     CUdeviceptr *mc_va_out);

/* Collectively register a user device buffer for zero-copy NVLS: creates a
 * per-buffer multicast object (fabric handle), binds the user VA into it and
 * returns its multicast alias. Cached per team. Returns UCC_ERR_NOT_SUPPORTED
 * if the buffer cannot be multicast-bound (e.g. not VMM-backed). */
ucc_status_t ucc_tl_cuda_nvls_register(struct ucc_tl_cuda_team *team, void *ptr,
                                       size_t size, CUdeviceptr *mc_va_out);

#endif // UCC_TL_CUDA_NVLS_H_
