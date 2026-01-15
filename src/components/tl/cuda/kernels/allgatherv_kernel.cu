/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif
#include "utils/arch/cuda_def.h"
#include "../tl_cuda.h"

#ifdef __cplusplus
}
#endif

#include "nvls.cuh"

namespace cg = cooperative_groups;

__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allgatherv_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *src_u32, uint32_t *base_u32, size_t my_offset,
        size_t my_count, uint32_t tsize)
{
    cg::thread_block          tb        = cg::this_thread_block();
    cg::grid_group            grid      = cg::this_grid();
    bool                      is_leader = (blockIdx.x == 0);

    // Only leader block participates in cross-GPU barrier
    NvlsBar<cg::thread_block> nvls_barrier(
        tb, tsize, mc_bar, uc_bar, is_leader);

    // PRE-BARRIER (hierarchical)
    if (is_leader) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
    }
    grid.sync();

    // KERNEL EXECUTION
    // Each rank copies its data to NVLS mc buffer using multimem store
    uint32_t *dst_ptr = base_u32 + my_offset;
    size_t    stride  = blockDim.x * gridDim.x * 4;
    size_t    tid     = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    for (size_t idx = tid; idx < my_count; idx += stride) {
        uint4 val = reinterpret_cast<uint4 *>(src_u32 + idx)[0];
        MULTIMEM_ST_U32(val, dst_ptr + idx);
    }

    // POST-BARRIER (hierarchical)
    grid.sync();
    if (is_leader) {
        nvls_barrier.sync(cuda::memory_order_release);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allgatherv_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr src_ptr, CUdeviceptr mc_base_addr, size_t my_offset,
    size_t my_count, CUdeviceptr mc_control_addr, CUdeviceptr uc_control_addr,
    uint32_t tsize)
{
    ucc_tl_cuda_nvls_control_t *mc_bar;
    ucc_tl_cuda_nvls_control_t *uc_bar;
    uint32_t                   *src_u32;
    uint32_t                   *base_u32;
    cudaError_t                 cuda_st;

    ucc_assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    ucc_assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    /* NVLS requires 16-byte alignment */
    ucc_assert(my_offset % 4 == 0);
    ucc_assert(my_count % 4 == 0);
    ucc_assert(mc_base_addr % 16 == 0);
    ucc_assert(mc_control_addr % 16 == 0);
    ucc_assert(src_ptr % 16 == 0);

    src_u32  = reinterpret_cast<uint32_t *>(src_ptr);
    base_u32 = reinterpret_cast<uint32_t *>(mc_base_addr);
    mc_bar   = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    uc_bar   = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);

    void *kernel_args[] = {
        &mc_bar, &uc_bar, &src_u32, &base_u32, &my_offset, &my_count, &tsize};

    cuda_st = cudaLaunchCooperativeKernel(
        (void *)allgatherv_kernel_vec32,
        dim3(sm_count),
        dim3(threads),
        kernel_args,
        0,
        stream);
    if (cuda_st != cudaSuccess) {
        return cuda_error_to_ucc_status(cuda_st);
    }

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
