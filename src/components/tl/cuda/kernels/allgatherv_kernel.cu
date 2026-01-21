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

// Small message kernel (≤16KB = 4096 uint32s): NO LOOP, bounds check only
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allgatherv_kernel_small(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *src_u32, uint32_t *base_u32, size_t my_offset,
        size_t my_count, uint32_t tsize)
{
    NvlsBar nvls_barrier(tsize, mc_bar, uc_bar);

    // No loop - just bounds check (1024 threads × 4 = 4096 uint32s = 16KB)
    size_t idx = threadIdx.x * 4;
    if (idx < my_count) {
        uint4 val = *reinterpret_cast<uint4 *>(src_u32 + idx);
        MULTIMEM_ST_U32(val, base_u32 + my_offset + idx);
    }

    // POST-BARRIER
    __syncthreads();
    if (threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
}

// Multi-block kernel: 1-4 SMs, no cooperative launch needed
// Uses lightweight grid barrier for POST synchronization
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allgatherv_kernel_multiblock(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *src_u32, uint32_t *base_u32, size_t my_offset,
        size_t my_count, uint32_t tsize, uint32_t num_blocks)
{
    NvlsBar            nvls_barrier(tsize, mc_bar, uc_bar);
    NvlsControlLayout *uc_ctrl = reinterpret_cast<NvlsControlLayout *>(uc_bar);

    // Pointer-based loop
    size_t    tid     = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    uint32_t *src_ptr = src_u32 + tid;
    uint32_t *src_end = src_u32 + my_count;
    uint32_t *dst_ptr = base_u32 + my_offset + tid;
    size_t    stride  = blockDim.x * num_blocks * 4;

    for (; src_ptr < src_end; src_ptr += stride, dst_ptr += stride) {
        uint4 val = *reinterpret_cast<uint4 *>(src_ptr);
        MULTIMEM_ST_U32(val, dst_ptr);
    }

    // POST-BARRIER: sync all blocks, then cross-GPU barrier
    grid_barrier_sync(uc_ctrl, num_blocks);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
}

// Multi-CTA kernel using cooperative groups (for larger SM counts)
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allgatherv_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *src_u32, uint32_t *base_u32, size_t my_offset,
        size_t my_count, uint32_t tsize)
{
    cg::grid_group grid = cg::this_grid();
    NvlsBar        nvls_barrier(tsize, mc_bar, uc_bar);

    // Pointer-based loop
    size_t    tid     = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    uint32_t *src_ptr = src_u32 + tid;
    uint32_t *src_end = src_u32 + my_count;
    uint32_t *dst_ptr = base_u32 + my_offset + tid;
    size_t    stride  = blockDim.x * gridDim.x * 4;

    for (; src_ptr < src_end; src_ptr += stride, dst_ptr += stride) {
        uint4 val = *reinterpret_cast<uint4 *>(src_ptr);
        MULTIMEM_ST_U32(val, dst_ptr);
    }

    // POST-BARRIER (hierarchical)
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Unified allgatherv kernel launcher
// - my_count <= 4096 (≤16KB): small kernel (no loop, 1 block)
// - sm_count <= 4: multiblock kernel (lightweight grid barrier)
// - sm_count > 4: cooperative kernel with grid.sync()
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

    // Kernel selection:
    // - ≤16KB (4096 uint32s): small kernel (no loop, 1 block)
    // - 1-4 SMs: multiblock kernel (lightweight grid barrier)
    // - >4 SMs: cooperative kernel with grid.sync()

    cudaLaunchConfig_t config = {};
    config.blockDim           = dim3(threads);
    config.dynamicSmemBytes   = 0;
    config.stream             = stream;

    if (my_count <= 4096) {
        // ≤16KB: small kernel with no loop
        void *kernel_args[] = {
            &mc_bar, &uc_bar, &src_u32, &base_u32, &my_offset, &my_count,
            &tsize};
        config.gridDim = dim3(1);
        cuda_st = cudaLaunchKernelExC(
            &config, (void *)allgatherv_kernel_small, kernel_args);
    } else if (sm_count <= 4) {
        // 1-4 SMs: multiblock kernel with lightweight barrier
        void *kernel_args[] = {
            &mc_bar, &uc_bar, &src_u32, &base_u32, &my_offset, &my_count,
            &tsize, &sm_count};
        config.gridDim = dim3(sm_count);
        cuda_st = cudaLaunchKernelExC(
            &config, (void *)allgatherv_kernel_multiblock, kernel_args);
    } else {
        // >4 SMs: cooperative kernel with grid.sync()
        void *kernel_args[] = {
            &mc_bar, &uc_bar, &src_u32, &base_u32, &my_offset, &my_count,
            &tsize};
        cuda_st = cudaLaunchCooperativeKernel(
            (void *)allgatherv_kernel_vec32,
            dim3(sm_count),
            dim3(threads),
            kernel_args,
            0,
            stream);
    }

    if (cuda_st != cudaSuccess) {
        return cuda_error_to_ucc_status(cuda_st);
    }

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
