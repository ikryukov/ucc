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

// Specialized 4-CTA kernel for 1-2MB messages
// Uses lightweight atomic barrier (in uc_bar->grid_barrier) instead of grid.sync()
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel_4cta(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t *dst_u32,
        uint32_t tsize)
{
    NvlsBar            nvls_barrier(tsize, mc_bar, uc_bar);
    NvlsControlLayout *uc_ctrl =
        reinterpret_cast<NvlsControlLayout *>(uc_bar);

    // PRE-BARRIER: block 0 does cross-GPU sync
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
        nvls_barrier.commit();
    }

    // Lightweight 4-block barrier using embedded grid_barrier
    grid_barrier_sync(uc_ctrl, 4);

    // KERNEL EXECUTION (pointer-based loop for minimal address math)
    size_t    tid     = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    uint32_t *src_ptr = base_u32 + offset + tid;
    uint32_t *src_end = base_u32 + offset + count;
    uint32_t *dst_ptr = dst_u32 + tid;
    size_t    stride  = blockDim.x * 4 * 4; // 4 CTAs × blockDim.x × 4

    for (; src_ptr < src_end; src_ptr += stride, dst_ptr += stride) {
        uint4 val;
        NvlsOps::ld(val, src_ptr);
        reinterpret_cast<uint4 *>(dst_ptr)[0] = val;
    }
    // Sense-reversing barrier auto-resets, no manual reset needed
}

// Multi-CTA kernel using cooperative groups (for larger SM counts)
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t *dst_u32,
        uint32_t tsize)
{
    cg::grid_group grid = cg::this_grid();
    NvlsBar        nvls_barrier(tsize, mc_bar, uc_bar);

    // PRE-BARRIER: block 0 does cross-GPU sync
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
        nvls_barrier.commit();
    }
    grid.sync(); // release all blocks on this GPU

    // KERNEL EXECUTION (pointer-based loop for minimal address math)
    size_t    tid     = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    uint32_t *src_ptr = base_u32 + offset + tid;
    uint32_t *src_end = base_u32 + offset + count;
    uint32_t *dst_ptr = dst_u32 + tid;
    size_t    stride  = blockDim.x * gridDim.x * 4;

    for (; src_ptr < src_end; src_ptr += stride, dst_ptr += stride) {
        uint4 val;
        NvlsOps::ld(val, src_ptr);
        reinterpret_cast<uint4 *>(dst_ptr)[0] = val;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Unified reduce_scatter kernel launcher
// - sm_count == 4: uses lightweight 4-CTA kernel with cudaLaunchKernelExC
// - sm_count > 4:  uses cooperative kernel with grid.sync()
ucc_status_t post_reduce_scatter_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr dst_ptr, CUdeviceptr mc_base_addr, CUdeviceptr mc_control_addr,
    CUdeviceptr uc_control_addr, size_t offset, size_t count,
    ucc_datatype_t datatype, uint32_t tsize)
{
    ucc_tl_cuda_nvls_control_t *mc_bar;
    ucc_tl_cuda_nvls_control_t *uc_bar;
    uint32_t                   *base_u32;
    uint32_t                   *dst_u32;
    cudaError_t                 cuda_st;

    ucc_assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    ucc_assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    /* NVLS requires 16-byte alignment */
    ucc_assert(offset % 4 == 0);
    ucc_assert(count % 4 == 0);
    ucc_assert(mc_base_addr % 16 == 0);
    ucc_assert(mc_control_addr % 16 == 0);

    base_u32 = reinterpret_cast<uint32_t *>(mc_base_addr);
    mc_bar   = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    uc_bar   = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);
    dst_u32  = reinterpret_cast<uint32_t *>(dst_ptr);

    void *kernel_args[] = {
        &mc_bar, &uc_bar, &base_u32, &offset, &count, &dst_u32, &tsize};

    if (sm_count == 4) {
        // Use lightweight 4-CTA kernel with cudaLaunchKernelExC
        // Grid barrier is embedded in uc_bar->grid_barrier
        cudaLaunchConfig_t config = {};
        config.gridDim            = dim3(4);
        config.blockDim           = dim3(threads);
        config.dynamicSmemBytes   = 0;
        config.stream             = stream;

        switch (datatype) {
        case UCC_DT_FLOAT32:
            cuda_st = cudaLaunchKernelExC(
                &config,
                (void *)reduce_scatter_kernel_4cta<NvlsFp32Ops>,
                kernel_args);
            break;
        case UCC_DT_BFLOAT16:
            cuda_st = cudaLaunchKernelExC(
                &config,
                (void *)reduce_scatter_kernel_4cta<NvlsBf16Ops>,
                kernel_args);
            break;
        default:
            return UCC_ERR_NOT_SUPPORTED;
        }
    } else {
        // Use cooperative kernel with grid.sync()
        switch (datatype) {
        case UCC_DT_FLOAT32:
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)reduce_scatter_kernel_vec32<NvlsFp32Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
            break;
        case UCC_DT_BFLOAT16:
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)reduce_scatter_kernel_vec32<NvlsBf16Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
            break;
        default:
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    if (cuda_st != cudaSuccess) {
        return cuda_error_to_ucc_status(cuda_st);
    }

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
