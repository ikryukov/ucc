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

// vectorized allreduce kernel for 32-bit lanes
// Uses two-level hierarchical barrier:
//   1. grid.sync() - all blocks on this GPU sync
//   2. Block 0 does NVLS cross-GPU barrier
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t tsize)
{
    cg::thread_block          tb        = cg::this_thread_block();
    cg::grid_group            grid      = cg::this_grid();
    bool                      is_leader = (blockIdx.x == 0);

    // Only block 0 participates in cross-GPU barrier (tsize arrivals expected)
    NvlsBar<cg::thread_block> nvls_barrier(
        tb, tsize, mc_bar, uc_bar, is_leader);

    // PRE-BARRIER (hierarchical)
    if (is_leader) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
    }
    // Release all blocks on this GPU
    grid.sync();

    // KERNEL EXECUTION
    uint32_t *ptr    = base_u32 + offset;
    size_t    stride = blockDim.x * gridDim.x * 4;
    size_t    tid    = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    for (size_t idx = tid; idx < count; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, ptr + idx);
        NvlsOps::st(val, ptr + idx);
    }

    // POST-BARRIER (hierarchical)
    grid.sync();
    if (is_leader) {
        nvls_barrier.sync(cuda::memory_order_release);
    }
}

// vectorized allreduce kernel for 32-bit lanes, single-block launch only
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32_1sm(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t tsize)
{
    // This kernel is for single-block launch only
    assert(gridDim.x == 1);

    cg::thread_block          tb = cg::this_thread_block();
    NvlsBar<cg::thread_block> nvls_barrier(tb, tsize, mc_bar, uc_bar);

    // PRE-BARRIER
    nvls_barrier.sync(cuda::memory_order_relaxed);

    // KERNEL EXECUTION (single block: simplified indexing)
    uint32_t *ptr    = base_u32 + offset;
    size_t    stride = blockDim.x * 4;

    for (size_t idx = threadIdx.x * 4; idx < count; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, ptr + idx);
        NvlsOps::st(val, ptr + idx);
    }

    // POST-BARRIER
    nvls_barrier.sync(cuda::memory_order_release);
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allreduce_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr mc_base_addr, size_t src_size_bytes,
    CUdeviceptr mc_control_addr, CUdeviceptr uc_control_addr, uint32_t rank,
    uint32_t tsize, ucc_datatype_t datatype)
{
    ucc_tl_cuda_nvls_control_t *mc_bar;
    ucc_tl_cuda_nvls_control_t *uc_bar;
    uint32_t                   *base_u32;
    cudaError_t                 cuda_st;
    size_t                      count_u32;
    size_t                      offset;
    size_t                      count;

    assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    base_u32  = reinterpret_cast<uint32_t *>(mc_base_addr);
    count_u32 = src_size_bytes / sizeof(uint32_t);
    mc_bar    = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    uc_bar    = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);
    // compute chunk boundaries on host instead of in kernel
    offset    = (count_u32 * rank) / tsize;
    count     = (count_u32 * (rank + 1)) / tsize - offset;

    void *kernel_args[] = {
        &mc_bar, &uc_bar, &base_u32, &offset, &count, &tsize};

    switch (datatype) {
    case UCC_DT_FLOAT32:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        if (sm_count == 1) {
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)allreduce_kernel_vec32_1sm<NvlsFp32Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
        } else {
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)allreduce_kernel_vec32<NvlsFp32Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
        }
        break;
    case UCC_DT_BFLOAT16:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        if (sm_count == 1) {
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)allreduce_kernel_vec32_1sm<NvlsBf16Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
        } else {
            cuda_st = cudaLaunchCooperativeKernel(
                (void *)allreduce_kernel_vec32<NvlsBf16Ops>,
                dim3(sm_count),
                dim3(threads),
                kernel_args,
                0,
                stream);
        }
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (cuda_st != cudaSuccess) {
        return cuda_error_to_ucc_status(cuda_st);
    }

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
