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
    cg::grid_group grid = cg::this_grid();
    NvlsBar        nvls_barrier(tsize, mc_bar, uc_bar);

    // PRE-BARRIER: block 0 thread 0 does cross-GPU sync
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
    }
    grid.sync(); // release all blocks on this GPU

    // KERNEL EXECUTION
    uint32_t *ptr    = base_u32 + offset;
    size_t    stride = blockDim.x * gridDim.x * 4;
    size_t    tid    = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    for (size_t idx = tid; idx < count; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, ptr + idx);
        NvlsOps::st(val, ptr + idx);
    }

    // POST-BARRIER: gather all blocks, then cross-GPU sync
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
}

// vectorized allreduce kernel for 32-bit lanes, single-block launch only
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32_1sm(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t tsize)
{
    NvlsBar nvls_barrier(tsize, mc_bar, uc_bar);

    // PRE-BARRIER: wait for all GPUs to be ready
    if (threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_relaxed);
    }
    __syncthreads();

    // KERNEL EXECUTION (single block: simplified indexing)
    uint32_t *ptr    = base_u32 + offset + threadIdx.x * 4;
    uint32_t *end    = base_u32 + offset + count;
    size_t    stride = blockDim.x * 4;

#pragma unroll 1
    for (; ptr < end; ptr += stride) {
        uint4 val;
        NvlsOps::ld(val, ptr);
        NvlsOps::st(val, ptr);
    }

    // POST-BARRIER: signal completion and wait for all GPUs
    if (threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
    __syncthreads();
}

// Low-latency allreduce: copy to NVLS buffer, reduce, copy back
// Single-block launch only, for small messages
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32_lowlatency(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        const uint32_t *__restrict__ src_u32, uint32_t *__restrict__ dst_u32,
        uint32_t *__restrict__ mc_nvls_u32, uint32_t *__restrict__ uc_nvls_u32,
        size_t count_u32, uint32_t tsize)
{
    NvlsBar nvls_barrier(tsize, mc_bar, uc_bar);

    size_t  stride = blockDim.x * 4;
    size_t  tid    = threadIdx.x * 4;

    // Stage 1: Vectorized copy from src to NVLS unicast buffer
    for (size_t idx = tid; idx < count_u32; idx += stride) {
        uint4 val = reinterpret_cast<const uint4 *>(src_u32 + idx)[0];
        reinterpret_cast<uint4 *>(uc_nvls_u32 + idx)[0] = val;
    }

    // BARRIER: ensure all GPUs have copied to their NVLS buffers
    if (threadIdx.x == 0) {
        nvls_barrier.sync(cuda::memory_order_release);
        nvls_barrier.commit();
    }
    __syncthreads();

    // Stage 2: NVLS reduce (read via multicast, store to dst)
    for (size_t idx = tid; idx < count_u32; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, mc_nvls_u32 + idx);
        reinterpret_cast<uint4 *>(dst_u32 + idx)[0] = val;
    }
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

    // Verify alignment requirements for vectorized uint4 access
    assert(count_u32 % 4 == 0); // total count must be 16-byte aligned
    assert(offset % 4 == 0);    // chunk offset must be 16-byte aligned
    assert(count % 4 == 0);     // chunk count must be 16-byte aligned

    void *kernel_args[] = {
        &mc_bar, &uc_bar, &base_u32, &offset, &count, &tsize};

    switch (datatype) {
    case UCC_DT_FLOAT32:
        assert(((uintptr_t)(mc_base_addr) % 16) == 0);
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
        assert(((uintptr_t)(mc_base_addr) % 16) == 0);
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

ucc_status_t post_allreduce_lowlatency_kernel(
    cudaStream_t stream, uint32_t threads, CUdeviceptr src_ptr,
    CUdeviceptr dst_ptr, CUdeviceptr mc_nvls_ptr, CUdeviceptr uc_nvls_ptr,
    size_t src_size_bytes, CUdeviceptr mc_control_addr,
    CUdeviceptr uc_control_addr, uint32_t tsize, ucc_datatype_t datatype)
{
    ucc_tl_cuda_nvls_control_t *mc_bar;
    ucc_tl_cuda_nvls_control_t *uc_bar;
    uint32_t                   *src_u32;
    uint32_t                   *dst_u32;
    uint32_t                   *uc_nvls_u32;
    uint32_t                   *mc_nvls_u32;
    cudaError_t                 cuda_st;
    size_t                      count_u32;

    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    src_u32     = reinterpret_cast<uint32_t *>(src_ptr);
    dst_u32     = reinterpret_cast<uint32_t *>(dst_ptr);
    uc_nvls_u32 = reinterpret_cast<uint32_t *>(uc_nvls_ptr);
    mc_nvls_u32 = reinterpret_cast<uint32_t *>(mc_nvls_ptr);
    count_u32   = src_size_bytes / sizeof(uint32_t);
    mc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    uc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);

    assert(((uintptr_t)(mc_nvls_ptr) % 16) == 0);
    assert(((uintptr_t)(uc_nvls_ptr) % 16) == 0);
    assert(((uintptr_t)(mc_control_addr) % 16) == 0);

    void *kernel_args[] = {
        &mc_bar,
        &uc_bar,
        &src_u32,
        &dst_u32,
        &mc_nvls_u32,
        &uc_nvls_u32,
        &count_u32,
        &tsize};

    switch (datatype) {
    case UCC_DT_FLOAT32:
        cuda_st = cudaLaunchCooperativeKernel(
            (void *)allreduce_kernel_vec32_lowlatency<NvlsFp32Ops>,
            dim3(1),
            dim3(threads),
            kernel_args,
            0,
            stream);
        break;
    case UCC_DT_BFLOAT16:
        cuda_st = cudaLaunchCooperativeKernel(
            (void *)allreduce_kernel_vec32_lowlatency<NvlsBf16Ops>,
            dim3(1),
            dim3(threads),
            kernel_args,
            0,
            stream);
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
