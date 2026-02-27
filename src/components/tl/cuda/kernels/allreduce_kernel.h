/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_ALLREDUCE_KERNEL_H_
#define UCC_TL_CUDA_ALLREDUCE_KERNEL_H_

#include <cuda.h>
#include "ucc/api/ucc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel function declaration
ucc_status_t post_allreduce_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr mc_base_addr, size_t src_size_bytes,
    CUdeviceptr mc_control_addr, CUdeviceptr uc_control_addr, uint32_t rank,
    uint32_t tsize, ucc_datatype_t datatype);

// Low-latency allreduce: copies src -> NVLS, reduces, copies NVLS -> dst
ucc_status_t post_allreduce_lowlatency_kernel(
    cudaStream_t stream, uint32_t threads, CUdeviceptr src_ptr,
    CUdeviceptr dst_ptr, CUdeviceptr mc_nvls_ptr, CUdeviceptr uc_nvls_ptr,
    size_t src_size_bytes, CUdeviceptr mc_control_addr,
    CUdeviceptr uc_control_addr, uint32_t tsize, ucc_datatype_t datatype);

#ifdef __cplusplus
}
#endif

#endif // UCC_TL_CUDA_ALLREDUCE_KERNEL_H_
