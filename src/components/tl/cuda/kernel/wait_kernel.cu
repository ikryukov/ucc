/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "utils/arch/cuda_def.h"
#include "../tl_cuda.h"

#ifdef __cplusplus
}
#endif

__global__ void wait_kernel(CUdeviceptr addr, uint32_t val) {
    volatile uint32_t tmp;
    do
    {
        tmp = *((uint32_t*)addr);
    } while (tmp != val);
    return;
}

__global__ void write_kernel(CUdeviceptr addr, uint32_t val) {
    *((uint32_t*)addr) = val;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t
post_wait_kernel(cudaStream_t stream, CUdeviceptr addr, uint32_t val) {
    wait_kernel<<<1, 1, 0, stream>>>(addr, val);
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

ucc_status_t
post_write_kernel(cudaStream_t stream, CUdeviceptr addr, uint32_t val) {
    write_kernel<<<1, 1, 0, stream>>>(addr, val);
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
