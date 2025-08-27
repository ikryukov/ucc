/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_CUH_
#define UCC_TL_CUDA_NVLS_CUH_

#include <cuda.h>
#include <stdint.h>
#include <cuda/atomic>

#include "components/tl/cuda/tl_cuda_nvls.h"

#define MULTIMEM_ST(val, ptr)                                                  \
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),  \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                \
                 : "memory");

#define MULTIMEM_LD(val, ptr)                                                  \
    asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"            \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#define MULTIMEM_ST_BF16(val, ptr)                                               \
    asm volatile("multimem.st.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                  \
                 : "memory");

#define MULTIMEM_LD_BF16(val, ptr)                                             \
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"         \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#ifdef __cplusplus
// NVLS global barrier helper used by kernels to synchronize via multicast/unicast counters
__device__ __forceinline__ void nvls_bar(ucc_tl_cuda_nvls_barrier_t *mc_barrier,
                                         ucc_tl_cuda_nvls_barrier_t *uc_barrier,
                                         ucc_rank_t team_size)
{
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_system> sense_ref(uc_barrier[blockIdx.x].sense);
        int local_sense = 1 - sense_ref.load(cuda::memory_order_relaxed);

        // first thread in block increments the counter
        cuda::atomic_ref<ucc_rank_t, cuda::thread_scope_system> count_ref(uc_barrier[blockIdx.x].count);
        ucc_rank_t pos = count_ref.fetch_add(1, cuda::memory_order_relaxed);

        if (pos == team_size - 1) {
            count_ref.store(0, cuda::memory_order_relaxed);
            sense_ref.store(local_sense, cuda::memory_order_release);
            printf("I'm the last one\n");
        } else {
            printf("waiting for sense\n");
            // wait for other blocks/GPUs to publish the new sense
            while (sense_ref.load(cuda::memory_order_acquire) != local_sense) {
                __nanosleep(64);
            }
        }
    }
    // all other threads in block wait for the first thread to finish
    __syncthreads();
}

// Traits wrapping NVLS LD/ST variants on 32-bit lanes
struct NvlsFp32Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr) {
        MULTIMEM_LD(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr) {
        MULTIMEM_ST(v, ptr);
    }
};

struct NvlsBf16Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr) {
        MULTIMEM_LD_BF16(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr) {
        MULTIMEM_ST_BF16(v, ptr);
    }
};
#endif // __cplusplus

#endif // UCC_TL_CUDA_NVLS_CUH_
