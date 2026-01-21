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
#include <cooperative_groups.h>
#include <cassert>

#define MULTIMEM_ST(val, ptr)                                                  \
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),  \
                 "r"(val.x),                                                   \
                 "r"(val.y),                                                   \
                 "r"(val.z),                                                   \
                 "r"(val.w)                                                    \
                 : "memory");

#define MULTIMEM_ST_U32(val, ptr)                                              \
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),  \
                 "r"(val.x),                                                   \
                 "r"(val.y),                                                   \
                 "r"(val.z),                                                   \
                 "r"(val.w)                                                    \
                 : "memory");

#define MULTIMEM_LD(val, ptr)                                                  \
    asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"            \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#define MULTIMEM_ST_BF16(val, ptr)                                             \
    asm volatile(                                                              \
        "multimem.st.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr),        \
        "r"(val.x),                                                            \
        "r"(val.y),                                                            \
        "r"(val.z),                                                            \
        "r"(val.w)                                                             \
        : "memory");

#define MULTIMEM_LD_BF16(val, ptr)                                             \
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"         \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#ifdef __cplusplus

struct NvlsControlLayout {
    uint32_t base;
    uint32_t counter;
};

// Optimized NVLS barrier - minimal overhead, explicit sync control
struct NvlsBar {
    uint32_t           base;
    uint32_t           tsize;
    NvlsControlLayout *mc_ctrl;
    NvlsControlLayout *uc_ctrl;

    __device__ __forceinline__
    NvlsBar(uint32_t tsize, void *mc_control_ptr, void *uc_control_ptr)
        : base(reinterpret_cast<NvlsControlLayout *>(uc_control_ptr)->base),
          tsize(tsize),
          mc_ctrl(reinterpret_cast<NvlsControlLayout *>(mc_control_ptr)),
          uc_ctrl(reinterpret_cast<NvlsControlLayout *>(uc_control_ptr))
    {
    }

    // Arrive: signal this GPU is ready
    // MUST be called by leader thread only (threadIdx.x == 0)
    __device__ __forceinline__ void arrive(cuda::memory_order order)
    {
        if (order == cuda::memory_order_release) {
            asm volatile(
                "multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(
                    &mc_ctrl->counter),
                "n"(1)
                : "memory");
        } else {
            asm volatile("multimem.red.global.add.u32 [%0], %1;" ::"l"(
                             &mc_ctrl->counter),
                         "n"(1)
                         : "memory");
        }
        // Critical: ensure NVLS writes are visible before proceeding
        // asm volatile("fence.proxy.alias;" ::: "memory");
    }

    // Wait: spin until all GPUs have arrived
    // MUST be called by leader thread only (threadIdx.x == 0)
    __device__ __forceinline__ void wait()
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ac(
            uc_ctrl->counter);
        uint32_t expected = base + tsize;
        while (ac.load(cuda::memory_order_acquire) - expected > 0x7fffffff) {
        }
        base = expected;
    }

    // Commit base for next kernel launch
    // MUST be called by leader thread only (threadIdx.x == 0)
    __device__ __forceinline__ void commit()
    {
        uc_ctrl->base = base;
    }

    // Full barrier: arrive + wait
    // MUST be called by leader thread only (threadIdx.x == 0)
    // Caller MUST call __syncthreads() after this returns
    __device__ __forceinline__ void sync(cuda::memory_order order)
    {
        arrive(order);
        wait();
    }
};

// Vectorized 128-bit load/store (LDG.E.128 / STG.E.128)
__device__ __forceinline__ void vec_ld(uint4 &v, const uint32_t *ptr)
{
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr)
        : "memory");
}

__device__ __forceinline__ void vec_st(const uint4 &v, uint32_t *ptr)
{
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Traits wrapping NVLS LD/ST variants on 32-bit lanes
struct NvlsFp32Ops {
    __device__ __forceinline__ static void ld(uint4 &v, const uint32_t *ptr)
    {
        MULTIMEM_LD(v, ptr);
    }
    __device__ __forceinline__ static void st(const uint4 &v, uint32_t *ptr)
    {
        MULTIMEM_ST(v, ptr);
    }
};

struct NvlsBf16Ops {
    __device__ __forceinline__ static void ld(uint4 &v, const uint32_t *ptr)
    {
        MULTIMEM_LD_BF16(v, ptr);
    }
    __device__ __forceinline__ static void st(const uint4 &v, uint32_t *ptr)
    {
        MULTIMEM_ST_BF16(v, ptr);
    }
};
#endif // __cplusplus

#endif // UCC_TL_CUDA_NVLS_CUH_
