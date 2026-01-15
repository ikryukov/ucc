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

template <typename Cooperative> struct NvlsBar {
    Cooperative cooperative;
    uint32_t    base;
    uint32_t    tsize;
    uint32_t   *mc_counter_ptr;
    uint32_t   *uc_counter_ptr;

    __device__  NvlsBar(
         Cooperative coop, uint32_t tsize, void *mc_control_ptr,
         void *uc_control_ptr)
        : cooperative(coop),
          base(reinterpret_cast<NvlsControlLayout *>(uc_control_ptr)->base),
          tsize(tsize),
          mc_counter_ptr(
              &reinterpret_cast<NvlsControlLayout *>(mc_control_ptr)->counter),
          uc_counter_ptr(
              &reinterpret_cast<NvlsControlLayout *>(uc_control_ptr)->counter)
    {
    }

    __device__ ~NvlsBar()
    {
        if (cooperative.thread_rank() == 0) {
            *uc_counter_ptr = base;
        }
        cooperative.sync();
    }

    __device__ void arrive(cuda::memory_order order)
    {
        cooperative.sync();
        if (cooperative.thread_rank() == 0) {
            if (order == cuda::memory_order_release) {
                asm volatile(
                    "multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(
                        mc_counter_ptr),
                    "n"(1)
                    : "memory");
            } else if (order == cuda::memory_order_relaxed) {
                asm volatile("multimem.red.global.add.u32 [%0], %1;" ::"l"(
                                 mc_counter_ptr),
                             "n"(1)
                             : "memory");
            } else {
                assert(false);
            }
            // asm volatile("fence.proxy.alias;" ::: "memory"); // for future ?
        }
    }

    __device__ void wait(cuda::memory_order order)
    {
        if (cooperative.thread_rank() == 0) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ac(
                *uc_counter_ptr);
            uint32_t expected = base + tsize;
            while (ac.load(
                       order == cuda::memory_order_relaxed
                           ? cuda::memory_order_relaxed
                           : cuda::memory_order_acquire) -
                       expected >
                   0x7fffffff) {
            }
            base += tsize;
        }
        cooperative.sync();
    }

    __device__ void sync(cuda::memory_order order)
    {
        arrive(order);
        wait(order);
    }
};

// Traits wrapping NVLS LD/ST variants on 32-bit lanes
struct NvlsFp32Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr)
    {
        MULTIMEM_LD(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr)
    {
        MULTIMEM_ST(v, ptr);
    }
};

struct NvlsBf16Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr)
    {
        MULTIMEM_LD_BF16(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr)
    {
        MULTIMEM_ST_BF16(v, ptr);
    }
};
#endif // __cplusplus

#endif // UCC_TL_CUDA_NVLS_CUH_
