/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "core/ucc_team.h"
#include "components/mc/base/ucc_mc_base.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "bcast/bcast.h"
#include "reduce_scatter/reduce_scatter.h"
#include "reduce_scatterv/reduce_scatterv.h"

static ucc_config_field_t ucc_tl_cuda_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"MAX_CONCURRENT", "8",
     "Maximum number of outstanding colls",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, max_concurrent),
     UCC_CONFIG_TYPE_UINT},

    {"SCRATCH_SIZE", "2Mb",
     "Size of the internal scratch buffer",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, scratch_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLGATHER_RING_MAX_RINGS", "auto",
     "Max number of rings used in allgather and allgatherv ring algorithms",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, allgather_ring_max_rings),
     UCC_CONFIG_TYPE_ULUNITS},

    {"ALLGATHER_RING_NUM_CHUNKS", "4",
     "Number of chunks each ring message will be split into",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, allgather_ring_num_chunks),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_SCATTER_RING_MAX_RINGS", "auto",
     "Max number of rings used in reduce_scatter and "
     "reduce_scatterv ring algorithms",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, reduce_scatter_ring_max_rings),
     UCC_CONFIG_TYPE_ULUNITS},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t *base_attr);

ucc_status_t ucc_tl_cuda_get_lib_properties(ucc_base_lib_properties_t *prop);

static ucs_config_field_t ucc_tl_cuda_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {NULL}};

ucc_status_t ucc_tl_cuda_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *base_attr);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_cuda_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);

UCC_TL_IFACE_DECLARE(cuda, CUDA);

__attribute__((constructor)) static void tl_cuda_iface_init(void)
{

    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHER)] =
        ucc_tl_cuda_allgather_algs;
    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHERV)] =
        ucc_tl_cuda_allgatherv_algs;
    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_BCAST)] =
        ucc_tl_cuda_bcast_algs;
    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTER)] =
        ucc_tl_cuda_reduce_scatter_algs;
    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTERV)] =
        ucc_tl_cuda_reduce_scatterv_algs;
}

bool init_semaphore(stream_semaphore_t *sem)
{
    assert(sem);
    sem->dev_sem_val_ptr = 0;
    sem->host_val = -1; // TODO: pass init value?
    ucc_memory_cpu_store_fence();
    CUresult res  = cuMemHostRegister(&sem->host_val, sizeof(sem->host_val),
                                     CU_MEMHOSTREGISTER_DEVICEMAP);
    if (res != CUDA_SUCCESS) {
        ucc_error("Failed to cuMemHostRegister!");
        ucc_assert(0);
        return false;
    }
    res = cuMemHostGetDevicePointer(&sem->dev_sem_val_ptr, &sem->host_val,
                                    0 /* flags */);
    if (res != CUDA_SUCCESS) {
        ucc_error("cuMemHostGetDevicePointer failed!");
        ucc_assert(0);
        return false;
    }
    return true;
}

bool wait_semaphore(cudaStream_t stream, stream_semaphore_t *sem, int32_t value)
{
    ucc_assert(sem != NULL);
    ucc_assert(sem->dev_sem_val_ptr);
    CUresult res = cuStreamWaitValue32(stream, sem->dev_sem_val_ptr, value,
                                       CU_STREAM_WAIT_VALUE_EQ);
    if (res != CUDA_SUCCESS) {
        cudaError_t err = cudaGetLastError();
        ucc_error("cuStreamWaitValue32 failed with code: %d %s", res, cudaGetErrorString(err));
        ucc_assert(0);
        return false;
    }
    return true;
}

void set_val_semaphore(cudaStream_t stream, stream_semaphore_t *sem, int32_t value)
{
    ucc_assert(sem != NULL);
    ucc_assert(sem->dev_sem_val_ptr);
    CUresult res = cuStreamWriteValue32(stream, sem->dev_sem_val_ptr, value, 0);
    if (res != CUDA_SUCCESS) {
        cudaError_t err = cudaGetLastError();
        ucc_error("cuStreamWriteValue32 failed with code: %d %s", res, cudaGetErrorString(err));
        ucc_assert(0);
        return;
    }
}

bool init_remote_semaphore(remote_semaphore_t* sem, int32_t* remote_val)
{
    assert(sem);
    assert(remote_val);
    sem->host_val_ptr = remote_val;
    CUresult res  = cuMemHostRegister(sem->host_val_ptr, sizeof(int32_t),
                                     CU_MEMHOSTREGISTER_DEVICEMAP);
    if (res != CUDA_SUCCESS) {
        ucc_error("Failed to cuMemHostRegister!");
        ucc_assert(0);
        return false;
    }
    res = cuMemHostGetDevicePointer(&sem->dev_sem_val_ptr, sem->host_val_ptr,
                                    0 /* flags */);
    if (res != CUDA_SUCCESS) {
        ucc_error("cuMemHostGetDevicePointer init_remote_semaphore failed!");
        ucc_assert(0);
        return false;
    }
    return true;
}

bool wait_remote_semaphore(cudaStream_t stream, remote_semaphore_t* sem, int32_t value)
{
    ucc_assert(sem != NULL);
    ucc_assert(sem->dev_sem_val_ptr);
    CUresult res = cuStreamWaitValue32(stream, sem->dev_sem_val_ptr, value,
                                       CU_STREAM_WAIT_VALUE_EQ);
    if (res != CUDA_SUCCESS) {
        ucc_error("cuStreamWaitValue32 failed for remote_semaphore_t with code: %d !", res);
        ucc_assert(0);
        return false;
    }
    return true;
}