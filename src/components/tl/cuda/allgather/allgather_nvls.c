/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv/allgatherv.h"
#include "allgather/allgather.h"

#include "tl_cuda_nvls.h"
#include "components/ec/ucc_ec.h"

ucc_status_t ucc_tl_cuda_allgather_nvls_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     tl_team,
                                             ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team    = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          trank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize   = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt      = coll_args->args.dst.info.datatype;
    size_t              dt_size = ucc_dt_size(dt);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;
    size_t              total_count;
    size_t              count_per_rank;
    size_t              total_count_bytes;
    size_t              offset_bytes;
    size_t              count_bytes;

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_ec_create_event(
        &task->allgatherv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create CUDA event");
        ucc_tl_cuda_task_put(task);
        return status;
    }

    /* For allgather, count is evenly distributed across all ranks */
    total_count     = coll_args->args.dst.info.count;
    count_per_rank  = total_count / tsize;
    offset_bytes    = trank * count_per_rank * dt_size;
    count_bytes     = count_per_rank * dt_size;
    total_count_bytes = total_count * dt_size;

    /* Validate total size fits within NVLS symmetric buffer */
    if (ucc_unlikely(
            total_count_bytes >
            UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS allgather total size %zu bytes exceeds symmetric buffer "
            "size %zu bytes",
            total_count_bytes,
            UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size);
        goto err_cleanup;
    }

    /* Convert bytes to uint32_t units for the kernel */
    task->allgatherv_nvls.offset      = offset_bytes / sizeof(uint32_t);
    task->allgatherv_nvls.count       = count_bytes / sizeof(uint32_t);
    task->allgatherv_nvls.total_count = total_count_bytes / sizeof(uint32_t);

    /* NVLS requires 16-byte alignment (4 uint32_t elements).
     * For allgather, all ranks have the same count, so if count is aligned,
     * all offsets (0, count, 2*count, ...) are automatically aligned. */
    if (ucc_unlikely(task->allgatherv_nvls.count % 4 != 0)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS requires 16-byte alignment for count, got count=%zu bytes "
            "(not aligned to 16 bytes). count_per_rank=%zu dt_size=%zu",
            count_bytes,
            count_per_rank,
            dt_size);
        goto err_cleanup;
    }

    task->allgatherv_nvls.mc_va   = (CUdeviceptr)TASK_SYMMETRIC_MC(task);
    task->allgatherv_nvls.uc_va   = (CUdeviceptr)TASK_SYMMETRIC_UC(task);
    task->allgatherv_nvls.coll_id = team->nvls.coll_ids[task->coll_id]++;

    /* Reuse allgatherv_nvls functions - they handle both coll types */
    task->super.post              = ucc_tl_cuda_allgatherv_nvls_start;
    task->super.triggered_post    = ucc_tl_cuda_allgatherv_nvls_triggered_post;
    task->super.progress          = ucc_tl_cuda_allgatherv_nvls_progress;
    task->super.finalize          = ucc_tl_cuda_allgatherv_nvls_finalize;

    *task_p                       = &task->super;
    return UCC_OK;

err_cleanup:
    ucc_ec_destroy_event(
        task->allgatherv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    ucc_tl_cuda_task_put(task);
    return UCC_ERR_NOT_SUPPORTED;
}
