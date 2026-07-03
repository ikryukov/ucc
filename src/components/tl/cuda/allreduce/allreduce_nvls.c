/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce/allreduce.h"
#include "ucc/api/ucc.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"
#include "tl_cuda_nvls.h"
#include "../kernels/allreduce_kernel.h"
#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"

enum {
    STAGE_KERNEL, /*< Post memcpy to symmetric buffer, launch kernel, memcpy to destination */
    STAGE_WAIT, /*< Wait for the copies and kernel to complete */
};

ucc_status_t ucc_tl_cuda_allreduce_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_ee_h            ee   = task->super.ee;
    cudaStream_t stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    // stream is used in tl_trace below, but not used in this function
    (void)stream;

    task->allreduce_nvls.buf_size_bytes = args->dst.info.count *
                                          ucc_dt_size(task->allreduce_nvls.dt);
    /* Pad to 16*tsize so chunk_start = count_u32*rank/tsize is 4-aligned for
     * the MULTIMEM v4.f32 16-byte accesses in vec32 kernels. Padding bytes are
     * uninitialized and never copied back to the user buffer. */
    task->allreduce_nvls.kernel_size_bytes = ucc_align_up(
        task->allreduce_nvls.buf_size_bytes, 16 * UCC_TL_TEAM_SIZE(team));
    task->allreduce_nvls.rbuf = args->dst.info.buffer;
    task->allreduce_nvls.sbuf = UCC_IS_INPLACE(*args) ? args->dst.info.buffer
                                                      : args->src.info.buffer;

    tl_trace(
        UCC_TASK_LIB(task),
        "task: %p stream: %p allreduce_nvls_start symmetric uc addr: %p "
        "mc addr: %p "
        "buf_size_bytes: %zu, is inplace: %d",
        task,
        stream,
        (void *)task->allreduce_nvls.uc_va,
        (void *)task->allreduce_nvls.mc_va,
        task->allreduce_nvls.buf_size_bytes,
        UCC_IS_INPLACE(*args));

    task->allreduce_nvls.stage = STAGE_KERNEL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

/* Large-message allreduce with copy/reduce overlap: the message is split into
 * chunks; copy-in runs on s_in and copy-out on s_out (separate streams) so the
 * HBM staging copies overlap the NVLink-bound per-chunk partitioned reduce on
 * rstream. Ephemeral streams/events are used so it is safe under concurrent
 * collectives (CUDA defers their destruction until the work completes). */
static ucc_status_t ucc_tl_cuda_allreduce_nvls_pipeline(
    ucc_tl_cuda_team_t *team, void *sbuf, void *rbuf, CUdeviceptr mc_va,
    CUdeviceptr uc_va, CUdeviceptr mc_ctrl, CUdeviceptr uc_ctrl,
    size_t buf_size, uint32_t sm_count, uint32_t threads, ucc_rank_t trank,
    ucc_datatype_t dt, cudaStream_t rstream)
{
    uint32_t     tsize = UCC_TL_TEAM_SIZE(team);
    size_t       align = (size_t)16 * tsize;
    cudaStream_t s_in  = NULL;
    cudaStream_t s_out = NULL;
    cudaEvent_t  ev_in[UCC_TL_CUDA_NVLS_MAX_PIPE_CHUNKS]  = {0};
    cudaEvent_t  ev_red[UCC_TL_CUDA_NVLS_MAX_PIPE_CHUNKS] = {0};
    cudaEvent_t  ev_out[UCC_TL_CUDA_NVLS_MAX_PIPE_CHUNKS] = {0};
    ucc_status_t status = UCC_OK;
    size_t       chunk, off;
    int          nchunks, k, c;

    k = (int)(buf_size / UCC_TL_CUDA_NVLS_PIPE_THRESH);
    if (k < 2) {
        k = 2;
    }
    if (k > UCC_TL_CUDA_NVLS_MAX_PIPE_CHUNKS) {
        k = UCC_TL_CUDA_NVLS_MAX_PIPE_CHUNKS;
    }
    chunk   = ((buf_size / k) + align - 1) / align * align;
    nchunks = (int)((buf_size + chunk - 1) / chunk);

    status = CUDA_FUNC(cudaStreamCreateWithFlags(&s_in, cudaStreamNonBlocking));
    if (status != UCC_OK) {
        goto out;
    }
    status = CUDA_FUNC(cudaStreamCreateWithFlags(&s_out, cudaStreamNonBlocking));
    if (status != UCC_OK) {
        goto out;
    }
    for (c = 0; c < nchunks; c++) {
        if (CUDA_FUNC(cudaEventCreateWithFlags(
                &ev_in[c], cudaEventDisableTiming)) != UCC_OK ||
            CUDA_FUNC(cudaEventCreateWithFlags(
                &ev_red[c], cudaEventDisableTiming)) != UCC_OK ||
            CUDA_FUNC(cudaEventCreateWithFlags(
                &ev_out[c], cudaEventDisableTiming)) != UCC_OK) {
            status = UCC_ERR_NO_RESOURCE;
            goto out;
        }
    }

    /* copy-in chunks (run ahead on s_in, overlapping the reduces) */
    for (c = 0, off = 0; c < nchunks; c++) {
        size_t csz = (buf_size - off < chunk) ? (buf_size - off) : chunk;
        status = CUDA_FUNC(cudaMemcpyAsync((void *)(uc_va + off),
                                           PTR_OFFSET(sbuf, off), csz,
                                           cudaMemcpyDeviceToDevice, s_in));
        if (status != UCC_OK) {
            goto out;
        }
        status = CUDA_FUNC(cudaEventRecord(ev_in[c], s_in));
        if (status != UCC_OK) {
            goto out;
        }
        off += csz;
    }
    /* partitioned reduce per chunk on rstream, each gated on its copy-in */
    for (c = 0, off = 0; c < nchunks; c++) {
        size_t csz = (buf_size - off < chunk) ? (buf_size - off) : chunk;
        status = CUDA_FUNC(cudaStreamWaitEvent(rstream, ev_in[c], 0));
        if (status != UCC_OK) {
            goto out;
        }
        status = post_allreduce_kernel(rstream, sm_count, threads, mc_va + off,
                                       csz, mc_ctrl, uc_ctrl, trank, tsize, dt);
        if (status != UCC_OK) {
            goto out;
        }
        status = CUDA_FUNC(cudaEventRecord(ev_red[c], rstream));
        if (status != UCC_OK) {
            goto out;
        }
        off += csz;
    }
    /* copy-out chunks on s_out, each gated on its reduce (overlaps reduces) */
    for (c = 0, off = 0; c < nchunks; c++) {
        size_t csz = (buf_size - off < chunk) ? (buf_size - off) : chunk;
        status = CUDA_FUNC(cudaStreamWaitEvent(s_out, ev_red[c], 0));
        if (status != UCC_OK) {
            goto out;
        }
        status = CUDA_FUNC(cudaMemcpyAsync(PTR_OFFSET(rbuf, off),
                                           (void *)(uc_va + off), csz,
                                           cudaMemcpyDeviceToDevice, s_out));
        if (status != UCC_OK) {
            goto out;
        }
        status = CUDA_FUNC(cudaEventRecord(ev_out[c], s_out));
        if (status != UCC_OK) {
            goto out;
        }
        off += csz;
    }
    /* rstream (on which the caller records the completion event) waits for the
     * last copy-out; s_out is ordered so this covers all chunks. */
    status = CUDA_FUNC(cudaStreamWaitEvent(rstream, ev_out[nchunks - 1], 0));

out:
    if (s_in) {
        cudaStreamDestroy(s_in);
    }
    if (s_out) {
        cudaStreamDestroy(s_out);
    }
    for (c = 0; c < nchunks; c++) {
        if (ev_in[c]) {
            cudaEventDestroy(ev_in[c]);
        }
        if (ev_red[c]) {
            cudaEventDestroy(ev_red[c]);
        }
        if (ev_out[c]) {
            cudaEventDestroy(ev_out[c]);
        }
    }
    return status;
}

void ucc_tl_cuda_allreduce_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team  = TASK_TEAM(task);
    ucc_rank_t           trank = UCC_TL_TEAM_RANK(team);
    ucc_ec_cuda_event_t *ec_event = (ucc_ec_cuda_event_t *)
                                        task->allreduce_nvls.evt_completion;
    cudaEvent_t    evt    = ec_event->event;
    CUdeviceptr    mc_va  = task->allreduce_nvls.mc_va;
    CUdeviceptr    uc_va  = task->allreduce_nvls.uc_va;
    ucc_ee_h       ee     = task->super.ee;
    cudaStream_t   stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    ucc_datatype_t dt     = task->allreduce_nvls.dt;
    uint32_t       sm_count = ucc_tl_cuda_nvls_sm_count(
        task->allreduce_nvls.buf_size_bytes,
        UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count);
    uint32_t       threads  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;

    ucc_status_t   status;
    cudaError_t    cuda_status;

    switch (task->allreduce_nvls.stage) {
    case STAGE_KERNEL:
        if (task->allreduce_nvls.buf_size_bytes <= UCC_TL_CUDA_NVLS_LL_THRESH) {
            /* Small messages: latency-optimized path (in-kernel copy +
             * multicast reduce, no host-side staging copies). */
            status = post_allreduce_lowlatency_kernel(
                stream,
                threads,
                (CUdeviceptr)task->allreduce_nvls.sbuf,
                (CUdeviceptr)task->allreduce_nvls.rbuf,
                mc_va,
                uc_va,
                task->allreduce_nvls.buf_size_bytes,
                TASK_NVLS_CONTROL_MC(task),
                TASK_NVLS_CONTROL_UC(task),
                UCC_TL_TEAM_SIZE(team),
                dt);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "failed to post allreduce lowlatency kernel");
                task->super.status = status;
                return;
            }
        } else if (task->allreduce_nvls.buf_size_bytes >=
                       UCC_TL_CUDA_NVLS_PIPE_THRESH &&
                   (task->allreduce_nvls.buf_size_bytes %
                    (16 * UCC_TL_TEAM_SIZE(team))) == 0) {
            /* Large messages: overlap the staging copies with the partitioned
             * reduce (chunked, separate copy streams). */
            status = ucc_tl_cuda_allreduce_nvls_pipeline(
                team,
                task->allreduce_nvls.sbuf,
                task->allreduce_nvls.rbuf,
                mc_va,
                uc_va,
                TASK_NVLS_CONTROL_MC(task),
                TASK_NVLS_CONTROL_UC(task),
                task->allreduce_nvls.buf_size_bytes,
                sm_count,
                threads,
                trank,
                dt,
                stream);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "failed to post pipelined allreduce");
                task->super.status = status;
                return;
            }
        } else {
            /* Copy the user source into the symmetric buffer. */
            status = CUDA_FUNC(cudaMemcpyAsync(
                (void *)uc_va,
                task->allreduce_nvls.sbuf,
                task->allreduce_nvls.buf_size_bytes,
                cudaMemcpyDeviceToDevice,
                stream));
            if (status != UCC_OK) {
                task->super.status = status;
                return;
            }
            /* kernel_size_bytes is padded to 16*tsize so per-rank chunks are
             * 16-byte aligned for the MULTIMEM v4 accesses; padding is never
             * copied back to the user buffer. */
            status = post_allreduce_kernel(
                stream,
                sm_count,
                threads,
                mc_va,
                task->allreduce_nvls.kernel_size_bytes,
                TASK_NVLS_CONTROL_MC(task),
                TASK_NVLS_CONTROL_UC(task),
                trank,
                UCC_TL_TEAM_SIZE(team),
                dt);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to post allreduce kernel");
                task->super.status = status;
                return;
            }
            /* Copy back only the user data range. */
            status = CUDA_FUNC(cudaMemcpyAsync(
                (void *)task->allreduce_nvls.rbuf,
                (void *)uc_va,
                task->allreduce_nvls.buf_size_bytes,
                cudaMemcpyDeviceToDevice,
                stream));
            if (status != UCC_OK) {
                task->super.status = status;
                return;
            }
        }
        status = CUDA_FUNC(cudaEventRecord(evt, stream));
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->allreduce_nvls.stage = STAGE_WAIT;
        // fallthrough
    case STAGE_WAIT:
        cuda_status = cudaEventQuery(evt);
        if (cuda_status == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        task->super.status = UCC_OK;
        break;
    }
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(tl_task), "task: %p allreduce_nvls_finalize", task);

    ucc_ec_destroy_event(
        tl_task->allreduce_nvls.evt_completion, UCC_EE_CUDA_STREAM);

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

// NOLINTNEXTLINE(misc-unused-parameters): ev parameter unused as it's not needed for this implementation
ucc_status_t ucc_tl_cuda_allreduce_nvls_triggered_post(
    ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_trace(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);

    task->allreduce_nvls.stage = STAGE_KERNEL;

    status                     = coll_task->post(coll_task);
    if (ucc_likely(status == UCC_OK)) {
        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.ev_context      = NULL;
        post_event.req             = &coll_task->super;
        status                     = ucc_ee_set_event_internal(
            coll_task->ee, &post_event, &coll_task->ee->event_out_queue);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task),
                "failed to set EE event: %s",
                ucc_status_string(status));
            return status;
        }
    }
    return status;
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team     = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    size_t              buf_size = coll_args->args.dst.info.count *
                      ucc_dt_size(coll_args->args.dst.info.datatype);
    size_t              tsize       = UCC_TL_TEAM_SIZE(team);
    size_t              kernel_size = ucc_align_up(buf_size, 16 * tsize);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    /* kernel_size must be a multiple of 16*tsize; see allreduce_nvls_start. */
    if (buf_size == 0 || coll_args->args.op != UCC_OP_SUM ||
        !ucc_tl_cuda_allreduce_nvls_dt_supported(
            coll_args->args.dst.info.datatype)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS allreduce is supported only with SUM operation "
            "and float32, bfloat16, int32, uint32, int64, or uint64 "
            "datatype, with non-zero message size");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(kernel_size >
                     UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS allreduce padded buffer size %zu bytes exceeds "
            "symmetric buffer size %zu bytes",
            kernel_size,
            UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to initialize CUDA task");
        return status;
    }

    status = ucc_ec_create_event(
        &task->allreduce_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create CUDA event");
        ucc_tl_cuda_task_put(task);
        return status;
    }

    task->allreduce_nvls.dt = coll_args->args.dst.info.datatype;
    tl_trace(
        UCC_TL_TEAM_LIB(team),
        "NVLS allreduce datatype: %s",
        ucc_datatype_str(task->allreduce_nvls.dt));

    task->super.post             = ucc_tl_cuda_allreduce_nvls_start;
    task->super.triggered_post   = ucc_tl_cuda_allreduce_nvls_triggered_post;
    task->super.progress         = ucc_tl_cuda_allreduce_nvls_progress;
    task->super.finalize         = ucc_tl_cuda_allreduce_nvls_finalize;

    task->allreduce_nvls.uc_va   = (CUdeviceptr)TASK_SYMMETRIC_UC(task);
    task->allreduce_nvls.mc_va   = (CUdeviceptr)TASK_SYMMETRIC_MC(task);

    *task_p                      = &task->super;
    return UCC_OK;
}
