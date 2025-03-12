/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"

#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"

#include <nvtx3/nvToolsExt.h>

// Returns the size of the scratch buffer used for data transfers
static inline size_t get_raw_scratch_size(ucc_tl_cuda_team_t *team)
{
    return UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_data_size;
}

static ucc_status_t ucc_tl_cuda_bcast_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

static ucc_status_t prepare_commands(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    // TODO: we can't take sync here because it uses task->coll_id to navigate in shared mem 
    ucc_assert(task->coll_id == 0);
    ucc_tl_cuda_sync_t *sync   = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    ucc_assert(sync != NULL);
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize  = UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))
                                     ? (ucc_rank_t)task->subset.map.ep_num
                                     : UCC_TL_TEAM_SIZE(team);
    cudaStream_t        stream = task->bcast_ce.stream;
    size_t              scratch_size = get_raw_scratch_size(team);
    void               *scratch_root = TASK_SCRATCH(task, task->bcast_ce.root);
    ucc_status_t        status;
    int i, step, peer;

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = trank == task->bcast_ce.root ? "Bcast CE Range: send" : "Bcast CE Range: recv";

    nvtxRangeId_t rangeId = nvtxRangeStartEx(&eventAttrib);
    task->bcast_ce.profilerId = rangeId;

    CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&task->bcast_ce.evtCompletion,
                                             cudaEventDisableTiming),
                    exit_err, status);

    if (trank == task->bcast_ce.root) {
        // root
        // find peer 
        for (i = 0; i < tsize; ++i)
        {
            if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
                // eval phys rank from virt
                peer = ucc_ep_map_eval(task->subset.map, i);
            } else {
                peer = i;
            }
            if (peer == trank) {
                continue;
            }
            break;
        }
        
        ucc_debug("hello from root [%d] peer [%d] / tsize = %d", trank, peer, tsize);

        remote_semaphore_t* iter_semaphore = &sync->remote_semaphores.iam_peer[peer][trank].iter_semaphore;
        remote_semaphore_t* done_semaphore = &sync->remote_semaphores.iam_peer[peer][trank].done_semaphore;

        stream_semaphore_t* peer_iter_semaphore = &sync->local_semaphores.iam_root[peer].iter_semaphore;

        // set_val_remote_semaphore(stream, iter_semaphore, -1); // reset

        CUstreamBatchMemOpParams batch_memops[3] = {};

        for (step = 0; step < task->bcast_ce.num_steps; ++step) {

            // wait for peer enter step            
            // wait_semaphore(stream, peer_iter_semaphore, step);

            // CUstreamBatchMemOpParams batch_memops[2] = {};
            batch_memops[0].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
            batch_memops[0].waitValue.address = peer_iter_semaphore->dev_sem_val_ptr;
            batch_memops[0].waitValue.value = step;
            batch_memops[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;

            if (step == 0) {
                // set_val_remote_semaphore(stream, done_semaphore, 0); // reset done sem
                batch_memops[1].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
                batch_memops[1].writeValue.address = done_semaphore->dev_sem_val_ptr;
                batch_memops[1].writeValue.value = 0;
            }

            cuStreamBatchMemOp(stream, step == 0 ? 2 : 1, batch_memops, 0);

            // copy
            size_t chunk_size = ucc_min(scratch_size, task->bcast_ce.size - step * scratch_size);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(scratch_root,
                                            PTR_OFFSET(task->bcast_ce.sbuf,
                                                       step * scratch_size),
                                            chunk_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_err, status);
            // root signals that it has placed its chunk of data to scratch
            set_val_remote_semaphore(stream, iter_semaphore, step);
        }

        // wait peer for completion of theirs copy
        // wait_semaphore(stream, peer_iter_semaphore, task->bcast_ce.num_steps);
        batch_memops[0].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        batch_memops[0].waitValue.address = peer_iter_semaphore->dev_sem_val_ptr;
        batch_memops[0].waitValue.value = task->bcast_ce.num_steps;
        batch_memops[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        // signal last step
        // set_val_remote_semaphore(stream, iter_semaphore, task->bcast_ce.num_steps);
        batch_memops[1].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        batch_memops[1].writeValue.address = iter_semaphore->dev_sem_val_ptr;
        batch_memops[1].writeValue.value = task->bcast_ce.num_steps;
        // set_val_remote_semaphore(stream, done_semaphore, 1); 
        batch_memops[2].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        batch_memops[2].writeValue.address = done_semaphore->dev_sem_val_ptr;
        batch_memops[2].writeValue.value = 1;

        cuStreamBatchMemOp(stream, 3, batch_memops, 0);

        // for tracking stream execution
        CUDA_CHECK_GOTO(cudaEventRecord(task->bcast_ce.evtCompletion, stream),
                        exit_err, status);
    } else {
        // peer scenario
        ucc_debug("hello from peer [%d] root [%d] / tsize = %d", trank, task->bcast_ce.root, tsize);
        remote_semaphore_t* iter_semaphore = &sync->remote_semaphores.iam_root[task->bcast_ce.root][trank].iter_semaphore;

        ucc_tl_cuda_local_semaphores_t* sem = &sync->local_semaphores.iam_peer[task->bcast_ce.root];
        stream_semaphore_t* root_iter_semaphore = &sem->iter_semaphore;
        stream_semaphore_t* root_done_semaphore = &sem->done_semaphore;

        CUstreamBatchMemOpParams batch_memops[2] = {};

        for (step = 0; step < task->bcast_ce.num_steps; ++step) {
            // 1 notify root that peer is ready to read data
            // set_val_remote_semaphore(stream, iter_semaphore, step);
            batch_memops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
            batch_memops[0].waitValue.address = iter_semaphore->dev_sem_val_ptr;
            batch_memops[0].waitValue.value = step;
            batch_memops[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;

            // wait while root places its chunk of data to scratch
            // wait_semaphore(stream, root_iter_semaphore, step);
            batch_memops[1].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
            batch_memops[1].writeValue.address = root_iter_semaphore->dev_sem_val_ptr;
            batch_memops[1].writeValue.value = step;
            
            cuStreamBatchMemOp(stream, 2, batch_memops, 0);

            size_t chunk_size = ucc_min(scratch_size, task->bcast_ce.size - step * scratch_size);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(task->bcast_ce.sbuf,
                                                       step * scratch_size),
                                            scratch_root, chunk_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_err, status);
        }
        // place event to signal completion
        // set_val_remote_semaphore(stream, iter_semaphore, task->bcast_ce.num_steps);
        batch_memops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        batch_memops[0].waitValue.address = iter_semaphore->dev_sem_val_ptr;
        batch_memops[0].waitValue.value = task->bcast_ce.num_steps;
        batch_memops[0].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        // wait root done sem
        // wait_semaphore(stream, root_done_semaphore, 1);
        batch_memops[1].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        batch_memops[1].writeValue.address = root_done_semaphore->dev_sem_val_ptr;
        batch_memops[1].writeValue.value = 1;

        cuStreamBatchMemOp(stream, 2, batch_memops, 0);
        // for tracking stream execution
        CUDA_CHECK_GOTO(cudaEventRecord(task->bcast_ce.evtCompletion, stream),
                        exit_err, status);
    }

    return UCC_OK;

exit_err:
    ucc_error("error prepare_commands!");
    ucc_assert(0);
    return status;
}

static void ucc_tl_cuda_bcast_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task        = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    cudaError_t         cuda_status = cudaEventQuery(task->bcast_ce.evtCompletion);

    if (cuda_status == cudaSuccess) {
        ucc_debug("cuda stage finished");
        ucc_status_t status;
        CUDA_CHECK_GOTO(cudaEventDestroy(task->bcast_ce.evtCompletion),
                        exit_err, status);
        nvtxRangeEnd(task->bcast_ce.profilerId); // End the range
        task->super.status = UCC_OK;
    } else if (cuda_status == cudaErrorNotReady) {
        // still in progress
        task->super.status = UCC_INPROGRESS;
    } else {
        ucc_error("error cudaEventQuery %s!", cudaGetErrorString(cuda_status));
        task->super.status = UCC_ERR_NO_MESSAGE;
        ucc_assert(0);
    }
    return;

exit_err:
    ucc_error("error ucc_tl_cuda_bcast_ce_progress!");
    ucc_assert(0);
}

static ucc_status_t ucc_bcast_ce_post_with_stream(cudaStream_t stream, ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_datatype_t      dt   = task->bcast_ce.dt;

    task->bcast_ce.stream = stream;

    ucc_debug("bcast ce dt: %s, buffer size: %ld, num_steps: %d",
              ucc_datatype_str(dt), task->bcast_ce.size,
              task->bcast_ce.num_steps);

    ucc_status_t st = prepare_commands(task);
    if (ucc_unlikely(st != UCC_OK)) {
        ucc_error("failed prepare_commands");
        task->super.status = st;
        return st;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

static ucc_status_t ucc_bcast_ce_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    return ucc_bcast_ce_post_with_stream(team->stream, coll_task);
}

static ucc_status_t ucc_bcast_ce_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_ev_t post_event;

    ucc_assert(ee != NULL); // ensure contract
    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    ucc_assert(ee->ee_context != NULL);

    coll_task->ee = ee;

    ucc_status_t status = ucc_bcast_ce_post_with_stream((cudaStream_t) ee->ee_context, coll_task);
    if (ucc_likely(status == UCC_OK)) {
        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.ev_context      = NULL;
        post_event.req             = &coll_task->super;
        ucc_ee_set_event_internal(coll_task->ee, &post_event,
                                  &coll_task->ee->event_out_queue);
    }
    return status;
}

ucc_status_t ucc_tl_cuda_bcast_ce_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *tl_team,
                                           ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_coll_args_t    *args = &coll_args->args;
    ucc_datatype_t      dt   = coll_args->args.src.info.datatype;
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    ucc_debug("bcast ce init");

    if (!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->bcast_ce.sbuf = args->src.info.buffer;
    task->bcast_ce.size = ucc_dt_size(dt) * args->src.info.count;
    task->bcast_ce.num_steps = ucc_div_round_up(task->bcast_ce.size, get_raw_scratch_size(team));

    task->bcast_ce.root = coll_args->args.root;
    task->bcast_ce.dt   = coll_args->args.src.info.datatype;

    task->super.post           = ucc_bcast_ce_post;
    task->super.triggered_post = ucc_bcast_ce_triggered_post;
    task->super.progress       = ucc_tl_cuda_bcast_ce_progress;
    task->super.finalize       = ucc_tl_cuda_bcast_ce_finalize;

    *task_p = &task->super;
    return UCC_OK;
}
