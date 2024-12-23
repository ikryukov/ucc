/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"

#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"


enum {
    // Barrier setup stages
    STAGE_INIT_BAR_ROOT,            // Initial stage for the root rank to identify and claim a free barrier
    STAGE_FIND_BAR_PEER,            // Stage where peer ranks wait while the root rank identifies a free barrier

    STAGE_SYNC,                     // Initialize the barrier and synchronize the segment required for the current task
    STAGE_SETUP,                    // Verify that all ranks are aligned and have reached the barrier
    // Stages specific to the root rank
    STAGE_COPY,                     // Post copy task: copy data block from src to a scratch buffer
    STAGE_WAIT_COPY,                // The root waits for the completion of its copy operation
    STAGE_WAIT_ALL,                 // The root rank waits until all other ranks have reached the same operational step
    STAGE_WAIT_COMPLETION,          // The root rank waits for all other ranks to complete the broadcast operation
    // non-root
    STAGE_WAIT_ROOT,                // Wait while the root rank writes data to its scratch buffer
    STAGE_CLIENT_COPY,              // Initiate their own copy tasks after the root's operations
    STAGE_CLIENT_COPY_WAIT,         // Wait for the completion of the copy operation from the root's scratch buffer
    STAGE_CLIENT_WAIT_COMPLETION,   // Wait for the completion of algorithm on all ranks, global sync with root
};

static inline ucc_status_t ucc_tl_cuda_bcast_ce_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;

    set_rank_step(task, trank, 0, 0); // Initialize rank step tracking
    ucc_memory_cpu_store_fence();
    // initiate barrier wait while all ranks set theirs steps to 0
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

// Tests if setup is complete for a linear broadcast task
static inline ucc_status_t ucc_tl_cuda_bcast_ce_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

// Returns the size of the scratch buffer used for data transfers
static inline size_t get_raw_scratch_size(ucc_tl_cuda_team_t *team)
{
    return UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
}

// Posts a copy task to the CUDA executor
static inline ucc_status_t ecopy(void *dst, void *src, size_t size,
                                 ucc_ee_executor_t       *exec,
                                 ucc_ee_executor_task_t **etask)
{
    ucc_ee_executor_task_args_t exec_args = {0};

    exec_args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    exec_args.copy.dst  = dst;
    exec_args.copy.src  = src;
    exec_args.copy.len  = size;
    return ucc_ee_executor_task_post(exec, &exec_args, etask);
}

// Root rank searches for and claims a free barrier
static inline ucc_status_t root_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    uint32_t max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;

    // Iterate over available barriers in active set pool to find a free one
    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        // try to set user specified tag to mark that this barrier is used by this task
        if (ucc_atomic_cswap64(&curr_bar->tag, UCC_TAG_FREE,
                               task->bcast_ce.key) == UCC_TAG_FREE) {
            ucc_debug("Acquire barrier: %p idx: %d marked with tag: %ld",
                      curr_bar, i, curr_bar->tag);
            task->bar = curr_bar;
            st        = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_ce.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init root barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            // Assign a collective ID (index of barrier)
            task->coll_id = i + max_concurrent;
            return UCC_OK;
        }
    }
    // try next time
    return UCC_ERR_NOT_FOUND;
}

// Peer rank searches for a barrier claimed by the root
static inline ucc_status_t peer_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    uint32_t max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;

    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        // Check if the barrier is claimed by the task's root
        if (curr_bar->tag == task->bcast_ce.key) {
            task->bar = curr_bar;
            st        = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_ce.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init peer barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            task->coll_id = i + max_concurrent;
            return UCC_OK;
        }
    }
    // try next time
    return UCC_ERR_NOT_FOUND;
}

static ucc_status_t ucc_tl_cuda_bcast_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

static void ucc_tl_cuda_bcast_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task              = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    task->super.status = UCC_INPROGRESS;

    cudaError_t status = cudaEventQuery(task->bcast_ce.evtCompletion);
    if (status == cudaErrorNotReady)
    {
        return;
    }

    task->super.status = UCC_OK;
    return;
}


static ucc_status_t prepare_commands(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_tl_cuda_sync_t *sync   = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    // ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize  = UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))
                                     ? (ucc_rank_t)task->subset.map.ep_num
                                     : UCC_TL_TEAM_SIZE(team);
    cudaStream_t        stream = task->bcast_ce.stream;
    ucc_status_t        status;
    // volatile ucc_tl_cuda_sync_t *peer_sync;
    int i;

    CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&task->bcast_ce.evtCompletion,
                                             cudaEventDisableTiming),
                    exit_err, status);
    if (trank == task->bcast_ce.root) {
        // root
        // wait for peer to send
        for (i = 0; i < tsize; ++i) {
            if (i == trank) {
                continue;
            }
            // peer_sync = TASK_SYNC(task, i);
            CUDA_CHECK_GOTO(
                cudaStreamWaitEvent(stream, sync->data[i].ipc_event_remote, 0),
                exit_err, status);
        }
        CUDA_CHECK_GOTO(cudaMemcpyAsync(TASK_SCRATCH(task, trank),
                                        task->bcast_ce.sbuf,
                                        task->bcast_ce.size,
                                        cudaMemcpyDeviceToDevice, stream),
                        exit_err, status);
        // root ready to send event
        CUDA_CHECK_GOTO(cudaEventRecord(sync->ipc_event_local, stream),
                        exit_err, status);
        // wait all peers for completion of theirs copy
        for (i = 0; i < tsize; ++i) {
            if (i == trank) {
                continue;
            }
            // peer_sync = TASK_SYNC(task, i);
            CUDA_CHECK_GOTO(
                cudaStreamWaitEvent(stream, sync->data[i].ipc_event_remote, 0),
                exit_err, status);
        }
    } else {
        // peer scenario
        // 1 notify root that peer is ready to read data
        CUDA_CHECK_GOTO(cudaEventRecord(sync->ipc_event_local, stream),
                        exit_err, status);
        // wait while root places its chunk of data to scratch
        CUDA_CHECK_GOTO(
            cudaStreamWaitEvent(
                stream, sync->data[task->bcast_ce.root].ipc_event_remote, 0),
            exit_err, status);
        CUDA_CHECK_GOTO(
            cudaMemcpyAsync(task->bcast_ce.sbuf,
                            TASK_SCRATCH(task, task->bcast_ce.root),
                            task->bcast_ce.size, cudaMemcpyDeviceToDevice,
                            stream),
            exit_err, status);
        // place event to signal completion
        CUDA_CHECK_GOTO(cudaEventRecord(sync->ipc_event_local, stream),
                        exit_err, status);
    }

    // for tracking stream execution
    CUDA_CHECK_GOTO(cudaEventRecord(task->bcast_ce.evtCompletion, stream),
                    exit_err, status);
    return UCC_OK;

exit_err:
    return status;
}

static ucc_status_t ucc_bcast_ce_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_datatype_t      dt   = task->bcast_ce.dt;
    size_t              half_scratch_size = get_raw_scratch_size(team) / 2;

    task->bcast_ce.stream  = team->stream;

    task->bcast_ce.stage = STAGE_SYNC;

    // in case of active set bcast we need to do additional steps to find free barriers
    if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
        task->bcast_ce.stage = UCC_TL_TEAM_RANK(team) == task->bcast_ce.root ? STAGE_INIT_BAR_ROOT : STAGE_FIND_BAR_PEER;
    }
    
    task->bcast_ce.size = ucc_dt_size(dt) * args->src.info.count;
    task->bcast_ce.num_steps =
        ucc_div_round_up(task->bcast_ce.size, half_scratch_size);

    ucc_debug("bcast linear dt: %s, buffer size: %ld, num_steps: %d",
              ucc_datatype_str(dt), task->bcast_ce.size,
              task->bcast_ce.num_steps);

    task->bcast_ce.sbuf = args->src.info.buffer;
    task->bcast_ce.step = 0;

    ucc_status_t st = prepare_commands(task);
    if (st != UCC_OK)
    {
        return st;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

static ucc_status_t ucc_bcast_ce_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    ucc_assert(ee != NULL); // ensure contract

    task->bcast_ce.stream = (cudaStream_t) ee->ee_context;
    
    return UCC_OK; // TODO: just stub
}

ucc_status_t ucc_tl_cuda_bcast_ce_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *tl_team,
                                           ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    ucc_print("bcast ce init");

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
                     UCC_TL_TEAM_SIZE(team) - 1 >
                         UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->bcast_ce.root = coll_args->args.root;
    task->bcast_ce.dt   = coll_args->args.src.info.datatype;
    task->bcast_ce.sbuf = coll_args->args.src.info.buffer;

    task->super.flags         |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_bcast_ce_post;
    task->super.triggered_post = ucc_bcast_ce_triggered_post;

    task->super.progress = ucc_tl_cuda_bcast_ce_progress;
    task->super.finalize = ucc_tl_cuda_bcast_ce_finalize;

    *task_p = &task->super;
    return UCC_OK;
}
