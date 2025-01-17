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
    STAGE_PREPARE,
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

// Tests if setup is complete for a copy engine broadcast task
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
        if (ucc_atomic_cswap64(&curr_bar->tag, UCC_TL_CUDA_TAG_FREE,
                               task->bcast_ce.key) == UCC_TL_CUDA_TAG_FREE) {
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

static ucc_status_t prepare_commands(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_tl_cuda_sync_t *sync   = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize  = UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))
                                     ? (ucc_rank_t)task->subset.map.ep_num
                                     : UCC_TL_TEAM_SIZE(team);
    cudaStream_t        stream = task->bcast_ce.stream;
    size_t              scratch_size = get_raw_scratch_size(team);
    void               *scratch_root = TASK_SCRATCH(task, task->bcast_ce.root);
    ucc_status_t        status;
    int i, step, peer;

    CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&task->bcast_ce.evtCompletion,
                                             cudaEventDisableTiming),
                    exit_err, status);
    // TODO: do remap for active set
    if (trank == task->bcast_ce.root) {
        ucc_print("hello from root [%d] / tsize = %d", trank, tsize);
        // root
        // wait for peer to send
        for (step = 0; step < task->bcast_ce.num_steps; ++step) {
            size_t chunk_size =
                step < task->bcast_ce.num_steps
                    ? ucc_min(scratch_size, task->bcast_ce.size)
                    : task->bcast_ce.size -
                          (step - 1) * scratch_size;

            for (i = 0; i < tsize; ++i) {
                if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
                    // eval phys rank from virt
                    peer = ucc_ep_map_eval(task->subset.map, i);
                } else {
                    peer = i;
                }
                if (peer == trank) {
                    continue;
                }
                wait_remote_semaphore(stream, &sync->data[peer].remote_semaphore,
                                      step);
            }
            CUDA_CHECK_GOTO(cudaMemcpyAsync(scratch_root,
                                            PTR_OFFSET(task->bcast_ce.sbuf,
                                                       step * scratch_size),
                                            chunk_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_err, status);
            // root ready to send event
            set_val_semaphore(stream, &sync->semaphore, step);
        }
        // wait all peers for completion of theirs copy
        for (i = 0; i < tsize; ++i) {
            if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
                // eval phys rank from virt
                peer = ucc_ep_map_eval(task->subset.map, i);
            } else {
                peer = i;
            }
            if (peer == trank) {
                continue;
            }
            wait_remote_semaphore(stream, &sync->data[peer].remote_semaphore,
                                  task->bcast_ce.num_steps);
        }
        set_val_semaphore(stream, &sync->semaphore, -1); // TODO: wait values more than step
        // for tracking stream execution
        CUDA_CHECK_GOTO(cudaEventRecord(task->bcast_ce.evtCompletion, stream),
                        exit_err, status);
    } else {
        // peer scenario
        ucc_print("hello from peer [%d] / tsize = %d", trank, tsize);

        for (step = 0; step < task->bcast_ce.num_steps; ++step) {
            size_t chunk_size =
                step < task->bcast_ce.num_steps
                    ? ucc_min(scratch_size, task->bcast_ce.size)
                    : task->bcast_ce.size -
                          (step - 1) * scratch_size;

            // 1 notify root that peer is ready to read data
            set_val_semaphore(stream, &sync->semaphore, step);
            // wait while root places its chunk of data to scratch
            wait_remote_semaphore(
                stream, &sync->data[task->bcast_ce.root].remote_semaphore,
                step);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(task->bcast_ce.sbuf,
                                                       step * scratch_size),
                                            scratch_root, chunk_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_err, status);
        }
        // place event to signal completion
        set_val_semaphore(stream, &sync->semaphore, task->bcast_ce.num_steps);
        // for tracking stream execution
        CUDA_CHECK_GOTO(cudaEventRecord(task->bcast_ce.evtCompletion, stream),
                        exit_err, status);

        wait_remote_semaphore(
            stream, &sync->data[task->bcast_ce.root].remote_semaphore, -1);
    }

    return UCC_OK;

exit_err:
    ucc_error("error prepare_commands!");
    ucc_assert(0);
    return status;
}

static void ucc_tl_cuda_bcast_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t st;

    task->super.status = UCC_INPROGRESS;

    switch (task->bcast_ce.stage) {
    case STAGE_INIT_BAR_ROOT:
        st = root_find_free_barrier(task);
        if (st == UCC_OK) {
            task->bcast_ce.stage = STAGE_SYNC;
        } else if (st != UCC_ERR_NOT_FOUND) {
            task->super.status = st;
        }
        // no free barriers found, try next time
        return;
    case STAGE_FIND_BAR_PEER:
        st = peer_find_free_barrier(task);
        if (st == UCC_OK) {
            // barrier found, continue to next stages
            task->bcast_ce.stage = STAGE_SYNC;
        } else if (st != UCC_ERR_NOT_FOUND) {
            task->super.status = st;
        }
        // no free barriers found by root, try next time
        return;
    case STAGE_SYNC:
        if (ucc_tl_cuda_get_sync_root(task, task->bcast_ce.root) != UCC_OK) {
            return;
        }
        st = ucc_tl_cuda_bcast_ce_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->bcast_ce.stage = STAGE_SETUP;
    case STAGE_SETUP:
        st = ucc_tl_cuda_bcast_ce_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->bcast_ce.stage = STAGE_WAIT_COPY;
    default:
        break;
    }

    switch (task->bcast_ce.stage) {
        case STAGE_WAIT_COPY:
            cudaError_t status = cudaEventQuery(task->bcast_ce.evtCompletion);
            if (status == cudaSuccess)
            {
                ucc_print("cuda stage finished");
                task->bcast_ce.stage = STAGE_WAIT_ALL;
            } else if (status == cudaErrorNotReady) {
                // if (trank == task->bcast_ce.root) {
                    // int32_t val_host = *TASK_SYNC(task, UCC_TL_TEAM_RANK(team))->data[1].remote_semaphore.host_val_ptr;
                    // int32_t my_val_host = TASK_SYNC(task, UCC_TL_TEAM_RANK(team))->semaphore.host_val;
                    // ucc_print("remote semaphore val: %d, local: %d", val_host, my_val_host);
                // }

                // still in progress
                task->super.status = UCC_INPROGRESS;
                return;
            } else {
                ucc_error("error cudaEventQuery %d!", status);
                task->super.status = UCC_ERR_NO_MESSAGE;
                ucc_assert(0);
                return;
            }
            break;
        case STAGE_WAIT_ALL:
            // finish
            st = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to start barrier");
                task->super.status = st;
                return;
            }
            task->bcast_ce.stage = STAGE_WAIT_COMPLETION;
        case STAGE_WAIT_COMPLETION:
            st = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
            if (st != UCC_OK) {
                // peers still working, lets check next time
                task->super.status = st;
                return;
            }
            if (trank == task->bcast_ce.root) {
                // set barrier free to unlock others, this is roots responsibility
                ucc_debug("Release bar: %p with tag: %ld", task->bar,
                        task->bar->tag);
                task->bar->tag = UCC_TL_CUDA_TAG_FREE;
                ucc_tl_cuda_put_sync_root(task, task->bcast_ce.root);
            }

            // ucc_tl_cuda_sync_t *sync   = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
            // set_val_semaphore(task->bcast_ce.stream, &sync->semaphore, -1); // Init but no ready
            // cudaEventDestroy(task->bcast_ce.evtCompletion);

            task->super.status = UCC_OK;
            break;
        default:
            ucc_assert(0);
            break;
        }
    return;
}

static ucc_status_t ucc_bcast_ce_post_with_stream(cudaStream_t stream, ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_datatype_t      dt   = task->bcast_ce.dt;
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));

    task->bcast_ce.stream = stream;
    task->bcast_ce.stage  = STAGE_SYNC;

    set_val_semaphore(task->bcast_ce.stream, &sync->semaphore, -1); // Init but no ready

    // in case of active set bcast we need to do additional steps to find free barriers
    if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
        task->bcast_ce.stage = UCC_TL_TEAM_RANK(team) == task->bcast_ce.root
                                   ? STAGE_INIT_BAR_ROOT
                                   : STAGE_FIND_BAR_PEER;
    }

    task->bcast_ce.size = ucc_dt_size(dt) * args->src.info.count;
    task->bcast_ce.num_steps = ucc_div_round_up(task->bcast_ce.size, get_raw_scratch_size(team));
    task->bcast_ce.sbuf = args->src.info.buffer;

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

    task->super.post           = ucc_bcast_ce_post;
    task->super.triggered_post = ucc_bcast_ce_triggered_post;
    task->super.progress = ucc_tl_cuda_bcast_ce_progress;
    task->super.finalize = ucc_tl_cuda_bcast_ce_finalize;

    *task_p = &task->super;
    return UCC_OK;
}
