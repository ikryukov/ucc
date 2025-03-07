/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <iomanip>
#include "ucc_pt_benchmark.h"
#include "components/mc/ucc_mc.h"
#include "ucc_perftest.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_ee.h"

ucc_pt_benchmark::ucc_pt_benchmark(ucc_pt_benchmark_config cfg,
                                   ucc_pt_comm *communicator):
    config(cfg),
    comm(communicator)
{
    switch (cfg.op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
        coll = new ucc_pt_coll_allgather(cfg.dt, cfg.mt, cfg.inplace,
                                         cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_ALLGATHERV:
        coll = new ucc_pt_coll_allgatherv(cfg.dt, cfg.mt, cfg.inplace,
                                          cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_ALLREDUCE:
        coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                         cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_ALLTOALL:
        coll = new ucc_pt_coll_alltoall(cfg.dt, cfg.mt, cfg.inplace,
                                        cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_ALLTOALLV:
        coll = new ucc_pt_coll_alltoallv(cfg.dt, cfg.mt, cfg.inplace,
                                         cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_BARRIER:
        coll = new ucc_pt_coll_barrier(comm);
        break;
    case UCC_PT_OP_TYPE_BCAST:
        coll = new ucc_pt_coll_bcast(cfg.dt, cfg.mt, cfg.root_shift,
                                     cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_GATHER:
        coll = new ucc_pt_coll_gather(cfg.dt, cfg.mt, cfg.inplace,
                                      cfg.persistent, cfg.root_shift, comm);
        break;
    case UCC_PT_OP_TYPE_GATHERV:
        coll = new ucc_pt_coll_gatherv(cfg.dt, cfg.mt, cfg.inplace,
                                       cfg.persistent, cfg.root_shift, comm);
        break;
    case UCC_PT_OP_TYPE_REDUCE:
        coll = new ucc_pt_coll_reduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                      cfg.persistent, cfg.root_shift, comm);
        break;
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
        coll = new ucc_pt_coll_reduce_scatter(cfg.dt, cfg.mt, cfg.op,
                                              cfg.inplace,
                                              cfg.persistent, comm);
        break;
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
        coll = new ucc_pt_coll_reduce_scatterv(cfg.dt, cfg.mt, cfg.op,
                                               cfg.inplace, cfg.persistent,
                                               comm);
        break;
    case UCC_PT_OP_TYPE_SCATTER:
        coll = new ucc_pt_coll_scatter(cfg.dt, cfg.mt, cfg.inplace,
                                       cfg.persistent, cfg.root_shift, comm);
        break;
    case UCC_PT_OP_TYPE_SCATTERV:
        coll = new ucc_pt_coll_scatterv(cfg.dt, cfg.mt, cfg.inplace,
                                        cfg.persistent, cfg.root_shift, comm);
        break;
    case UCC_PT_OP_TYPE_MEMCPY:
        coll = new ucc_pt_op_memcpy(cfg.dt, cfg.mt, cfg.n_bufs, comm);
        break;
    case UCC_PT_OP_TYPE_REDUCEDT:
        coll = new ucc_pt_op_reduce(cfg.dt, cfg.mt, cfg.op, cfg.n_bufs, comm);
        break;
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
        coll = new ucc_pt_op_reduce_strided(cfg.dt, cfg.mt, cfg.op, cfg.n_bufs,
                                            comm);
        break;
    case UCC_PT_OP_TYPE_RING:
        coll = new ucc_pt_coll_ring(cfg.dt, cfg.mt, cfg.root_shift,
                                    cfg.persistent, comm);
        break;
    default:
        throw std::runtime_error("not supported collective");
    }
}

ucc_status_t ucc_pt_benchmark::run_bench() noexcept
{
    size_t min_count = coll->has_range() ? config.min_count : 1;
    size_t max_count = coll->has_range() ? config.max_count : 1;
    ucc_status_t       st;
    ucc_pt_test_args_t args[2]; // for ring we need to send to next and receive from prev - two collectives per rank
    double             time;

    print_header();
    for (size_t cnt = min_count; cnt <= max_count; cnt *= config.mult_factor) {
        size_t coll_size = cnt * ucc_dt_size(config.dt);
        int iter = config.n_iter_small;
        int warmup = config.n_warmup_small;
        if (coll_size >= config.large_thresh) {
            iter = config.n_iter_large;
            warmup = config.n_warmup_large;
        }
        args[0].coll_args.root = config.root;
        UCCCHECK_GOTO(coll->init_args(cnt, args[0]), exit_err, st);
        if ((uint64_t)config.op_type < (uint64_t)UCC_COLL_TYPE_LAST) {
            UCCCHECK_GOTO(run_single_coll_test(args[0].coll_args, warmup, iter, time),
                          free_coll, st);
        } else if (config.op_type == UCC_PT_OP_TYPE_RING) {
            // UCCCHECK_GOTO(coll->init_args(cnt, args[1]), exit_err, st);
            ucc_coll_args_t ring_args[2] = {};
            ucc_pt_coll_ring* coll_ring = dynamic_cast<ucc_pt_coll_ring*>(coll);
            UCCCHECK_GOTO(coll_ring->alloc_buffers(cnt, ring_args), exit_err, st);
            
            int rank_to_send = (comm->get_rank() + 1) % comm->get_size();
            int rank_from_recv = (comm->get_rank() + comm->get_size() - 1) % comm->get_size();
            // send
            ring_args[0].mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;
            ring_args[0].coll_type = UCC_COLL_TYPE_BCAST;
            ring_args[0].root = comm->get_rank();
            ring_args[0].active_set.size = 2;
            ring_args[0].active_set.start = comm->get_rank();
            ring_args[0].active_set.stride = rank_to_send - comm->get_rank(); // send to next rank
            ring_args[0].tag = (cnt + 1) % 777;

            // recv
            ring_args[1].mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;
            ring_args[1].coll_type = UCC_COLL_TYPE_BCAST;
            ring_args[1].root = rank_from_recv;
            ring_args[1].active_set.size = 2;
            ring_args[1].active_set.start = comm->get_rank();
            ring_args[1].active_set.stride = rank_from_recv - comm->get_rank(); // recv from prev rank
            ring_args[1].tag = (cnt + 1) % 777;
            
            UCCCHECK_GOTO(run_paired_coll_test(ring_args, warmup, iter, time), free_coll, st);

            coll_ring->free_buffers();
        } else {
            UCCCHECK_GOTO(run_single_executor_test(args[0].executor_args,
                                                   warmup, iter, time),
                          free_coll, st);
        }
        print_time(cnt, args[0], time);
        coll->free_args(args[0]);
        if (max_count == 0) {
            /* exit from loop when min_count == max_count == 0 */
            break;
        }
    }

    return UCC_OK;
free_coll:
    coll->free_args(args[0]);
exit_err:
    return st;
}

static inline double get_time_us(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

ucc_status_t ucc_pt_benchmark::run_single_coll_test(ucc_coll_args_t args,
                                                    int nwarmup, int niter,
                                                    double &time)
                                                    noexcept
{
    const bool    triggered  = config.triggered;
    const bool    persistent = config.persistent;
    ucc_team_h    team       = comm->get_team();
    ucc_context_h ctx        = comm->get_context();
    ucc_status_t  st         = UCC_OK;
    ucc_coll_req_h req;
    ucc_ee_h ee;
    ucc_ev_t comp_ev, *post_ev;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = 0;

    if (triggered) {
        try {
            ee = comm->get_ee();
        } catch(std::exception &e) {
            std::cerr << e.what() << std::endl;
            return UCC_ERR_NO_MESSAGE;
        }
        /* dummy event, for benchmark purposes no real event required */
        comp_ev.ev_type         = UCC_EVENT_COMPUTE_COMPLETE;
        comp_ev.ev_context      = nullptr;
        comp_ev.ev_context_size = 0;
    }

    if (persistent) {
        UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
    }

    args.root = config.root % comm->get_size();
    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();

        if (!persistent) {
            UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
        }

        if (triggered) {
            comp_ev.req = req;
            UCCCHECK_GOTO(ucc_collective_triggered_post(ee, &comp_ev),
                          free_req, st);
            UCCCHECK_GOTO(ucc_ee_get_event(ee, &post_ev), free_req, st);
            ucc_assert(post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
            UCCCHECK_GOTO(ucc_ee_ack_event(ee, post_ev), free_req, st);
        } else {
            UCCCHECK_GOTO(ucc_collective_post(req), free_req, st);
        }

        st = ucc_collective_test(req);
        while (st > 0) {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
            st = ucc_collective_test(req);
        }

        if (!persistent) {
            ucc_collective_finalize(req);
        }
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
        args.root = (args.root + config.root_shift) % comm->get_size();
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }

    if (persistent) {
        ucc_collective_finalize(req);
    }

    if (niter != 0) {
        time /= niter;
    }
    return UCC_OK;
free_req:
    ucc_collective_finalize(req);
exit_err:
    return st;
}

ucc_status_t ucc_pt_benchmark::run_paired_coll_test(ucc_coll_args_t args[2],
                                                    int nwarmup, int niter,
                                                    double &time) noexcept
{
    const bool     triggered  = config.triggered;
    const bool     persistent = config.persistent;
    ucc_team_h     team       = comm->get_team();
    ucc_context_h  ctx        = comm->get_context();
    ucc_status_t   st         = UCC_OK;
    ucc_coll_req_h req[2];
    ucc_ee_h       ee;
    ucc_ev_t       comp_ev, *post_ev;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = 0;

    if (triggered) {
        try {
            ee = comm->get_ee();
        } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
            return UCC_ERR_NO_MESSAGE;
        }
        /* dummy event, for benchmark purposes no real event required */
        comp_ev.ev_type         = UCC_EVENT_COMPUTE_COMPLETE;
        comp_ev.ev_context      = nullptr;
        comp_ev.ev_context_size = 0;
    }

    if (persistent) {
        UCCCHECK_GOTO(ucc_collective_init(&args[0], &req[0], team), exit_err, st);
        UCCCHECK_GOTO(ucc_collective_init(&args[1], &req[1], team), exit_err, st);
    }

    // args[0].root = config.root % comm->get_size();
    // args[1].root = config.root % comm->get_size();
    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();

        if (!persistent) {
            UCCCHECK_GOTO(ucc_collective_init(&args[0], &req[0], team), exit_err, st);
            UCCCHECK_GOTO(ucc_collective_init(&args[1], &req[1], team), exit_err, st);
        }

        if (triggered) {
            // TODO: add support for triggered paired collectives
            comp_ev.req = req[0];
            UCCCHECK_GOTO(ucc_collective_triggered_post(ee, &comp_ev), free_req,
                          st);
            UCCCHECK_GOTO(ucc_ee_get_event(ee, &post_ev), free_req, st);
            ucc_assert(post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
            UCCCHECK_GOTO(ucc_ee_ack_event(ee, post_ev), free_req, st);
        } else {

            if (comm->get_rank() % 2 == 0)
            {
                UCCCHECK_GOTO(ucc_collective_post(req[0]), free_req, st);
                UCCCHECK_GOTO(ucc_collective_post(req[1]), free_req, st);
            }
            else
            {
                UCCCHECK_GOTO(ucc_collective_post(req[1]), free_req, st);
                UCCCHECK_GOTO(ucc_collective_post(req[0]), free_req, st);
            }
        }

        ucc_status_t st_send = ucc_collective_test(req[0]);
        ucc_status_t st_recv = ucc_collective_test(req[1]);
        while (st_send > 0 || st_recv > 0) {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
            st_send = ucc_collective_test(req[0]);
            st_recv = ucc_collective_test(req[1]);
        }

        if (!persistent) {
            ucc_collective_finalize(req[0]);
            ucc_collective_finalize(req[1]);
        }
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
        // args.root = (args.root + config.root_shift) % comm->get_size();
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }

    if (persistent) {
        ucc_collective_finalize(req[0]);
        ucc_collective_finalize(req[1]);
    }

    if (niter != 0) {
        time /= niter;
    }
    return UCC_OK;
free_req:
    ucc_collective_finalize(req[0]);
    ucc_collective_finalize(req[1]);
exit_err:
    return st;
}

ucc_status_t
ucc_pt_benchmark::run_single_executor_test(ucc_ee_executor_task_args_t args,
                                           int nwarmup, int niter,
                                           double &time) noexcept
{
    const bool              triggered = config.triggered;
    ucc_ee_executor_t      *executor  = comm->get_executor();
    ucc_status_t            st        = UCC_OK;
    ucc_ee_h                ee;
    ucc_ee_executor_task_t *task;

    time = 0;
    if (triggered) {
        try {
            ee = comm->get_ee();
        } catch(std::exception &e) {
            std::cerr << e.what() << std::endl;
            return UCC_ERR_NO_MESSAGE;
        }
        UCCCHECK_GOTO(ucc_ee_executor_start(executor, ee->ee_context),
                      exit_err, st);
    } else {
        UCCCHECK_GOTO(ucc_ee_executor_start(executor, nullptr), exit_err, st);
    }

    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();

        UCCCHECK_GOTO(ucc_ee_executor_task_post(executor, &args, &task),
                      stop_exec, st);
        st = ucc_ee_executor_task_test(task);
        while (st > 0) {
            st = ucc_ee_executor_task_test(task);
        }
        ucc_ee_executor_task_finalize(task);
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
    }

    UCCCHECK_GOTO(ucc_ee_executor_stop(executor), exit_err, st);
    if (niter != 0) {
        time /= niter;
    }
    return UCC_OK;

stop_exec:
    ucc_ee_executor_stop(executor);
exit_err:
    return st;
}

void ucc_pt_benchmark::print_header()
{
    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::left << std::setw(24)
                  << "Collective: " << ucc_pt_op_type_str(config.op_type)
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Memory type: " << ucc_memory_type_names[config.mt]
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Datatype: " << ucc_datatype_str(config.dt)
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Reduction: "
                  << (coll->has_reduction() ?
                        ucc_reduction_op_str(config.op):
                        "N/A")
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Inplace: "
                  << (coll->has_inplace() ?
                        std::to_string(config.inplace):
                        "N/A")
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Warmup:" << std::endl
                  << std::left << std::setw(24)
                  << "  small" << config.n_warmup_small << std::endl
                  << std::left << std::setw(24)
                  << "  large" << config.n_warmup_large << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Iterations:" << std::endl
                  << std::left << std::setw(24)
                  << "  small" << config.n_iter_small << std::endl
                  << std::left << std::setw(24)
                  << "  large" << config.n_iter_large << std::endl;
        std::cout.copyfmt(iostate);
        std::cout << std::endl;
        std::cout << std::setw(12) << "Count"
                  << std::setw(12) << "Size"
                  << std::setw(24) << "Time, us";
        if (config.full_print) {
            std::cout << std::setw(42) << "Bandwidth, GB/s";
        }
        std::cout << std::endl;
        std::cout << std::setw(36) << "avg"
                  << std::setw(12) << "min"
                  << std::setw(12) << "max";
        if (config.full_print) {
            std::cout << std::setw(12) << "avg"
                      << std::setw(12) << "max"
                      << std::setw(12) << "min";
        }
        std::cout << std::endl;
    }
}

void ucc_pt_benchmark::print_time(size_t count, ucc_pt_test_args_t args,
                                  double time)
{
    double time_us = time;
    size_t size    = count * ucc_dt_size(config.dt);
    int    gsize   = comm->get_size();
    double time_avg, time_min, time_max;

    comm->allreduce(&time_us, &time_min, 1, UCC_OP_MIN);
    comm->allreduce(&time_us, &time_max, 1, UCC_OP_MAX);
    comm->allreduce(&time_us, &time_avg, 1, UCC_OP_SUM);
    time_avg /= gsize;

    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::setprecision(2) << std::fixed;
        std::cout << std::setw(12) << (coll->has_range() ?
                                        std::to_string(count):
                                        "N/A")
                  << std::setw(12) << (coll->has_range() ?
                                        std::to_string(size):
                                        "N/A")
                  << std::setw(12) << time_avg
                  << std::setw(12) << time_min
                  << std::setw(12) << time_max;

        if (config.full_print) {
            if (!coll->has_bw()) {
                std::cout << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A";
            } else {
                if (config.op_type == UCC_PT_OP_TYPE_GATHER ||
                    config.op_type == UCC_PT_OP_TYPE_SCATTER) {
                    std::cout << std::setw(12) << "N/A"
                              << std::setw(12) << "N/A"
                              << std::setw(12) << coll->get_bw(time_max, gsize,
                                                               args);
                } else {
                    std::cout << std::setw(12) << coll->get_bw(time_avg, gsize,
                                                               args)
                              << std::setw(12) << coll->get_bw(time_min, gsize,
                                                               args)
                              << std::setw(12) << coll->get_bw(time_max, gsize,
                                                               args);
                }
            }
        }
        std::cout << std::endl;
        std::cout.copyfmt(iostate);
    }
}

ucc_pt_benchmark::~ucc_pt_benchmark()
{
    delete coll;
}
