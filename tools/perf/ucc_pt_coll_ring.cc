/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_ring::ucc_pt_coll_ring(ucc_datatype_t dt, ucc_memory_type mt,
                                     int root_shift, bool is_persistent,
                                     ucc_pt_comm *communicator)
                   : ucc_pt_coll(communicator)
{
    has_inplace_   = false;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = root_shift;

    coll_args.mask              = UCC_COLL_ARGS_FIELD_ACTIVE_SET;
    coll_args.flags             = 0;
    coll_args.coll_type         = UCC_COLL_TYPE_BCAST;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;

    if (is_persistent) {
        coll_args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }
}

ucc_status_t ucc_pt_coll_ring::init_args(size_t count,
                                          ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args     = test_args.coll_args;
    size_t           dt_size  = ucc_dt_size(coll_args.src.info.datatype);
    size_t           size     = count * dt_size;
    ucc_status_t     st;

    args                = coll_args;
    args.src.info.count = count;

    args.root = test_args.coll_args.root;
    args.active_set.size = 2;
    args.active_set.start = comm->get_rank();
    args.active_set.stride = 1; // send to next rank

    UCCCHECK_GOTO(ucc_pt_alloc(&src_header, size, args.src.info.mem_type), exit,
                  st);
    args.src.info.buffer = src_header->addr;
exit:
    return st;
}

void ucc_pt_coll_ring::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_pt_free(src_header);
}

float ucc_pt_coll_ring::get_bw(float time_us, int grsize,
                                ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float            S    = args.src.info.count *
                            ucc_dt_size(args.src.info.datatype);

    return S / time_us / 1000.0f;
}

ucc_status_t ucc_pt_coll_ring::alloc_buffers(size_t count, ucc_coll_args_t (&test_args)[2])
{
    size_t           dt_size  = ucc_dt_size(coll_args.src.info.datatype);
    size_t           size     = count * dt_size;
    ucc_status_t     st;

    UCCCHECK_GOTO(ucc_pt_alloc(&send_header, size, test_args[0].src.info.mem_type), exit, st);
    test_args[0].src.info.buffer = src_header->addr;
    test_args[0].src.info.count = count;
    test_args[0].src.info.datatype = coll_args.src.info.datatype;
    test_args[0].src.info.mem_type = coll_args.src.info.mem_type;

    UCCCHECK_GOTO(ucc_pt_alloc(&recv_header, size, test_args[0].src.info.mem_type), exit, st);
    test_args[1].src.info.buffer = recv_header->addr;
    test_args[1].src.info.count = count;
    test_args[1].src.info.datatype = coll_args.src.info.datatype;
    test_args[1].src.info.mem_type = coll_args.src.info.mem_type;

    exit:
    return st;
}

void ucc_pt_coll_ring::free_buffers()
{
    ucc_pt_free(send_header);
    ucc_pt_free(recv_header);
}
