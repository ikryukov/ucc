/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

/* NVLS requires 16-byte alignment (4 elements for float32, 8 for bf16) */
#define NVLS_COUNT_ALIGN 4

static inline size_t align_count(size_t c)
{
    return (c + NVLS_COUNT_ALIGN - 1) / NVLS_COUNT_ALIGN * NVLS_COUNT_ALIGN;
}

TestReduceScatterv::TestReduceScatterv(
    ucc_test_team_t &_team, TestCaseParams &params)
    : TestCase(_team, UCC_COLL_TYPE_REDUCE_SCATTERV, params)
{
    size_t dt_size = ucc_dt_size(params.dt);
    size_t count   = msgsize / dt_size;
    int    rank, comm_size;
    counts = NULL;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);
    op = params.op;
    dt = params.dt;

    if (skip_reduce(test_max_size < msgsize, TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }
    counts = (int *)ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK(counts);

    /* Use aligned counts for NVLS compatibility */
    size_t base_count   = align_count(count / comm_size);
    size_t total        = 0;
    size_t aligned_left = count;

    for (int i = 0; i < comm_size; i++) {
        if (i == comm_size - 1) {
            /* Last rank gets the remainder, aligned */
            counts[i] = align_count(aligned_left);
        } else {
            counts[i] = base_count;
            aligned_left -= base_count;
        }
        total += counts[i];
    }

    /* Use aligned total for buffer sizes */
    size_t total_size = total * dt_size;

    check_buf         = ucc_malloc(total_size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (inplace) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, total_size, mem_type));
        rbuf = rbuf_mc_header->addr;
    } else {
        UCC_CHECK(
            ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size, mem_type));
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, total_size, mem_type));
        rbuf                   = rbuf_mc_header->addr;
        sbuf                   = sbuf_mc_header->addr;
        args.src.info.buffer   = sbuf;
        args.src.info.count    = total;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
    }

    args.op                  = op;
    args.dst.info_v.counts   = (ucc_count_t *)counts;
    args.dst.info_v.buffer   = rbuf;
    args.dst.info_v.datatype = dt;
    args.dst.info_v.mem_type = mem_type;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduceScatterv::set_input(int iter_persistent)
{
    size_t dt_size = ucc_dt_size(dt);
    int    rank, comm_size;
    size_t total = 0;
    void  *buf;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);

    /* Compute total from aligned counts */
    for (int i = 0; i < comm_size; i++) {
        total += counts[i];
    }

    if (inplace) {
        buf = rbuf;
    } else {
        buf = sbuf;
    }
    init_buffer(buf, total, dt, mem_type, rank * (iter_persistent + 1));
    UCC_CHECK(ucc_mc_memcpy(
        check_buf, buf, total * dt_size, UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestReduceScatterv::check()
{
    ucc_status_t status;
    int          comm_rank, comm_size, completed;
    MPI_Request  req;
    size_t       offset;

    MPI_Comm_rank(team.comm, &comm_rank);
    MPI_Comm_size(team.comm, &comm_size);
    MPI_Ireduce_scatter(
        MPI_IN_PLACE,
        check_buf,
        counts,
        ucc_dt_to_mpi(dt),
        op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op),
        team.comm,
        &req);

    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while (!completed);

    if (op == UCC_OP_AVG) {
        status = divide_buffer(
            check_buf, team.team->size, counts[comm_rank], dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    if (inplace) {
        offset = 0;
        for (int i = 0; i < comm_rank; i++) {
            offset += counts[i];
        }
        return compare_buffers(
            PTR_OFFSET(rbuf, offset * ucc_dt_size(dt)),
            check_buf,
            counts[comm_rank],
            dt,
            mem_type);
    }
    return compare_buffers(rbuf, check_buf, counts[comm_rank], dt, mem_type);
}

TestReduceScatterv::~TestReduceScatterv()
{
    if (counts) {
        ucc_free(counts);
    }
}

std::string TestReduceScatterv::str()
{
    return std::string("tc=") + ucc_coll_type_str(args.coll_type) +
           " team=" + team_str(team.type) +
           " msgsize=" + std::to_string(msgsize) +
           " inplace=" + (inplace ? "1" : "0") +
           " persistent=" + (persistent ? "1" : "0") +
           " dt=" + ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
