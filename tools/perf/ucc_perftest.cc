#include <ucc/api/ucc.h>
#include "ucc_pt_comm.h"
#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_cuda.h"
#include "ucc_pt_rocm.h"
#include "ucc_pt_benchmark.h"

#include <memory>
#include <iostream>

int main(int argc, char *argv[])
{
    ucc_pt_config pt_config;
    std::unique_ptr<ucc_pt_comm> comm;
    std::unique_ptr<ucc_pt_benchmark> bench;
    ucc_status_t st;

    pt_config.process_args(argc, argv);
    ucc_pt_cuda_init();
    ucc_pt_rocm_init();
    try {
        comm = std::make_unique<ucc_pt_comm>(pt_config.comm);
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }
    st = comm->init();
    if (st != UCC_OK) {
        std::exit(1);
    }
    try {
        bench = std::make_unique<ucc_pt_benchmark>(pt_config.bench, comm.get());
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        comm->finalize();
        std::exit(1);
    }
    bench->run_bench();
    comm->finalize();
    return 0;
}
