#include <cstdlib>
#include <vector>

#include "logger.h"
#include "pipeline.hpp"
#include "tfpws_schedule.hpp"

extern "C"
{
#include "arguproc.h"
}

struct TfpwsRunState
{
    ARGUTYPE args;
    std::vector<GpuWorkerConfig> workers;
};

static void log_tfpws_run_start(const ARGUTYPE *args)
{
    LOG_INFO("tfpws_run_start",
             "input_sourcepack_list=\"%s\" output_sourcepack=\"%s\" gpu_list=\"%s\" memory_budget_mib=\"%s\" sub_stack_size=%d band_limited=%d band_fmin_hz=%.8g band_fmax_hz=%.8g band_taper_hz=%.8g progress_file=\"%s\"",
             args->sourcepack_list ? args->sourcepack_list : "",
             args->output_sourcepack ? args->output_sourcepack : "",
             args->gpu_list ? args->gpu_list : "",
             args->gpu_ram_limit_mib_list ? args->gpu_ram_limit_mib_list : "",
             args->sub_stack_size,
             args->band_limited,
             args->band_fmin,
             args->band_fmax,
             args->band_taper_hz,
             args->progress_file ? args->progress_file : "");
}

static int prepare_tfpws_run(int argc, char **argv, TfpwsRunState *run)
{
    ArgumentProcess(argc, argv, &run->args);
    log_tfpws_run_start(&run->args);

    run->workers = make_tfpws_worker_configs(&run->args);
    LOG_INFO("tfpws_workers_ready",
             "worker_count=%zu",
             run->workers.size());
    return run->workers.empty() ? 1 : 0;
}

static int run_tfpws(const TfpwsRunState *run)
{
    /*
     * Formal TFPWS data flow:
     *   sourcepack_inputs.txt -> sourcepack_index.tsv -> pack records
     *   -> group by path_id + component_slot
     *   -> TF-PWS compute
     *   -> output SourcePack with sourcepack_index.tsv
     */
    return run_tfpws_pipeline(&run->args, run->workers);
}

int main(int argc, char **argv)
{
    TfpwsRunState run;

    if (prepare_tfpws_run(argc, argv, &run) != 0)
        return EXIT_FAILURE;

    const int status = run_tfpws(&run);
    LOG_INFO("tfpws_run_finish",
             "status=%s",
             status == 0 ? "success" : "failed");
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
