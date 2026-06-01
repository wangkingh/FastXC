#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_config.hpp"
#include "logger.h"
#include "sourcepack_pipeline.hpp"

extern "C"
{
#include "arguproc.h"
}

struct PwsRunState
{
    PwsSourcePackArgs args;
    PwsGpuWorkerConfig gpu_config;
};

static const char *safe_text(const char *value)
{
    return value ? value : "";
}

static void log_pws_run_start(const PwsSourcePackArgs &args)
{
    LOG_INFO("pws_run_start",
             "index_list=\"%s\" output_dir=\"%s\" gpu_workers=\"%s\" gpu_memory_mib=\"%s\" substack_size=%d staged_group_limit=%zu progress=\"%s\"",
             safe_text(args.index_list_path),
             safe_text(args.output_dir),
             safe_text(args.gpu_worker_list),
             safe_text(args.gpu_memory_mib_list),
             args.substack_size,
             args.staged_group_limit,
             safe_text(args.progress_path));
}

static std::string join_size_values(const std::vector<std::size_t> &values)
{
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (!out.empty())
            out += ",";
        out += std::to_string(values[i]);
    }
    return out;
}

static std::string join_int_values(const std::vector<int> &values)
{
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (!out.empty())
            out += ",";
        out += std::to_string(values[i]);
    }
    return out;
}

static int configure_gpu_workers(const PwsSourcePackArgs &args,
                                 PwsGpuWorkerConfig *gpu_config)
{
    LOG_INFO("pws_gpu_config_start",
             "gpu_workers=\"%s\" gpu_memory_mib=\"%s\"",
             safe_text(args.gpu_worker_list),
             safe_text(args.gpu_memory_mib_list));

    return parse_pws_gpu_worker_config(args.gpu_worker_list,
                                       args.gpu_memory_mib_list,
                                       gpu_config);
}

static int prepare_pws_run(int argc, char **argv, PwsRunState *run)
{
    ParsePwsSourcePackArgs(argc, argv, &run->args);
    log_pws_run_start(run->args);

    if (configure_gpu_workers(run->args, &run->gpu_config) != 0)
        return 1;

    LOG_INFO("pws_gpu_config_ready",
             "worker_count=%zu gpu_ids=\"%s\" memory_limits_mib=\"%s\" physical_worker_counts=\"%s\"",
             run->gpu_config.gpu_ids.size(),
             join_int_values(run->gpu_config.gpu_ids).c_str(),
             join_size_values(run->gpu_config.gpu_memory_limits_mib).c_str(),
             join_size_values(run->gpu_config.physical_worker_counts).c_str());
    return run->gpu_config.gpu_ids.empty() ? 1 : 0;
}

static int run_sourcepack_pws(const PwsRunState *run)
{
    /*
     * Formal PWS data flow:
     *   index-list -> sourcepack_index.tsv streams -> pack records
     *   -> group by path_id + component_slot
     *   -> substack/linear-stack staging
     *   -> GPU PWS compute
     *   -> output SourcePack shards and merged sourcepack_index.tsv
     */
    return run_pws_sourcepack_pipeline(run->args, run->gpu_config);
}

int main(int argc, char **argv)
{
    PwsRunState run;

    if (prepare_pws_run(argc, argv, &run) != 0)
        return EXIT_FAILURE;

    const int status = run_sourcepack_pws(&run);
    LOG_INFO("pws_run_finish",
             "status=%s",
             status == 0 ? "success" : "failed");
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
