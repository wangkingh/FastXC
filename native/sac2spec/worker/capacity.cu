#include "worker_runtime.hpp"
#include "logger.h"

static void SortWorkerPlansByCapacity(WorkerCapacityPlan *plans, int count)
{
    for (int i = 0; i < count; i++)
    {
        int best = i;
        for (int j = i + 1; j < count; j++)
        {
            if (plans[j].gpu_cap > plans[best].gpu_cap)
            {
                best = j;
            }
        }
        if (best != i)
        {
            WorkerCapacityPlan tmp = plans[i];
            plans[i] = plans[best];
            plans[best] = tmp;
        }
    }
}

int PlanWorkerCapacities(WorkerCapacityPlan *plans, int gpu_worker_count,
                         size_t total_groups, size_t host_cap,
                         int total_io_threads)
{
    int usable = 0;
    for (int i = 0; i < gpu_worker_count; i++)
    {
        plans[i].capacity = 0;
        plans[i].io_threads = 0;
        plans[i].enabled = 0;
        if (plans[i].gpu_cap > 0)
        {
            usable++;
        }
    }

    SortWorkerPlansByCapacity(plans, gpu_worker_count);

    if (total_groups == 0 || host_cap == 0 || usable == 0)
    {
        return 0;
    }

    size_t usable_count = (size_t)usable;
    size_t active_limit = usable_count < total_groups ? usable_count : total_groups;
    active_limit = active_limit < host_cap ? active_limit : host_cap;
    int active_count = (int)active_limit;
    if (active_count < 1)
    {
        return 0;
    }

    size_t active_cap_sum = 0;
    for (int i = 0; i < active_count; i++)
    {
        plans[i].enabled = 1;
        plans[i].capacity = 1;
        active_cap_sum += plans[i].gpu_cap;
    }

    size_t budget = host_cap < total_groups ? host_cap : total_groups;
    budget = budget < active_cap_sum ? budget : active_cap_sum;
    size_t assigned = (size_t)active_count;
    if (budget > assigned)
    {
        size_t remaining = budget - assigned;
        for (int i = 0; i < active_count; i++)
        {
            size_t room = plans[i].gpu_cap - plans[i].capacity;
            size_t extra = (remaining * plans[i].gpu_cap) / active_cap_sum;
            if (extra > room)
            {
                extra = room;
            }
            plans[i].capacity += extra;
            assigned += extra;
        }

        while (assigned < budget)
        {
            int best = -1;
            for (int i = 0; i < active_count; i++)
            {
                if (plans[i].capacity < plans[i].gpu_cap &&
                    (best < 0 || plans[i].gpu_cap > plans[best].gpu_cap))
                {
                    best = i;
                }
            }
            if (best < 0)
            {
                break;
            }
            plans[best].capacity++;
            assigned++;
        }
    }

    if (total_io_threads < active_count)
    {
        LOG_WARN("io_threads_oversubscribed",
                 "requested=%d active_workers=%d effective_min_threads=%d",
                 total_io_threads, active_count, active_count);
        total_io_threads = active_count;
    }

    for (int i = 0; i < active_count; i++)
    {
        plans[i].io_threads = 1;
    }
    int remaining_threads = total_io_threads - active_count;
    if (remaining_threads > 0)
    {
        size_t capacity_sum = 0;
        for (int i = 0; i < active_count; i++)
        {
            capacity_sum += plans[i].capacity;
        }

        int assigned_threads = active_count;
        for (int i = 0; i < active_count; i++)
        {
            int extra = (int)(((size_t)remaining_threads * plans[i].capacity) / capacity_sum);
            plans[i].io_threads += extra;
            assigned_threads += extra;
        }

        while (assigned_threads < total_io_threads)
        {
            int best = 0;
            for (int i = 1; i < active_count; i++)
            {
                if (plans[i].capacity > plans[best].capacity)
                {
                    best = i;
                }
            }
            plans[best].io_threads++;
            assigned_threads++;
        }
    }

    for (int i = 0; i < gpu_worker_count; i++)
    {
        LOG_INFO("gpu_worker_capacity_plan",
                 "gpu=%d gpu_frame_cap=%zu gpu_file_cap=%zu enabled=%d file_capacity=%zu io_threads=%d",
                 plans[i].gpu_id, plans[i].gpu_frame_cap, plans[i].gpu_cap,
                 plans[i].enabled, plans[i].capacity, plans[i].io_threads);
    }

    return active_count;
}
