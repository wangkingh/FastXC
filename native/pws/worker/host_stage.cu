#include "host_stage.hpp"

#include <cstdlib>

void cleanup_host_item(PwsHostItem *item)
{
    if (!item)
        return;
    std::free(item->prestack_data);
    std::free(item->linear_stack);
    std::free(item->group_weights);
    item->prestack_data = NULL;
    item->linear_stack = NULL;
    item->group_weights = NULL;
}

void release_host_staging(PwsHostItem *item, HostGroupBudget *budget)
{
    std::size_t groups = item->ngroups;
    cleanup_host_item(item);
    budget->release(groups);
}
