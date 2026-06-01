#ifndef HOST_STAGE_HPP
#define HOST_STAGE_HPP

#include "concurrency.hpp"
#include "types.hpp"

void cleanup_host_item(PwsHostItem *item);
void release_host_staging(PwsHostItem *item, HostGroupBudget *budget);

#endif
