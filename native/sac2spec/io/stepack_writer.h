#ifndef _STEPACK_WRITER_H_
#define _STEPACK_WRITER_H_

#include "in_out_node.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct StepackWriter StepackWriter;

StepackWriter *CreateStepackWriter(const char *stepack_root, int worker_id);
int StepackWriterAppendBatch(StepackWriter *writer, const InOutNode *nodes,
                             size_t node_count, size_t batch_seq,
                             size_t start_group, size_t group_count);
void DestroyStepackWriter(StepackWriter *writer);

#ifdef __cplusplus
}
#endif

#endif
