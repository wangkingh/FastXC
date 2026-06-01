#ifndef _SPACK_WRITER_H_
#define _SPACK_WRITER_H_

#include "in_out_node.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct SpackWriter SpackWriter;

SpackWriter *CreateSpackWriter(const char *spack_root, int worker_id);
int SpackWriterAppend(SpackWriter *writer, const InOutNode *node);
int SpackWriterCloseBatch(SpackWriter *writer);
void DestroySpackWriter(SpackWriter *writer);

#ifdef __cplusplus
}
#endif

#endif
