#ifndef SAC2SPEC_TASK_QUEUE_HPP
#define SAC2SPEC_TASK_QUEUE_HPP

#include <cstddef>
#include <pthread.h>

typedef struct TaskQueue
{
    size_t next_group;
    size_t next_batch_seq;
    size_t total_groups;
    pthread_mutex_t mutex;
} TaskQueue;

void TaskQueueInit(TaskQueue *queue, size_t total_groups);
void TaskQueueDestroy(TaskQueue *queue);
int TaskQueuePop(TaskQueue *queue, size_t capacity,
                 size_t *start_group, size_t *group_count,
                 size_t *batch_seq);

#endif
