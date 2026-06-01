#include "task_queue.hpp"

void TaskQueueInit(TaskQueue *queue, size_t total_groups)
{
    queue->next_group = 0;
    queue->total_groups = total_groups;
    pthread_mutex_init(&queue->mutex, NULL);
}

void TaskQueueDestroy(TaskQueue *queue)
{
    pthread_mutex_destroy(&queue->mutex);
}

int TaskQueuePop(TaskQueue *queue, size_t capacity, size_t *start_group, size_t *group_count)
{
    pthread_mutex_lock(&queue->mutex);

    if (queue->next_group >= queue->total_groups)
    {
        pthread_mutex_unlock(&queue->mutex);
        *start_group = 0;
        *group_count = 0;
        return 0;
    }

    size_t remaining = queue->total_groups - queue->next_group;
    size_t take = capacity < remaining ? capacity : remaining;

    *start_group = queue->next_group;
    *group_count = take;
    queue->next_group += take;

    pthread_mutex_unlock(&queue->mutex);
    return 1;
}
