#include "read_segspec.h"

void compare_strings(const char *str1, const char *str2)
{
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    printf("Strings differ in length: %d vs %d\n", len1, len2);
    if (len1 != len2)
    {
        printf("Strings differ in length: %d vs %d\n", len1, len2);
    }
    else
    {
        for (int i = 0; i < len1; i++)
        {
            if (str1[i] != str2[i])
            {
                printf("Difference at position %d: '%c' (0x%X) vs '%c' (0x%X)\n", i, str1[i], (unsigned char)str1[i], str2[i], (unsigned char)str2[i]);
            }
        }
        printf("No differences found.\n");
    }
}

ThreadPoolRead *create_threadpool_read(size_t num_threads)
{
    ThreadPoolRead *pool = malloc(sizeof(ThreadPoolRead));
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = malloc(num_threads * sizeof(thread_info_read));
    pool->num_threads = num_threads;
    return pool;
}

void destroy_threadpool_read(ThreadPoolRead *pool)
{
    free(pool->threads);
    free(pool->tinfo);
    free(pool);
}

int read_spec_buffer(const char *file_path, complex *buffer, size_t vec_size)
{
    FILE *strm;
    SEGSPEC *segspec_hd = malloc(sizeof(SEGSPEC));

    if ((strm = fopen(file_path, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n", file_path);
        return -1;
    }

    if (fseek(strm, sizeof(SEGSPEC), SEEK_SET) != 0)
    {
        fprintf(stderr, "Error in skipping SEGSPEC header %s\n", file_path);
        fclose(strm);
        return -1;
    }

    if (fread(buffer, vec_size, 1, strm) != 1)
    {
        fprintf(stderr, "Error in reading SEGSPEC data %s\n", file_path);
        return -1;
    }

    fclose(strm);
    free(segspec_hd);
    return 0;
}

void *read_in_segspec(void *arg)
{
    thread_info_read *tinfo = (thread_info_read *)arg;

    for (size_t i = tinfo->start; i < tinfo->end; i++)
    {
        size_t offset = i * (tinfo->vec_size) / sizeof(complex);
        //  读取source_path到src_buffer
        if (read_spec_buffer(tinfo->filepairs[i].source_path, tinfo->src_buffer + offset, tinfo->vec_size) == -1)
        {
            fprintf(stderr, "Error reading SEGSPEC file: %s\n", tinfo->filepairs[i].source_path);
            return (void *)-1;
        }

        // 读取station_path到sta_buffer
        if (read_spec_buffer(tinfo->filepairs[i].station_path, tinfo->sta_buffer + offset, tinfo->vec_size) == -1)
        {
            fprintf(stderr, "Error reading SEGSPEC file: %s\n", tinfo->filepairs[i].station_path);
            return (void *)-1;
        }
    }
    return NULL;
}

int parallel_read_segspec(ThreadPoolRead *pool, size_t proccnt, FilePair *pairs,
                          complex *src_buffer, complex *sta_buffer, size_t vec_size,
                          int num_threads)
{
    printf("[Waiting:] Reading in SEGSPEC data, take some time...\n");

    // divide the work
    size_t chunk = proccnt / num_threads;
    size_t remainder = proccnt % num_threads; // calculate the remainder

    size_t start = 0;
    for (int i = 0; i < num_threads; i++)
    {
        pool->tinfo[i].start = start;
        // if i<remainder, then add 1 to the chunk size
        // the start of the next thread is the end of this
        // current thread
        pool->tinfo[i].end = start + chunk + (i < remainder ? 1 : 0);
        start = pool->tinfo[i].end;
        pool->tinfo[i].filepairs = pairs;
        pool->tinfo[i].src_buffer = src_buffer;
        pool->tinfo[i].sta_buffer = sta_buffer;
        pool->tinfo[i].vec_size = vec_size;
        int ret = pthread_create(&pool->threads[i], NULL, read_in_segspec, &pool->tinfo[i]);

        if (ret)
        {
            fprintf(stderr, "Error creating thread\n");
            return -1;
        }
    }

    for (int j = 0; j < pool->num_threads; j++)
    {
        void *status;
        if (pthread_join(pool->threads[j], &status))
        {
            fprintf(stderr, "Error joining thread\n");
            return -1;
        }
        // printf("Thread %d joined\n", j);
        if ((int)(size_t)status == -1)
        {
            fprintf(stderr, "Error occurred in thread while reading SEGSPEC file.\n");
            return -1;
        }
    }

    printf("Done reading in SEGSPEC data.\n");
    return 0;
}

int read_spechead(char *file_path, SEGSPEC *hd)
{
    FILE *strm;

    if ((strm = fopen(file_path, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n", file_path);
        return -1;
    }

    if (fread(hd, sizeof(SEGSPEC), 1, strm) != 1)
    {
        fprintf(stderr, "Error in reading SAC header %s\n", file_path);
        fclose(strm);
        return -1;
    }

    fclose(strm);
    return 0;
}