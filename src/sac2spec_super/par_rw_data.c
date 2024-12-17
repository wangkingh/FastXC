#include "par_rw_data.h"

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

void *read_in_sacdata(void *arg)
{
  thread_info_read *tinfo = (thread_info_read *)arg;
  int i;
  for (i = tinfo->start; i < tinfo->end; i++)
  {
    // reading in sachead
    if (read_sachead(tinfo->InOutList[i].sacpath, tinfo->InOutList[i].sac_hd) != 0)
    {
      fprintf(stderr, "Error reading head of SAC file: %s\n",
              tinfo->InOutList[i].sacpath);
      return (void *)-1;
    }

    // reading in data
    if (read_sac_buffer(tinfo->InOutList[i].sacpath, tinfo->InOutList[i].sac_hd,
                        tinfo->InOutList[i].sac_data) == NULL)
    {
      fprintf(stderr, "Error reading SAC file: %s\n",
              tinfo->InOutList[i].sacpath);
      return (void *)-1;
    }
  }

  return NULL;
}

int parallel_read_sac(ThreadPoolRead *pool, size_t proccnt, InOutNode *pInOutList,
                      int num_threads)
{
  printf("[Waiting:] Reading in SAC data, take some time...\n");

  // divide the work
  size_t chunk = proccnt / num_threads;
  size_t remainder = proccnt % num_threads; // calculate the remainder

  size_t start = 0;
  int i;
  for (i = 0; i < num_threads; i++)
  {
    pool->tinfo[i].start = start;
    // if i<remainder, then add 1 to the chunk size
    pool->tinfo[i].end = start + chunk + (i < remainder ? 1 : 0);
    start = pool->tinfo[i].end; // the start of the next thread is the end of this
                                // current thread
    pool->tinfo[i].InOutList = pInOutList;
    int ret = pthread_create(&pool->threads[i], NULL, read_in_sacdata, &pool->tinfo[i]);

    if (ret)
    {
      fprintf(stderr, "Error creating thread\n");
      return -1;
    }
  }
  for (i = 0; i < pool->num_threads; i++)
  {
    void *status;
    if (pthread_join(pool->threads[i], &status))
    {
      fprintf(stderr, "Error joining thread\n");
      return -1;
    }
    if ((int)(size_t)status == -1)
    {
      fprintf(stderr, "Error occurred in thread while reading SAC file.\n");
      return -1;
    }
  }

  printf("Done reading in SAC data.\n");
  return 0;
}

ThreadPoolWrite *create_threadpool_write(size_t num_threads)
{
  ThreadPoolWrite *pool = malloc(sizeof(ThreadPoolWrite));
  pool->threads = malloc(num_threads * sizeof(pthread_t));
  pool->tinfo = malloc(num_threads * sizeof(thread_info_write));
  pool->num_threads = num_threads;
  return pool;
}

void destroy_threadpool_write(ThreadPoolWrite *pool)
{
  free(pool->threads);
  free(pool->tinfo);
  free(pool);
}

void *write_out_spec(void *arg)
{
  thread_info_write *tinfo = (thread_info_write *)arg;
  int i;
  for (i = tinfo->start; i < tinfo->end; i++)
  {
    FILE *fid = fopen(tinfo->InOutList[i].specpath, "wb");
    if (fid == NULL)
    {
      perror("Error opening file!\n");
      continue;
    }

    SEGSPEC *segspec_hd = tinfo->InOutList[i].segspec_hd;
    complex *specdata = tinfo->InOutList[i].spectrum;

    segspec_hd->stla = (tinfo->InOutList[i].sac_hd)->stla;
    segspec_hd->stlo = (tinfo->InOutList[i].sac_hd)->stlo;
    segspec_hd->nstep = tinfo->InOutList[i].nstep;
    segspec_hd->nspec = tinfo->InOutList[i].nspec;
    segspec_hd->df = tinfo->InOutList[i].df;
    segspec_hd->dt = tinfo->InOutList[i].dt;

    int nspec = segspec_hd->nspec;
    int nstep = segspec_hd->nstep;
    int write_size = sizeof(complex) * nspec * nstep;

    fwrite(segspec_hd, sizeof(SEGSPEC), 1, fid);
    fwrite(specdata, write_size, 1, fid);
    fclose(fid);
  }

  return NULL;
}

int parallel_write_spec(ThreadPoolWrite *pool, size_t h_proccnt,
                        InOutNode *pInOutList, int num_threads)
{
  printf("Writing the output spectra ... take some time\n");

  // divide the work
  size_t chunk = h_proccnt / num_threads;
  size_t remainder = h_proccnt % num_threads; // calculate the remainder

  size_t start = 0;
  int i;
  for (i = 0; i < num_threads; i++)
  {
    pool->tinfo[i].start = start;
    // if i<remainder, then add 1 to the chunk size
    pool->tinfo[i].end = start + chunk + (i < remainder ? 1 : 0);
    start = pool->tinfo[i].end; // the start of the next thread is the end of this current thread

    pool->tinfo[i].InOutList = pInOutList;

    int ret = pthread_create(&pool->threads[i], NULL, write_out_spec, &pool->tinfo[i]);
    if (ret)
    {
      fprintf(stderr, "Error creating thread\n");
      return -1;
    }
  }

  for (i = 0; i < pool->num_threads; i++)
  {
    void *status;
    if (pthread_join(pool->threads[i], &status))
    {
      fprintf(stderr, "Error joining thread\n");
      return -1;
    }
  }
  printf("Finish writing the output %lu spectra\n", h_proccnt);
  return 0;
}