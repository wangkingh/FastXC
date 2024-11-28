#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  parg->src_lst_path = NULL;
  parg->sta_lst_path = NULL;
  parg->ncf_dir = NULL;
  parg->cclength = 0;
  parg->gpu_id = 0;
  parg->cpu_count = 1;
  parg->gpu_task_num = 1;
  parg->max_distance = 400000;

  /* check argument */
  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "A:B:O:C:G:U:T:D:")) != -1)
  {
    switch (c)
    {
    case 'A':
      parg->src_lst_path = optarg;
      break;
    case 'B':
      parg->sta_lst_path = optarg;
      break;
    case 'O':
      parg->ncf_dir = optarg;
      break;
    case 'C':
      parg->cclength = atof(optarg);
      break;
    case 'G':
      parg->gpu_id = atof(optarg);
      break;
    case 'T':
      parg->cpu_count = atof(optarg);
      break;
    case 'U':
      parg->gpu_task_num = atof(optarg);
      break;
    case 'D':
      parg->max_distance = atof(optarg);
      break;
    case '?':
    default:
      fprintf(stderr, "Unknown option %c\n", optopt);
      exit(-1);
    }
  }
  /* end of parsing command line arguments */
}

void usage()
{
  fprintf(
      stderr,
      "\nUsage:\n"
      "specxc_mg -A virt_src_lst -B virt_sta_dir -C halfCCLength -O "
      "outputdir -G gpu num\n"
      "Options:\n"
      "    -A Specify the list file of input files for the 1st station, eg virtual "
      "source\n"
      "    -B Specify the list file of input files for the 2nd station, eg virtual "
      "station\n"
      "    -O Specify the output directory for NCF files as sac format\n"
      "    -C Half of cclenth (in seconds).\n"
      "    -D max ncf distances (in km).\n"
      "    -G ID of Gpu device to be launched \n"
      "    -U number of tasks deploy on a single GPU\n"
      "    -T number of CPUs will be used in this threads\n"
      "Version:\n"
      "  last update by wangjx@20240515\n"
      "  cuda version\n");
}