#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  parg->src_files_list = NULL;
  parg->sta_files_list = NULL;
  parg->ncf_dir = NULL;
  parg->cc_len = 0;
  parg->gpu_id = 0;
  parg->save_linear = 1;
  parg->save_pws = 0;
  parg->save_tfpws = 0;
  parg->save_segment = 0;
  parg->cpu_count = 1;
  parg->gpu_num = 1;
  parg->threshold_distance = 40000;

  /* check argument */
  if (argc <= 1)
  {
    // usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "A:B:O:C:G:S:T:D:")) != -1)
  {
    switch (c)
    {
    case 'A':
      parg->src_files_list = optarg;
      break;
    case 'B':
      parg->sta_files_list = optarg;
      break;
    case 'O':
      parg->ncf_dir = optarg;
      break;
    case 'C':
      parg->cc_len = atof(optarg);
      break;
    case 'G':
      parg->gpu_id = atoi(optarg);
      break;
    case 'T':
      parg->cpu_count = atoi(optarg);
      break;
    case 'S':
      if (strlen(optarg) != 4 || strspn(optarg, "01") != 4)
      {
        fprintf(stderr, "Error: Option -S requires a four-digit binary number consisting of 0s and 1s.\n");
        exit(-1);
      }
      parg->save_linear = (optarg[0] == '1') ? 1 : 0;  // 解析第一位
      parg->save_pws = (optarg[1] == '1') ? 1 : 0;     // 解析第二位
      parg->save_tfpws = (optarg[2] == '1') ? 1 : 0;   // 解析第三位
      parg->save_segment = (optarg[3] == '1') ? 1 : 0; // 解析第四位
      break;
    case 'D':
      parg->threshold_distance = atof(optarg);
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
      "    -S Save options: 4 digits binary number, 1 for save, 0 for not save for [linear,pws,tfpws,segments]\n"
      "Version:\n"
      "  last update by wangjx@20241127\n"
      "  cuda version\n");
}