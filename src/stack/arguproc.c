#include "arguproc.h"
#include "read_sac_lst.h" // for my_strdup
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    int opt;

    pargument->sac_lst = NULL;
    pargument->out_sac = NULL;
    pargument->normalize = 1;
    if (argc <= 1)
    {
        usage();
        exit(-1);
    }
    struct option long_options[] = {
        {"I", required_argument, NULL, 'I'},
        {"O", required_argument, NULL, 'O'},
        {"A", no_argument, &(pargument->normalize), 0},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "I:O:A", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'I':
            pargument->sac_lst = my_strdup(optarg);
            break;
        case 'O':
            pargument->out_sac = optarg;
            break;
        case 'A':
            pargument->normalize = 0;
            break;
        default:   // 处理其他任何错误的默认情况
            usage();
            exit(EXIT_FAILURE);
        }
    }
}

void usage()
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "ncfstack -I ncf_list -O outpath [-A] \n");
    fprintf(stderr, "  -I ncf_dir.\n");
    fprintf(stderr, "  if -A is set, do NOT normalize output NCF,in accumulate mode.\n");
    fprintf(stderr, " Else normalize the output NCF by devide N effect day as default.\n");
    fprintf(stderr, "Note:\n");
    fprintf(stderr, "   1. The input file should have same delta and npts.\n");
    fprintf(stderr, "      Since we create the NCFs list by ourselves, we do not  make too much check\n");
    fprintf(stderr, "   2. Header of the stacked ncfile will be taken from the first file in list\n");
    fprintf(stderr, "   3. We forcely set the KZTIME and KZDATE to one special day to be convenience.\n");
    fprintf(stderr, "   4. By default,we will always normlized the stacked NCF by divide N.\n");
    fprintf(stderr, "      The effective stacked N is save into hd.unused27.\n");
    fprintf(stderr, "   5. We have not limitation on number of files in the listfile.\n");
    fprintf(stderr, "Version:  updated wangwt@20190510\n");
    fprintf(stderr, "Version:  last updated wangjx@20231105\n");
    fprintf(stderr, "OpenMP version abandoned by wangjx@20230717, since we will do multiprocessing in higher level python script\n");
}
