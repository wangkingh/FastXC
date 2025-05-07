#include "arguproc.h"

// Designed by chatgpt in case that some compiler cannot handle strdup
char *my_strdup(const char *s)
{
    if (s == NULL)
    {
        return NULL;
    }
    char *new_str = (char *)malloc(strlen(s) + 1);
    if (new_str == NULL)
    {
        return NULL;
    }
    char *p = new_str;
    while (*s)
    {
        *p++ = *s++;
    }
    *p = '\0';
    return new_str;
}

/* check argument */
void Argumentprocess(int argc, char **argv, ARGUTYPE *parg)
{
    int opt;
    char *inputFile = NULL;
    char *outputFile = NULL;

    if (argc <= 1)
    {
        usage();
        exit(-1);
    }

    while ((opt = getopt(argc, argv, "I:O:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            inputFile = optarg;
            break;

        case 'O':
            outputFile = optarg;
            break;

        default:
            usage();
            exit(-1);
        }
    }

    // Check if inputFile and outputFile are set
    if (inputFile == NULL || outputFile == NULL)
    {
        fprintf(stderr, "Input and output files must be specified.\n");
        usage();
        exit(EXIT_FAILURE);
    }

    // process input file
    FILE *file_in = fopen(inputFile, "r");
    if (file_in == NULL)
    {
        perror("Failed to open the file");
        exit(EXIT_FAILURE);
    }

    char line_in[256]; // store each line of the file

    int flag = 0;
    int i;
    for (i = 1; i <= 9 && fgets(line_in, sizeof(line_in), file_in) != NULL; i++)
    {
        line_in[strcspn(line_in, "\n")] = '\0';
        char *file_content_in = my_strdup(line_in);
        if (file_content_in == NULL)
        {
            perror("Failed to allocate memory for input content");
            fclose(file_in);
            exit(EXIT_FAILURE);
        }

        // set the corresponding input file name
        switch (i)
        {
        case 1:
            parg->ee_in = file_content_in;
            flag++;
            break;
        case 2:
            parg->en_in = file_content_in;
            flag++;
            break;
        case 3:
            parg->ez_in = file_content_in;
            flag++;
            break;
        case 4:
            parg->ne_in = file_content_in;
            flag++;
            break;
        case 5:
            parg->nn_in = file_content_in;
            flag++;
            break;
        case 6:
            parg->nz_in = file_content_in;
            flag++;
            break;
        case 7:
            parg->ze_in = file_content_in;
            flag++;
            break;
        case 8:
            parg->zn_in = file_content_in;
            flag++;
            break;
        case 9:
            parg->zz_in = file_content_in;
            flag++;
            break;
        }
    }

    fclose(file_in);

    // Check if exactly 9 lines were processed in input file
    if (flag < 9)
    {
        fprintf(stderr, "Insufficient lines in input file.\n");
        exit(EXIT_FAILURE);
    }

    // process output file
    FILE *file_out = fopen(outputFile, "r");
    if (file_out == NULL)
    {
        perror("Failed to open the file");
        exit(EXIT_FAILURE);
    }

    flag = 0;
    char line_out[256]; // store each line of the file
    for (i = 1; i <= 9 && fgets(line_out, sizeof(line_out), file_out) != NULL; i++)
    {
        line_out[strcspn(line_out, "\n")] = '\0';

        char *file_content_out = my_strdup(line_out);
        if (file_content_out == NULL)
        {
            perror("Failed to allocate memory for output content");
            fclose(file_out);
            exit(EXIT_FAILURE);
        }

        switch (i)
        {
        case 1:
            parg->rr_out = file_content_out;
            flag++;
            break;
        case 2:
            parg->rt_out = file_content_out;
            flag++;
            break;
        case 3:
            parg->rz_out = file_content_out;
            flag++;
            break;
        case 4:
            parg->tr_out = file_content_out;
            flag++;
            break;
        case 5:
            parg->tt_out = file_content_out;
            flag++;
            break;
        case 6:
            parg->tz_out = file_content_out;
            flag++;
            break;
        case 7:
            parg->zr_out = file_content_out;
            flag++;
            break;
        case 8:
            parg->zt_out = file_content_out;
            flag++;
            break;
        case 9:
            parg->zz_out = file_content_out;
            flag++;
            break;
        }
    }

    fclose(file_out);

    // Check if exactly 9 lines were processed in output file
    if (flag < 9)
    {
        fprintf(stderr, "Insufficient lines in output file.\n");
        exit(EXIT_FAILURE);
    }
}
/* end of parsing command line arguments */

void usage()
{
    fprintf(stderr, "\nUsage:\n");
    fprintf(stderr, "RotateNCF\n");
    fprintf(stderr, "    -I input file list. Each line represents component pair EE EN EZ NE NN NZ ZE ZN ZE oder\n");
    fprintf(stderr, "    -O output file list. Each line represents RR RT RZ RZ TR TT TZ ZR ZT ZZ in order\n");

    fprintf(stderr, "\n\nOur Conventions for NCF\n");
    fprintf(stderr, "    if you cross correlate a with b to get ncf.sac,eg -E -N -Z for a and -e -n -z for station b\n");
    fprintf(stderr, "    then  we take a as source  and positon of a is saved as ncf.evlo and ncf.evla\n");
    fprintf(stderr, "    while we take b as station and positon of b is saved as ncf.stlo and ncf.stla\n");
    fprintf(stderr, "    so:\n");
    fprintf(stderr, "    1. In ncf.sac,the positive part stands for signal traveling from a to b\n");
    fprintf(stderr, "       and the negative part stands for signal traveling from b to a\n");
    fprintf(stderr, "    2. Azimuth is measured clockwise between vector North and vector Source->Station\n");
    fprintf(stderr, "       while Back-Azimuth  clockwise between vector North and vector Station->Source\n");
    fprintf(stderr, "    3. We currently using Net.Sta.Loc to identify one uniq station. for the headers of NCF:\n");
    fprintf(stderr, "       For Virt-Sta in NCF the Net.Sta.Loc and position is\n");
    fprintf(stderr, "            ncf.stla   = b.stla\n");
    fprintf(stderr, "            ncf.stlo   = b.stlo\n");
    fprintf(stderr, "       For Virt-Src in NCF the Net.Sta.Loc and position is\n");
    fprintf(stderr, "            ncf.evla   = a.stla\n");
    fprintf(stderr, "            ncf.evlo   = a.stlo\n");
    fprintf(stderr, "    Delete some usefule but redundant usage to suit the higher level python script calling this C program.\n");
    fprintf(stderr, "      \n");

    fprintf(stderr, "History: update by wangwt@20200223\n");
    fprintf(stderr, "Version: last update by wangjx@20230716\n");
    fprintf(stderr, "\n");
}