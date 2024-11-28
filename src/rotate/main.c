// Last Update: 2023-07-16 wang jingxi
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>  // for errno and EEXIST
#include <math.h>
#include "sac.h"
#include "arguproc.h"
#include "rotate.h"
#include "hddistance.h"

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif
#include <unistd.h>

/* Define structure to hold data pointers, file names, and headers */
typedef struct
{
    float **data;
    char *fileName;
    SACHEAD *header;
} SacData;

int create_parent_dir(const char *path)
{
    char *path_copy = my_strdup(path);
    char *parent_dir = dirname(path_copy);

    if (access(parent_dir, F_OK) == -1)
    {
        create_parent_dir(parent_dir);
        if (mkdir(parent_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST)
        {
            free(path_copy);
            return -1;
        }
    }

    free(path_copy);
    return 0;
}

/* Function to read and check data */
void read_and_check_sac_data(SacData *sacData)
{
    if ((*sacData->data = read_sac(sacData->fileName, sacData->header)) == NULL)
    {
        fprintf(stderr, "reading data error. file:%s\n", sacData->fileName);
        exit(-1);
    }
}

void check_sac_data(SacData *sacData)
{
    if (sacData == NULL)
    {
        fprintf(stderr, "Error: sacData pointer is NULL.\n");
        exit(-1);
    }

    if (sacData->fileName == NULL)
    {
        fprintf(stderr, "Error: sacData->fileName is NULL.\n");
        exit(-1);
    }

    if (sacData->header == NULL)
    {
        fprintf(stderr, "Error: sacData->header is NULL.\n");
        exit(-1);
    }

    if (sacData->data == NULL)
    {
        fprintf(stderr, "Error: sacData->data is NULL.\n");
        exit(-1);
    }

    if (*(sacData->data) == NULL)
    {
        fprintf(stderr, "Error: sacData->data points to NULL.\n");
        exit(-1);
    }
}

// Function write data
void write_sac_data(SacData *sacData)
{
    if (write_sac(sacData->fileName, *(sacData->header), *(sacData->data)) == -1)
    {
        fprintf(stderr, "writing data error. file:%s\n", sacData->fileName);
        exit(-1);
    }
}

void print_rotate_matrix(double **matrix)
{
    printf("Rotate Matrix:\n");
    int i,j;
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            printf("%8.4f ", matrix[i][j]); // 使用8个字符宽度并保留4位小数来打印每个元素
        }
        printf("\n");
    }
}

/* end of subs */

int main(int argc, char **argv)
{
    ARGUTYPE argument;

    /* get argument */
    Argumentprocess(argc, argv, &argument);

    /* input file by -IEE -IEN etc */
    char *ncfenz[9] = {argument.ee_in, argument.en_in, argument.ez_in,
                       argument.ne_in, argument.nn_in, argument.nz_in,
                       argument.ze_in, argument.zn_in, argument.zz_in};

    /* output file by -ORR -ORZ etc */
    char *ncfrtz[9] = {argument.rr_out, argument.rt_out, argument.rz_out,
                       argument.tr_out, argument.tt_out, argument.tz_out,
                       argument.zr_out, argument.zt_out, argument.zz_out};

    create_parent_dir(argument.zz_out);

    /* head in */
    SACHEAD hdee = sac_null, hden = sac_null, hdez = sac_null;
    SACHEAD hdne = sac_null, hdnn = sac_null, hdnz = sac_null;
    SACHEAD hdze = sac_null, hdzn = sac_null, hdzz = sac_null;

    /* head out */
    SACHEAD hd_out = sac_null; /* set to null to kill warning will be updated by zz */

    /* angle for rotation matrix after cmpaz correction */
    float rotate_az, rotate_baz;

    /* Find sac headers form zz sac file*/
    SACHEAD template_hd = sac_null;
    read_sachead(argument.zz_in, &template_hd);
    float delta = template_hd.delta;
    int npts = template_hd.npts;

    /* Set up arrays for data pointers */
    float *enz_data[9], *rtz_data[9];
    int i;
    for (i = 0; i < 9; i++)
    {
        enz_data[i] = (float *)malloc(sizeof(float) * npts);
        if (enz_data[i] == NULL)
        {
            fprintf(stderr, "Memory allocation failed for enz_data[%d]\n", i);
            exit(-1);
        }

        rtz_data[i] = (float *)malloc(sizeof(float) * npts);
        if (rtz_data[i] == NULL)
        {
            fprintf(stderr, "Memory allocation failed for rtz_data[%d]\n", i);
            exit(-1);
        }
    }

    SACHEAD *headers[9] = {&hdee, &hden, &hdez, &hdne, &hdnn, &hdnz, &hdze, &hdzn, &hdzz};

    /* Read in and check data */
    for (i = 0; i < 9; i++)
    {
        SacData sacData = {&enz_data[i], ncfenz[i], headers[i]};
        read_and_check_sac_data(&sacData);
    }

    if (delta != hdee.delta || npts != hdee.npts ||
        delta != hden.delta || npts != hden.npts ||
        delta != hdez.delta || npts != hdez.npts ||
        delta != hdne.delta || npts != hdne.npts ||
        delta != hdnn.delta || npts != hdnn.npts ||
        delta != hdnz.delta || npts != hdnz.npts ||
        delta != hdze.delta || npts != hdze.npts ||
        delta != hdzn.delta || npts != hdzn.npts)
    {
        printf("ee delta %f npts %d\n", hdee.delta, hdee.npts);
        printf("en delta %f npts %d\n", hden.delta, hden.npts);
        printf("ez delta %f npts %d\n", hdez.delta, hdez.npts);
        printf("ne delta %f npts %d\n", hdne.delta, hdne.npts);
        printf("nn delta %f npts %d\n", hdnn.delta, hdnn.npts);
        printf("nz delta %f npts %d\n", hdnz.delta, hdnz.npts);
        printf("ze delta %f npts %d\n", hdze.delta, hdze.npts);
        printf("zn delta %f npts %d\n", hdzn.delta, hdzn.npts);
        printf("zz delta %f npts %d\n", hdzz.delta, hdzz.npts);
        fprintf(stderr, "In constency of input nine tensors NCF files\n");
        exit(-1);
    }

    /* we copy zz to hdout and update for each tensor later  */
    hd_out = template_hd;

    /* recalculate the azi and baz
     * get azimuth from stlo stla of input source and station, updated to
     * Rudoe method by wangwt@20130906
     * prototype of new distaz routine is
     * void hd_distaz(float evlo,float evla,float stlo,float stla, float *gcarc,float *az,float *baz,float *distkm)
     */
    hd_distaz(hd_out.evlo, hd_out.evla, hd_out.stlo, hd_out.stla,
              &hd_out.gcarc, &hd_out.az, &hd_out.baz, &hd_out.dist);

    /* create elements of rotation matrix */

    /* As indicated by previous note, the rotation matrix is related to the real angle when you
     * rotate the E/N into R/T component. If E/N is exactly pointing to E/N, these two angle will
     * be azimuth and backazimuth-180.0. For those which cmpaz is not standard, we need subtract
     * the angle by the cmpaz of N component, whose cmpaz should be 0.0 for standard one.
     *
     * The true azimuth     angle is corrected by cmpaz of N component of virt source
     * The true backazimuth angle is corrected by cmpaz of N component of virt station
     *
     * Note sin and cos receive radians.
     *
     * wangwt@20160710 */

    // Assume that the cmpaz of N component is 0.0
    rotate_az = hd_out.az * M_PI / 180.0;
    rotate_baz = hd_out.baz * M_PI / 180.0;

    // generate rotate matrix
    double **rotate_matrix = malloc(9 * sizeof(double *));
    for (i = 0; i < 9; i++)
    {
        rotate_matrix[i] = malloc(9 * sizeof(double));
    }

    generate_rotate_matrix(rotate_az, rotate_baz, rotate_matrix);
    // print_rotate_matrix(rotate_matrix);

    rotate(rtz_data, enz_data, rotate_matrix, npts);

    hd_out.cmpaz = (hd_out.baz - 180.0 < 0.0) ? (hd_out.baz + 180.0) : (hd_out.baz - 180.0);
    hd_out.cmpinc = 90.0;
    char *component_strings[] = {"RR", "RT", "RZ", "TR", "TT", "TZ", "ZR", "ZT", "ZZ"};

    // writing output data
    for (i = 0; i < 9; i++)
    {
        strcpy(hd_out.kcmpnm, component_strings[i]);
        SacData sacData = {&rtz_data[i], ncfrtz[i], &hd_out};
        check_sac_data(&sacData);
        write_sac_data(&sacData);
    }

    /* Free memomry */
    for (i = 0; i < 9; i++)
    {
        free(enz_data[i]);
        free(rtz_data[i]);
        free(ncfenz[i]);
        free(ncfrtz[i]);
    }

    for (i = 0; i < 9; i++)
    {
        free(rotate_matrix[i]);
    }
    free(rotate_matrix);

    return 0;
}
