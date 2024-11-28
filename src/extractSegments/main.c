#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> // Include for PRIu64 macro
#include <stdint.h>   // Include for uint64_t
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "sac.h"
#include "arguproc.h"

void CreateDir(const char *sPathName)
{
    char DirName[512];
    strcpy(DirName, sPathName);
    int i, len = strlen(DirName);
    for (i = 1; i < len; i++)
    {
        if (DirName[i] == '/')
        {
            DirName[i] = '\0';
            if (access(DirName, 0) != 0)
            {
                if (mkdir(DirName, 0755) == -1)
                {
                    // Check again if the directory exists after the error
                    if (access(DirName, 0) != 0)
                    {
                        printf("[INFO] Error creating %s. Permission denied\n", DirName);
                    }
                }
            }
            DirName[i] = '/';
        }
    }
    if (len > 0 && access(DirName, 0) != 0)
    {
        if (mkdir(DirName, 0755) == -1)
        {
            // Check again if the directory exists after the error
            if (access(DirName, 0) != 0)
            {
                printf("[INFO] Error creating %s. Permission denied\n", DirName);
            }
        }
    }
}

typedef struct TimeInfo
{
    int year;
    int jday;
    int hourminute;
} TimeInfo;

int main(int argc, char *argv[])
{
    CommandLineOptions options;
    parse_arguments(argc, argv, &options);
    const char *input_file = options.infn;
    const char *output_dir = options.outdir;

    FILE *fp_in = fopen(input_file, "rb");
    if (!fp_in)
    {
        fprintf(stderr, "Error: Unable to open input file %s\n", input_file);
        return 1;
    }

    // 读取匹配对数量
    size_t pair_count;
    if (fread(&pair_count, sizeof(pair_count), 1, fp_in) != 1)
    {
        fprintf(stderr, "Error: Failed to read pair count\n");
        fclose(fp_in);
        return 1;
    }
    printf("pair count is %lu\n", pair_count);

    // 读取时间信息
    TimeInfo *time_infos = malloc(sizeof(TimeInfo) * pair_count);
    if (time_infos == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp_in);
        return 1; // 内存分配失败
    }

    if (fread(time_infos, sizeof(TimeInfo), pair_count, fp_in) != pair_count)
    {
        fprintf(stderr, "Error: Failed to read time info\n");
        fclose(fp_in);
        free(time_infos);
        return 1; // 读取失败
    }

    SACHEAD sac_hd;
    if (fread(&sac_hd, sizeof(SACHEAD), 1, fp_in) != 1)
    {
        fprintf(stderr, "Error in reading SAC header %s\n", input_file);
        fclose(fp_in);
        return -1;
    }

    char output_path[2048];
    CreateDir(output_dir);
    float *data = malloc(sizeof(float) * sac_hd.npts); // 假设我们知道如何计算数据大小
    for (int file_idx = 0; file_idx < pair_count; file_idx++)
    {
        if (fread(data, sizeof(float), sac_hd.npts, fp_in) != sac_hd.npts)
        {
            fprintf(stderr, "Error: Failed to read data block\n");
            fclose(fp_in);
            free(time_infos);
            free(data);
            return 1;
        }
        snprintf(output_path, sizeof(output_path), "%s/%04d%03d%04d.sac", output_dir,
                 time_infos[file_idx].year, time_infos[file_idx].jday, time_infos[file_idx].hourminute);

        // 写入数据
        write_sac(output_path, sac_hd, data);
    }
    // 清理资源
    fclose(fp_in);
    free(time_infos);
    free(data);

    return 0;
}