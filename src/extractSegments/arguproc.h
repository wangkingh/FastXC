#ifndef _ARGUPROC_H
#define _ARGUPROC_H
#include <stdio.h>
#define USAGE "[[-I:input file] [-O:output directory] [-W: the exponential term of the weight]"

typedef struct CommandLineOptions
{
    char infn[256];   // 假设输入文件名不会超过255字符
    char outdir[512]; // 输出目录同样限制为255字符
} CommandLineOptions;

void parse_arguments(int argc, char *argv[], CommandLineOptions *opts);
#endif
