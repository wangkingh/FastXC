#ifndef _GEN_CCF_PATH_H
#define _GEN_CCF_PATH_H
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include "config.h"
#include "sac.h"
#include "segspec.h"
#include "util.h"
#include "cal_dist.h"

void CreateDir(char *sPathName);

void SplitFileName(const char *fname, const char *delimiter, char *stastr,
                   char *yearstr, char *jdaystr, char *hmstr, char *chnstr);

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength);

void GenCCFPath(char *ccf_path, char *src_path, char *sta_path, char *output_dir);

#endif