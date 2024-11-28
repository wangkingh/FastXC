#include "sac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#include <sys/time.h>
#include "read_sac_lst.h"
#include "arguproc.h"
#include "util.h"

#define K_LEN_8 8
#define K_LEN_16 16

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

int main(int argc, char **argv)
{
  char *sac_lst = NULL;
  char *out_sac = NULL;

  // Argument parse
  ARGUTYPE argument;
  ArgumentProcess(argc, argv, &argument);

  sac_lst = argument.sac_lst;
  out_sac = argument.out_sac;
  int normalize = argument.normalize;

  FilePaths *ncfPathsRaw = read_sac_lst(sac_lst);
  size_t ncf_num_raw = ncfPathsRaw->count;
  if (ncf_num_raw == 0)
  {
    fprintf(stderr, "ERROR: No sac files in %s\n", sac_lst);
    return -1;
  }
  FilePaths *ncfPaths = filter_by_npts(ncfPathsRaw); /* Commnented by wangjx 2024.01.08 to accelerate*/
  size_t ncf_num = ncfPaths->count;
  if (ncf_num == 0)
  {
    fprintf(stderr, "ERROR: No sac files with npts > 0 in %s\n", sac_lst);
    return -1;
  }

  SACHEAD template_hd = sac_null, infilehd = sac_null;

  size_t nstack = 0;
  size_t k = 0;

  /* read first sac head */
  if (read_sachead(ncfPaths->paths[0], &template_hd) != 0)
  {
    fprintf(stderr, "ERROR in read header for %s\n", ncfPaths->paths[0]);
    return -1;
  }

  /* Get the original path */
  char *original_path = ncfPaths->paths[0];

  /* Create a new char array and copy the original path into it */
  char template_path[256]; // Adjust the size as needed
  strcpy(template_path, original_path);

  char *base_name = basename(template_path);

  /* Extract the required fields */
  char *fields[5];
  int i = 0;
  char *token = strtok(base_name, ".");
  while (token != NULL)
  {
    fields[i++] = token;
    token = strtok(NULL, ".");
  }

  char *sta_pair = fields[0];
  char *component_pair = fields[1];

  char *rest = sta_pair;
  char *saveptr;

  token = strtok_r(rest, "-", &saveptr);
  char *kevnm = strtok(sta_pair, "-");
  rest = NULL;
  char *kstnm = strtok_r(rest, "-", &saveptr);

  /* Write fields to the sac header */
  strncpy(template_hd.kstnm, kstnm, K_LEN_8 - 1);
  template_hd.kstnm[K_LEN_8 - 1] = '\0';

  strncpy(template_hd.kevnm, kevnm, K_LEN_16 - 1);
  template_hd.kevnm[K_LEN_16 - 1] = '\0';

  strncpy(template_hd.kcmpnm, component_pair, K_LEN_8 - 1);
  template_hd.kcmpnm[K_LEN_8 - 1] = '\0';

  /* take the 1st file's header as the output headr */
  int npts = template_hd.npts;
  SACHEAD hdstack = template_hd;

  /* change the reference time nzyear nzjday nzhour nzmin nzsec nzmsec */
  hdstack.nzyear = 2010;
  hdstack.nzjday = 214;
  hdstack.nzhour = 16;
  hdstack.nzmin = 0;
  hdstack.nzsec = 0;
  hdstack.nzmsec = 0;

  /* Copy coordinate infomation from first sac file */
  hdstack.stla = template_hd.stla;
  hdstack.stlo = template_hd.stlo;
  hdstack.evla = template_hd.evla;
  hdstack.evlo = template_hd.evlo;

  hdstack.dist = template_hd.dist;
  hdstack.az = template_hd.az;
  hdstack.baz = template_hd.baz;
  hdstack.gcarc = template_hd.gcarc;

  float *infilecc = NULL;
  float *stackcc = NULL;
  infilecc = (float *)malloc(sizeof(float) * npts);
  stackcc = (float *)malloc(sizeof(float) * npts);
  nstack = 0;

  // set stackcc to zero
  for (k = 0; k < npts; k++)
  {
    stackcc[k] = 0.0;
  }

  for (i = 0; i < ncf_num; i++)
  {
    // print path
    if (read_sac_buffer(ncfPaths->paths[i], &infilehd, infilecc) == NULL)
    {
      fprintf(stderr, "ERROR Reading data error for %s\n", ncfPaths->paths[i]);
      continue;
    }
    /* sum up the infilecc into stackcc */
    for (k = 0; k < npts; k++)
    {
      stackcc[k] = stackcc[k] + infilecc[k];
    }
    nstack++;
  }
  /* end of stack all the effective file in the list */

  /* Normalize the stackcc array, nstack >=1 after former check */
  if (normalize == 1)
  {
    for (k = 0; k < npts; k++)
    {
      stackcc[k] /= ncf_num;
    }
  }

  /* Save nstack into hd.unused27  add by wangwt@20130408 */
  hdstack.unused27 = nstack;
  char *out_sac_copy = my_strdup(out_sac);
  if (create_parent_dir(out_sac) == -1)
  {
    fprintf(stderr, "Error creating directory %s: ", dirname(out_sac_copy));
    perror(NULL);
    free(out_sac_copy);
    return 1;
  }
  write_sac(out_sac, hdstack, stackcc);

  /* clean up */
  free(infilecc);
  free(stackcc);

  freeFilePaths(ncfPathsRaw);
  freeFilePaths(ncfPaths);

  return (0);
}
