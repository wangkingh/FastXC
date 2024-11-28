#include "read_segspec.h"

/* read the segment spectrum and return whole spec array
 * and header using preallocated buffer
 *
 * wuchao@20211004
 * */
complex *read_spec_buffer(char *name, SEGSPEC *hd, complex *buffer)
{
  FILE *strm = NULL;
  int size;

  if ((strm = fopen(name, "rb")) == NULL)
  {
    fprintf(stderr, "Unable to open %s\n", name);
    return NULL;
  }

  if (fread(hd, sizeof(SEGSPEC), 1, strm) != 1)
  {
    fprintf(stderr, "Error in reading SEGSPEC header %s\n", name);
    return NULL;
  }

  /* read whole segment spectrum in
   * Total size is nseg*nspec*sizeof(our_float_complex) */
  size = sizeof(complex) * hd->nspec * hd->nstep;

  if (fread((char *)buffer, size, 1, strm) != 1)
  {
    fprintf(stderr, "Error in reading SEGSPEC data %s\n", name);
    return NULL;
  }

  fclose(strm);

  return buffer;
}

/* read the segment spectrum header
 *
 * wuchao@20211004
 * */

int read_spechead(const char *name, SEGSPEC *hd)
{
  FILE *strm;
  if ((strm = fopen(name, "rb")) == NULL)
  {
    fprintf(stderr, "Unable to open %s\n", name);
    return -1;
  }

  if (fread(hd, sizeof(SEGSPEC), 1, strm) != 1)
  {
    fprintf(stderr, "Error in reading SAC header %s\n", name);
    fclose(strm);
    return -1;
  }

  fclose(strm);
  return 0;
}
