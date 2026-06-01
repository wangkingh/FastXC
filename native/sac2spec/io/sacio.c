/*******************************************************************
*			sacio.c
* SAC I/O functions:
*	read_sachead	read SAC header
*	swab4		reverse byte order for integer/float
*	read_sac_buffer	read SAC samples into caller-owned buffer
*
* Legacy SAC helpers live in sacio_legacy.c.
*********************************************************************/


#include <stdio.h>
#include <string.h>
#include "sac.h"
#include "logger.h"

/* a SAC structure containing all null values */
SACHEAD sac_null = {
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345, -12345, -12345, -12345, -12345,
  -12345,      6, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }
};


/***********************************************************

  read_sachead

  Description:	read binary SAC header from file.

  Author:	Lupei Zhu

  Arguments:	const char *name 	file name
		SACHEAD *hd		SAC header to be filled

  Return:	0 if success, -1 if failed

  Modify history:
	05/29/97	Lupei Zhu	Initial coding
************************************************************/

int	read_sachead(const char	*name,
		SACHEAD		*hd
	)
{
  FILE		*strm;

  if ((strm = fopen(name, "rb")) == NULL) {
     LOG_ERROR("open_sac_failed", "path=\"%s\"", name);
     return -1;
  }

  if (fread(hd, sizeof(SACHEAD), 1, strm) != 1) {
     LOG_ERROR("read_sac_header_failed", "path=\"%s\"", name);
     fclose(strm);
     return -1;
  }

#ifdef BYTE_SWAP
  swab4((char *) hd, HD_SIZE);
#endif

  fclose(strm);
  return 0;

}


/*****************************************************

  swab4

  Description:	reverse byte order for float/integer

  Author:	Lupei Zhu

  Arguments:	char *pt	pointer to byte array
		int    n	number of bytes

  Return:	none

  Modify history:
	12/03/96	Lupei Zhu	Initial coding

************************************************************/

void	swab4(	char	*pt,
		int	n
	)
{
  int i;
  char temp;
  for(i=0;i<n;i+=4) {
    temp = pt[i+3];
    pt[i+3] = pt[i];
    pt[i] = temp;
    temp = pt[i+2];
    pt[i+2] = pt[i+1];
    pt[i+1] = temp;
  }
}



/***********************************************************
  read_sac_buffer

  Description:	read binary data from file. If succeed, it will return
                a float pointer to the data array.

  Author:	Lupei Zhu

  Modified by wang Jingxi

  Arguments:	const char *name 	file name
                int npts

  Return:	float pointer to the data array, NULL if failed

  Modify history:
        09/20/93	Lupei Zhu	Initial coding
        12/05/96	Lupei Zhu	adding error handling
        12/06/96	Lupei Zhu	swap byte-order on PC
************************************************************/

float *read_sac_buffer(const char *name, SACHEAD *sac_hd, float *buffer, int target_npts)
{
  FILE *strm;
  size_t read_npts;
  size_t read_bytes;

  if ((strm = fopen(name, "rb")) == NULL)
  {
    LOG_ERROR("open_sac_failed", "path=\"%s\"", name);
    return NULL;
  }

  if (target_npts <= 0)
  {
    LOG_ERROR("invalid_target_npts", "path=\"%s\" target_npts=%d", name, target_npts);
    fclose(strm);
    return NULL;
  }
  if (sac_hd->npts < 0)
  {
    LOG_ERROR("invalid_sac_npts", "path=\"%s\" npts=%d", name, sac_hd->npts);
    fclose(strm);
    return NULL;
  }

  if (fseek(strm, sizeof(SACHEAD), SEEK_SET) != 0)
  {
    LOG_ERROR("seek_sac_data_failed", "path=\"%s\"", name);
    fclose(strm);
    return NULL;
  }

  read_npts = (size_t)(sac_hd->npts < target_npts ? sac_hd->npts : target_npts);
  read_bytes = read_npts * sizeof(float);
  if (read_bytes > 0 && fread((char *)buffer, read_bytes, 1, strm) != 1)
  {
    LOG_ERROR("read_sac_data_failed", "path=\"%s\" bytes=%zu", name, read_bytes);
    fclose(strm);
    return NULL;
  }

  if (read_npts < (size_t)target_npts)
  {
    memset(buffer + read_npts, 0, ((size_t)target_npts - read_npts) * sizeof(float));
  }

  fclose(strm);

#ifdef BYTE_SWAP
  swab4((char *)buffer, (int)read_bytes);
#endif
  return buffer;
}

