#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "sac.h"

float *read_sac(const char *name, SACHEAD *hd)
{
  FILE *strm;
  float *ar;
  unsigned sz;

  if ((strm = fopen(name, "rb")) == NULL) {
     LOG_ERROR("open_sac_failed", "path=\"%s\"", name);
     return NULL;
  }

  if (fread(hd, sizeof(SACHEAD), 1, strm) != 1) {
     LOG_ERROR("read_sac_header_failed", "path=\"%s\"", name);
     fclose(strm);
     return NULL;
  }

#ifdef BYTE_SWAP
  swab4((char *) hd, HD_SIZE);
#endif

  sz = hd->npts * sizeof(float);
  if ((ar = (float *) malloc(sz)) == NULL) {
     LOG_ERROR("alloc_failed", "target=read_sac_data path=\"%s\" bytes=%u", name, sz);
     fclose(strm);
     return NULL;
  }

  if (fread((char *) ar, sz, 1, strm) != 1) {
     LOG_ERROR("read_sac_data_failed", "path=\"%s\" bytes=%u", name, sz);
     free(ar);
     fclose(strm);
     return NULL;
  }

  fclose(strm);

#ifdef BYTE_SWAP
  swab4((char *) ar, sz);
#endif

  return ar;
}

int write_sac(const char *name, SACHEAD hd, const float *ar)
{
  FILE *strm = NULL;
  unsigned sz;
  float *data;
  int error = 0;

  sz = hd.npts * sizeof(float);
  if (hd.iftype == IXY) sz *= 2;

  if ((data = (float *) malloc(sz)) == NULL) {
     LOG_ERROR("alloc_failed", "target=write_sac_data path=\"%s\" bytes=%u", name, sz);
     error = 1;
  }

  if (!error && memcpy(data, ar, sz) == NULL) {
     LOG_ERROR("copy_sac_data_failed", "path=\"%s\" bytes=%u", name, sz);
     error = 1;
  }

#ifdef BYTE_SWAP
  if (!error) {
     swab4((char *) data, sz);
     swab4((char *) &hd, HD_SIZE);
  }
#endif

  if (!error && (strm = fopen(name, "w")) == NULL) {
     LOG_ERROR("open_sac_write_failed", "path=\"%s\"", name);
     error = 1;
  }

  if (!error && fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1) {
     LOG_ERROR("write_sac_header_failed", "path=\"%s\"", name);
     error = 1;
  }

  if (!error && fwrite(data, sz, 1, strm) != 1) {
     LOG_ERROR("write_sac_data_failed", "path=\"%s\" bytes=%u", name, sz);
     error = 1;
  }

  free(data);
  if (strm != NULL) fclose(strm);

  return (error == 0) ? 0 : -1;
}

SACHEAD sachdr(float dt, int ns, float b0)
{
  SACHEAD hd = sac_null;
  hd.npts = ns;
  hd.delta = dt;
  hd.b = b0;
  hd.o = 0.;
  hd.e = b0 + (ns - 1) * hd.delta;
  hd.iztype = IO;
  hd.iftype = ITIME;
  hd.leven = TRUE;
  return hd;
}

int wrtsac2(const char *name, int n, const float *x, const float *y)
{
  SACHEAD hd = sac_null;
  float *ar;
  unsigned sz;
  int exit_code;

  hd.npts = n;
  hd.iftype = IXY;
  hd.leven = FALSE;

  sz = n * sizeof(float);

  if ((ar = (float *) malloc(2 * sz)) == NULL) {
     LOG_ERROR("alloc_failed", "target=wrtsac2_data path=\"%s\" bytes=%u", name, 2 * sz);
     return -1;
  }

  if (memcpy(ar, x, sz) == NULL) {
     LOG_ERROR("copy_wrtsac2_x_failed", "path=\"%s\" bytes=%u", name, sz);
     free(ar);
     return -1;
  }
  if (memcpy((char *)ar + sz, y, sz) == NULL) {
     LOG_ERROR("copy_wrtsac2_y_failed", "path=\"%s\" bytes=%u", name, sz);
     free(ar);
     return -1;
  }

  exit_code = write_sac(name, hd, ar);

  free(ar);

  return exit_code;
}

void rdsac0_(const char *name, float *dt, int *ns, float *b0, float *ar)
{
   int i;
   SACHEAD hd;
   float *temp;
   temp = read_sac(name, &hd);
   *dt = hd.delta;
   *ns = hd.npts;
   *b0 = hd.b;
   for (i = 0; i < *ns; i++) ar[i] = temp[i];
   free(temp);
}

void my_brsac_(char *name, float *hdr, int *hdi, char *hdc, float *ar, int *err)
{
   int i, *ipt;
   float *temp;
   char *cpt;
   cpt = strchr(name, (int) ' ');
   *cpt = 0;
   *err = 0;
   temp = read_sac(name, (SACHEAD *)hdr);
   if (temp == NULL) {
      *err = -1;
      return;
   }
   ipt = (int *) (hdr + 70);
   for (i = 0; i < 30; i++) hdi[i] = ipt[i];
   cpt = (char *) (hdr + 110);
   for (i = 0; i < 192; i++) hdc[i] = cpt[i];
   for (i = 0; i < hdi[9]; i++) ar[i] = temp[i];
   free(temp);
}

void wrtsac0_(const char *name, float *dt, int *ns, float *b0, float *dist, const float *ar)
{
  SACHEAD hd;
  hd = sachdr(*dt, *ns, *b0);
  hd.dist = *dist;
  write_sac(name, hd, ar);
}

void wrtsac2_(const char *name, int n, const float *x, const float *y)
{
  wrtsac2(name, n, x, y);
}

void wrtsac3_(const char *name, float dt, int ns, float b0, float dist, float cmpaz, float cmpinc, const float *ar)
{
  SACHEAD hd;
  hd = sachdr(dt, ns, b0);
  hd.dist = dist;
  hd.cmpaz = cmpaz;
  hd.cmpinc = cmpinc;
  write_sac(name, hd, ar);
}

float *read_sac2(const char *name, SACHEAD *hd, int tmark, float t1, float t2)
{
  FILE *strm;
  int nn, nt1, nt2, npts;
  float tref, *ar, *fpt;

  if ((strm = fopen(name, "rb")) == NULL) {
     LOG_ERROR("open_sac_failed", "path=\"%s\"", name);
     return NULL;
  }

  if (fread(hd, sizeof(SACHEAD), 1, strm) != 1) {
     LOG_ERROR("read_sac_header_failed", "path=\"%s\"", name);
     fclose(strm);
     return NULL;
  }

#ifdef BYTE_SWAP
  swab4((char *) hd, HD_SIZE);
#endif

  nn = (int) rint((t2 - t1) / hd->delta);
  if (nn <= 0 || (ar = (float *) calloc(nn, sizeof(float))) == NULL) {
     LOG_ERROR("alloc_failed", "target=read_sac2_data path=\"%s\" samples=%d", name, nn);
     fclose(strm);
     return NULL;
  }
  tref = 0.;
  if (tmark == -5 || tmark == -3 || tmark == -2 || (tmark >= 0 && tmark < 10)) {
     tref = *((float *) hd + 10 + tmark);
     if (tref == -12345.) {
        LOG_ERROR("sac_time_mark_undefined", "path=\"%s\" tmark=%d", name, tmark);
        free(ar);
        fclose(strm);
        return NULL;
     }
  }
  t1 += tref;
  nt1 = (int) rint((t1 - hd->b) / hd->delta);
  nt2 = nt1 + nn;
  npts = hd->npts;
  hd->npts = nn;
  hd->b = t1;
  hd->e = t1 + nn * hd->delta;

  if (nt1 >= npts || nt2 < 0) {
     fclose(strm);
     return ar;
  }

  if (nt1 < 0) {
     fpt = ar - nt1;
     nt1 = 0;
  } else {
     if (fseek(strm, nt1 * sizeof(float), SEEK_CUR) < 0) {
        LOG_ERROR("seek_sac_data_failed", "path=\"%s\" offset_samples=%d", name, nt1);
        free(ar);
        fclose(strm);
        return NULL;
     }
     fpt = ar;
  }
  if (nt2 > npts) nt2 = npts;
  nn = nt2 - nt1;
  if (fread((char *) fpt, sizeof(float), nn, strm) != (size_t)nn) {
     LOG_ERROR("read_sac_data_failed", "path=\"%s\" samples=%d", name, nn);
     free(ar);
     fclose(strm);
     return NULL;
  }
  fclose(strm);

#ifdef BYTE_SWAP
  swab4((char *) fpt, nn * sizeof(float));
#endif

  return ar;
}

void ResetSacTime(SACHEAD *hd)
{
     hd->o = 0.;
     hd->a = 0.;
     hd->nzyear = -12345;
     hd->nzjday = -12345;
     hd->nzhour = -12345;
     hd->nzmin = -12345;
     hd->nzsec = -12345;
     hd->nzmsec = -12345;
}

int sac_head_index(const char *name)
{
  if (strcmp(name, "delta") == 0) return 0;
  else if (strcmp(name, "depmin") == 0) return 1;
  else if (strcmp(name, "depmax") == 0) return 2;
  else if (strcmp(name, "b") == 0) return 5;
  else if (strcmp(name, "e") == 0) return 6;
  else if (strcmp(name, "o") == 0) return 7;
  else if (strcmp(name, "a") == 0) return 8;
  else if (strcmp(name, "t0") == 0) return 10;
  else if (strcmp(name, "t1") == 0) return 11;
  else if (strcmp(name, "t2") == 0) return 12;
  else if (strcmp(name, "t3") == 0) return 13;
  else if (strcmp(name, "t4") == 0) return 14;
  else if (strcmp(name, "t5") == 0) return 15;
  else if (strcmp(name, "t6") == 0) return 16;
  else if (strcmp(name, "t7") == 0) return 17;
  else if (strcmp(name, "t8") == 0) return 18;
  else if (strcmp(name, "t9") == 0) return 19;
  else if (strcmp(name, "stla") == 0) return 31;
  else if (strcmp(name, "stlo") == 0) return 32;
  else if (strcmp(name, "stel") == 0) return 33;
  else if (strcmp(name, "stdp") == 0) return 34;
  else if (strcmp(name, "evla") == 0) return 35;
  else if (strcmp(name, "evlo") == 0) return 36;
  else if (strcmp(name, "evel") == 0) return 37;
  else if (strcmp(name, "evdp") == 0) return 38;
  else if (strcmp(name, "user0") == 0) return 40;
  else if (strcmp(name, "user1") == 0) return 41;
  else if (strcmp(name, "user2") == 0) return 42;
  else if (strcmp(name, "user3") == 0) return 43;
  else if (strcmp(name, "user4") == 0) return 44;
  else if (strcmp(name, "user5") == 0) return 45;
  else if (strcmp(name, "user6") == 0) return 46;
  else if (strcmp(name, "user7") == 0) return 47;
  else if (strcmp(name, "user8") == 0) return 48;
  else if (strcmp(name, "user9") == 0) return 49;
  else if (strcmp(name, "dist") == 0) return 50;
  else if (strcmp(name, "az") == 0) return 51;
  else if (strcmp(name, "baz") == 0) return 52;
  else if (strcmp(name, "gcarc") == 0) return 53;
  else if (strcmp(name, "depmen") == 0) return 56;
  else if (strcmp(name, "cmpaz") == 0) return 57;
  else if (strcmp(name, "cmpinc") == 0) return 58;
  else if (strcmp(name, "kztime") == 0) return 70;
  else if (strcmp(name, "npts") == 0) return 79;
  else if (strcmp(name, "kstnm") == 0) return 110;
  else return -1;
}
