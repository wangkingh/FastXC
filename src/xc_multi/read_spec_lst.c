#include "read_spec_lst.h"

#include <stdio.h>

FilePaths *read_spec_lst(char *spec_lst_file)
{
  FILE *fp;
  char line[MAXLINE];
  int count = 0;
  FilePaths *data = (FilePaths *)malloc(sizeof(FilePaths));
  data->paths = NULL;

  if (!(fp = fopen(spec_lst_file, "r")))
    return NULL;

  while (fgets(line, sizeof(line), fp))
  {
    // remove newline character from line, if present
    char *pos;
    if ((pos = strchr(line, '\n')) != NULL)
      *pos = '\0';

    // Skip empty lines
    if (line[0] == '\0')
      continue;

    // realloc memory to fit the new path
    data->paths = (char **)realloc(data->paths, (count + 1) * sizeof(char *));
    data->paths[count] = my_strdup(line); // copy the string
    count++;
  }

  fclose(fp);
  data->count = count;
  return data;
}

void freeFilePaths(FilePaths *fp)
{
  int i;
  if (fp == NULL)
  {
    return;
  }

  if (fp->paths != NULL)
  {
    for (i = 0; i < fp->count; i++)
    {
      free(fp->paths[i]);
    }
    free(fp->paths);
  }

  free(fp);
}
