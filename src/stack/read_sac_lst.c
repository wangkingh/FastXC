#include "read_sac_lst.h"

// strdup designed by ChatGPT, in case that some compiler cannot use strdup
char* my_strdup(const char* s) {
    if (s == NULL) {
        return NULL;
    }
    char* new_str = (char*)malloc(strlen(s) + 1);
    if (new_str == NULL) {
        return NULL;
    }
    char* p = new_str;
    while (*s) {
        *p++ = *s++;
    }
    *p = '\0';
    return new_str;
}

// read sac file from giving list
FilePaths *read_sac_lst(char *sac_lst_file)
{
  FILE *fp;
  char line[MAXLINE];
  int count = 0;
  FilePaths *data = (FilePaths *)malloc(sizeof(FilePaths));
  data->paths = NULL;

  if (!(fp = fopen(sac_lst_file, "r")))
    return NULL;

  while (fgets(line, sizeof(line), fp))
  {
    // remove newline character from line, if present
    char *pos;
    if ((pos = strchr(line, '\n')) != NULL)
      *pos = '\0';

    // realloc memory to fit the new path
    char **new_paths = (char **)realloc(data->paths, (count + 1) * sizeof(char *));
    if (!new_paths){
        fprintf(stderr, "Memory reallocation failed!\n");
        freeFilePaths(data);
        fclose(fp);
        return NULL;
    }

    data->paths = new_paths;
    data->paths[count] = my_strdup(line); //Copy the string
    if (!data->paths[count]){ // Check if strdup failed
        fprintf(stderr,"Memory allocation for line failed!\n");
        freeFilePaths(data);
        fclose(fp);
        return NULL;
    }
    count ++;
  }

  fclose(fp);
  data->count = count;
  return data;
}

FilePaths *filter_by_npts(FilePaths *input)
{
    if (input == NULL || input->count == 0)
        return NULL;

    // Read the root of the spec file
    SACHEAD roothead;
    if (read_sachead(input->paths[0], &roothead) != 0)
    {
        fprintf(stderr, "Failed to read the root of segspec file %s\n", input->paths[0]);
        return NULL;
    }

    size_t rootnpts = roothead.npts;

    // Check the npts of each file ensuring that the npts are same
    size_t nInput = 0;
    size_t nValid = 0;

    // Create a new FilePaths structure to store the valid paths
    FilePaths *output = (FilePaths *)malloc(sizeof(FilePaths));
    output->paths = NULL;
    output->count = 0;
    int i;
    for (i = 0; i < input->count; i++)
    {
        SACHEAD currenthead;
        nInput++;

        if (read_sachead(input->paths[i], &currenthead) != 0)
        {
            fprintf(stderr, "Failed to read the root of segspec file %s\n", input->paths[i]);
            continue;
        }

        size_t current_npts = currenthead.npts;

        if (current_npts != rootnpts)
            continue;

        // If the file is valid, add it to the new FilePaths structure
        output->paths = (char **)realloc(output->paths, (nValid + 1) * sizeof(char *));
        output->paths[nValid] = (char *)malloc(MAXLINE);
        strcpy(output->paths[nValid], input->paths[i]);
        nValid++;
    }

    output->count = nValid;
    // printf("Valid/Input (Files) = %zu / %zu\n", nValid, nInput);

    return output;
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
