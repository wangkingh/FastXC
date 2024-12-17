#include "read_filelist.h"

/* read in the .txt file list and output a pointer of path list */
PathNode *readPathList(const char *filename)
{
    FILE *fp;
    char line[MAXLINE];
    PathNode *pathList = NULL;
    PathNode *currPathNode;
    int i;
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error openning file%s\n", filename);
        exit(1);
    }

    while (fgets(line, MAXLINE, fp) != NULL)
    {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0'; // Add null terminator
        }

        // Check if the line is empty or contains only whitespace characters
        int isEmpty = 1;
        for (i = 0; i < len; i++)
        {
            if (!isspace(line[i]))
            {
                isEmpty = 0;
                break;
            }
        }
        if (isEmpty)
        {
            continue; // Skip empty or whitespace-only lines
        }
        currPathNode = (PathNode *)malloc(sizeof(PathNode));
        if (currPathNode == NULL)
        {
            fprintf(stderr, "Error allocating memory for path node\n");
            exit(1);
        }

        currPathNode->next = pathList;
        currPathNode->path = (char *)malloc(
            sizeof(char) * (MAXLINE + 1)); // Allocate memory for the path buffer
        if (currPathNode->path == NULL)
        {
            fprintf(stderr, "Error allocating memory for path string\n");
            exit(1);
        }
        strncpy(currPathNode->path, line,
                MAXLINE); // Copy the line contents into the path buffer
        pathList = currPathNode;
    }
    fclose(fp);
    return pathList;
}

void createDirectoryRecursively(char *dir)
{
    struct stat st = {0};
    if (stat(dir, &st) == -1)
    { // Check if directory exists
        char *parentDir = my_strdup(dir);
        parentDir = dirname(parentDir);

        // Recursively create parent directories
        createDirectoryRecursively(parentDir);

// Directory does not exist, create it
#ifdef _WIN32
        _mkdir(dir); // Use this on Windows
#else
        mkdir(dir, 0700); // Use this on Linux
#endif
    }
}

void createDirectories(PathNode *pathList)
{
    PathNode *currentNode = pathList;
    while (currentNode != NULL)
    {
        char *dirPath = my_strdup(currentNode->path);
        dirPath = dirname(dirPath);

        createDirectoryRecursively(dirPath);

        currentNode = currentNode->next; // Move to next node
    }
}

//  Convert the linked list to array
FilePathArray PathList2Array(PathNode *head)
{
    // Count nodes in the list
    size_t num_files = 0;
    PathNode *current = head;
    int i;

    while (current != NULL)
    {
        num_files++;
        current = current->next;
    }

    // Allocate memory for array
    char **file_paths = malloc(num_files * sizeof(char *));
    if (file_paths == NULL)
    {
        fprintf(stderr, "Error allocating memory for file_paths array\n");
        exit(1);
    }

    // Fill the array with paths from the linked list
    current = head;
    for (i = 0; i < num_files; i++)
    {
        file_paths[i] = current->path;
        current = current->next;
    }

    FilePathArray result = {file_paths, num_files};
    return result;
}