#include "include/output_layout.hpp"

#include "include/logger.h"
#include "include/path_utils.h"

#include <stdlib.h>

static std::string JoinPathString(const std::string &root, const char *leaf)
{
    char *path = PathJoinAlloc(root.c_str(), leaf);
    std::string result = path == NULL ? std::string() : std::string(path);
    free(path);
    return result;
}

int InitOutputLayout(const char *output_root, OutputLayout *layout)
{
    if (layout == NULL)
    {
        LOG_ERROR("output_layout_missing", "layout=NULL");
        return -1;
    }

    layout->root = output_root == NULL ? std::string() : std::string(output_root);
    layout->stepack_root = JoinPathString(layout->root, "stepack");
    layout->stepack_success_file = JoinPathString(layout->stepack_root, "_SUCCESS");
    layout->progress_dir = JoinPathString(layout->root, "progress");
    layout->progress_file = JoinPathString(layout->progress_dir, "sac2spec_progress.tsv");

    if (layout->stepack_root.empty() || layout->stepack_success_file.empty() ||
        layout->progress_dir.empty() || layout->progress_file.empty() ||
        PathMakeDirectoryRecursive(layout->stepack_root.c_str()) != 0 ||
        PathMakeDirectoryRecursive(layout->progress_dir.c_str()) != 0)
    {
        LOG_ERROR("create_output_dirs_failed",
                  "output_root=\"%s\" stepack=\"%s\" progress=\"%s\"",
                  layout->root.c_str(), layout->stepack_root.c_str(),
                  layout->progress_dir.c_str());
        return -1;
    }

    return 0;
}
