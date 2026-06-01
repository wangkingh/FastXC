#ifndef OUTPUT_LAYOUT_HPP
#define OUTPUT_LAYOUT_HPP

#include <string>

typedef struct OutputLayout
{
    std::string root;
    std::string spack_root;
    std::string spack_success_file;
    std::string progress_dir;
    std::string progress_file;
} OutputLayout;

int InitOutputLayout(const char *output_root, OutputLayout *layout);

#endif
