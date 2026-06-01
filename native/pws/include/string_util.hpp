#ifndef STRING_UTIL_HPP
#define STRING_UTIL_HPP

#include <cctype>
#include <string>

static inline std::string trim_copy(const std::string &s)
{
    std::size_t first = 0;
    while (first < s.size() && std::isspace((unsigned char)s[first]))
        ++first;

    std::size_t last = s.size();
    while (last > first && std::isspace((unsigned char)s[last - 1]))
        --last;

    return s.substr(first, last - first);
}

#endif
