#ifndef _SAC2SPEC_LOGGER_H
#define _SAC2SPEC_LOGGER_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LogLevel
{
    LOG_LEVEL_ERROR = 0,
    LOG_LEVEL_WARN = 1,
    LOG_LEVEL_INFO = 2,
    LOG_LEVEL_DEBUG = 3
} LogLevel;

void logger_set_level(LogLevel level);
LogLevel logger_get_level(void);
void logger_log(LogLevel level, const char *event, const char *file, int line,
                const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#define LOG_ERROR(event, fmt, ...) \
    logger_log(LOG_LEVEL_ERROR, event, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(event, fmt, ...) \
    logger_log(LOG_LEVEL_WARN, event, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(event, fmt, ...) \
    logger_log(LOG_LEVEL_INFO, event, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(event, fmt, ...) \
    logger_log(LOG_LEVEL_DEBUG, event, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif
