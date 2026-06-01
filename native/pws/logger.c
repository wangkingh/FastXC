#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static LogLevel g_log_level = LOG_LEVEL_INFO;
static int g_log_initialized = 0;

static const char *level_name(LogLevel level)
{
  switch (level)
  {
  case LOG_LEVEL_ERROR:
    return "ERROR";
  case LOG_LEVEL_WARN:
    return "WARN";
  case LOG_LEVEL_INFO:
    return "INFO";
  case LOG_LEVEL_DEBUG:
    return "DEBUG";
  default:
    return "LOG";
  }
}

static void logger_init_from_env(void)
{
  const char *env = NULL;
  if (g_log_initialized)
    return;
  g_log_initialized = 1;

  env = getenv("FASTXC_LOG_LEVEL");
  if (env == NULL || env[0] == '\0')
    env = getenv("PWS_LOG_LEVEL");
  if (env == NULL || env[0] == '\0')
    return;

  if (strcmp(env, "0") == 0 || strcasecmp(env, "ERROR") == 0)
    g_log_level = LOG_LEVEL_ERROR;
  else if (strcmp(env, "1") == 0 || strcasecmp(env, "WARN") == 0 ||
           strcasecmp(env, "WARNING") == 0)
    g_log_level = LOG_LEVEL_WARN;
  else if (strcmp(env, "2") == 0 || strcasecmp(env, "INFO") == 0)
    g_log_level = LOG_LEVEL_INFO;
  else if (strcmp(env, "3") == 0 || strcasecmp(env, "DEBUG") == 0)
    g_log_level = LOG_LEVEL_DEBUG;
}

void logger_set_level(LogLevel level)
{
  g_log_initialized = 1;
  g_log_level = level;
}

LogLevel logger_get_level(void)
{
  logger_init_from_env();
  return g_log_level;
}

void logger_log(LogLevel level, const char *event, const char *file, int line,
                const char *fmt, ...)
{
  va_list args;
  logger_init_from_env();
  if (level > g_log_level)
    return;

  flockfile(stderr);
  fprintf(stderr, "%s event=%s file=%s line=%d ",
          level_name(level), event == NULL ? "unknown" : event,
          file == NULL ? "unknown" : file, line);

  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  fputc('\n', stderr);
  fflush(stderr);
  funlockfile(stderr);
}
