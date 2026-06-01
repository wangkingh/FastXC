#ifndef WRITE_MODE_H
#define WRITE_MODE_H

/* Direct modes materialize final .bigsac files in native code.
 * Pack mode writes job-local records and leaves final materialization to host code. */
#define MODE_APPEND 1
#define MODE_AGGREGATE 2
#define MODE_PACK 3

#endif
