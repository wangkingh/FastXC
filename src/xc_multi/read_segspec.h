#ifndef _RDSPEC_H
#define _RDSPEC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "complex.h"
#include "segspec.h"

// Read in a spec file and store in buffer
complex *read_spec_buffer(char *name, SEGSPEC *hd, complex *buffer);

// Read in a spec file header
int read_spechead(const char *name, SEGSPEC *hd);

#endif
