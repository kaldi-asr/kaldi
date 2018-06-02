/*
 * maxalloc --
 * 	Check how much memory can be allocated
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2010-2011 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/maxalloc.c,v 1.3 2011/11/21 02:41:58 stolcke Exp $";
#endif

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "option.h"

int debug = 0;

static Option options[] = {
    { OPT_INT, "debug", &debug, "debug level" },
};

#ifndef SIZE_MAX
# define SIZE_MAX	ULONG_MAX
#endif

int
main(int argc, char **argv)
{
    size_t nbytes, minsize, maxsize;
    void *mem;

    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (debug > 0) {
	fprintf(stderr, "SIZE_MAX = %llu bytes (%.1f Mbytes, %.1f Gbytes)\n",
				(unsigned long long)SIZE_MAX,
				SIZE_MAX / (1024.0 * 1024.0),
				SIZE_MAX / (1024.0 * 1024.0 * 1024.0));
    }

    /* 
     * Search for the largest power of 2 size that can be allocated
     */
    for (nbytes = 1; nbytes < SIZE_MAX/2; nbytes *= 2) {
	mem = malloc(nbytes * 2);

	if (debug > 0) {
	    fprintf(stderr, "trying %llu -- %s\n", (unsigned long long)nbytes * 2,
					mem == 0 ? "FAILED" : "SUCCESS");
	}

	if (mem == 0) break;
	
	free(mem);
    }


    /* 
     * Now do a binary search to find the upper bound
     */
    minsize = nbytes;
    if (nbytes < SIZE_MAX/2) {
	maxsize = nbytes * 2;
    } else {
	maxsize = SIZE_MAX;
    }

    while (minsize < maxsize - 1) {
	nbytes = minsize + (maxsize - minsize)/2;

	mem = malloc(nbytes);

	if (mem == 0) {
	    maxsize = nbytes - 1;
	} else {
	    minsize = nbytes;
	}

	if (debug > 0) {
	    fprintf(stderr, "trying %llu -- %s\n", (unsigned long long)nbytes,
					mem == 0 ? "FAILED" : "SUCCESS");
	}

	free(mem);
    }

    printf("managed to allocate %llu bytes (%f Mbytes, %f Gbytes)\n",
				(unsigned long long)nbytes,
				nbytes / (1024.0 * 1024.0),
				nbytes / (1024.0 * 1024.0 * 1024.0));

    exit(0);
}

