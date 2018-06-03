/*
 * MemStats.cc --
 *	Memory statistics.
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/MemStats.cc,v 1.5 2011/07/19 16:41:57 stolcke Exp $";
#endif

#include <string.h>

#include "MemStats.h"

const size_t MB = 1024 * 1024;

MemStats::MemStats()
    : total(0), wasted(0)
{
    clearAllocStats();
}

void
MemStats::clear()
{
    total = 0;
    wasted = 0;
    clearAllocStats();
}

void
MemStats::clearAllocStats()
{
    memset(allocStats, 0, sizeof(allocStats));
}

ostream &
MemStats::print(ostream &stream)
{
    stream << "total memory " << total
		  << " (" << ((float)total/MB) << "M)" 
		  << ", used " << (total - wasted)
		  << " (" << ((float)(total - wasted)/MB) << "M)"
		  << ", wasted " << wasted
		  << " (" << ((float)wasted/MB) << "M)"
		  << endl;

    for (unsigned size = 0; size < MAX_ALLOC_STATS; size ++) {
 	if (allocStats[size] > 0) {
	    stream << "allocations of size " << size << ": "
	           << allocStats[size] << endl;
	}
    }
    if (allocStats[MAX_ALLOC_STATS] > 0) {
	stream << "allocations of size >= " << MAX_ALLOC_STATS << ": "
	       << allocStats[MAX_ALLOC_STATS] << endl;
    }

    return stream;
}

