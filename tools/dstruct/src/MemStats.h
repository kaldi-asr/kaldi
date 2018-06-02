/*
 * MemStats.h --
 *	Memory statistics.
 *
 * Copyright (c) 1995-2011 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/MemStats.h,v 1.4 2011/07/19 16:41:57 stolcke Exp $
 *
 */

#ifndef _MemStats_h_
#define _MemStats_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stddef.h>

#define MAX_ALLOC_STATS		256	/* keep track of allocations
					 * up to this size */

/*
 * The MemStats structure is used to return memory accounting 
 * information from the memstats() methods of various data types.
 */
class MemStats
{
public:
	MemStats();
	void clear();			/* reset memory stats */
	void clearAllocStats();		/* reset alloc stats */
	ostream &print(ostream &stream = cerr);
					/* print to cerr */

	size_t	total;			/* total allocated memory */
	size_t	wasted;			/* unused allocated memory */

	unsigned allocStats[MAX_ALLOC_STATS + 1];
					/* allocation units by size */
};

#endif /* _MemStats_h_ */
