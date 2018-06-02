/*
 * NgramStatsXCount.cc --
 *	Instantiation of NgramCounts<XCount>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsXCount.cc,v 1.1 2005/09/25 04:34:57 stolcke Exp $";
#endif

#include "XCount.h"
#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(XCount);
#endif

