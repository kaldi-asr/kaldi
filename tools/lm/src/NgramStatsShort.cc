/*
 * NgramStatsShort.cc --
 *	Instantiation of NgramCounts<unsigned short>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsShort.cc,v 1.1 2005/09/23 19:27:45 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(unsigned short);
#endif

