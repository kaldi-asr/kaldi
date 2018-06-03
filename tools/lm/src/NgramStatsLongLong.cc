/*
 * NgramStatsLongLong.cc --
 *	Instantiation of NgramCounts<unsigned long long>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2006, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsLongLong.cc,v 1.1 2006/07/29 11:06:30 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(unsigned long long);
#endif

