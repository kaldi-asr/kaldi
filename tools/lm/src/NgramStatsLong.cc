/*
 * NgramStatsLong.cc --
 *	Instantiation of NgramCounts<unsigned long>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1996, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsLong.cc,v 1.1 2006/07/29 00:30:57 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(unsigned long);
#endif

