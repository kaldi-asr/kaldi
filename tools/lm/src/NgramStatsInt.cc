/*
 * NgramStatsInt.cc --
 *	Instantiation of NgramCounts<unsigned>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1996, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsInt.cc,v 1.4 2005/09/23 19:27:45 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(unsigned);
#endif

