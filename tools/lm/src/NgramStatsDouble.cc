/*
 * NgramStatsFloat.cc --
 *	Instantiation of NgramCounts<double>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1996, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsDouble.cc,v 1.1 1999/10/23 06:16:01 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(double);
#endif

