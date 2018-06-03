/*
 * NgramStatsFloat.cc --
 *	Instantiation of NgramCounts<float>
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1999, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/NgramStatsFloat.cc,v 1.4 1999/10/23 06:15:46 stolcke Exp $";
#endif

#include "NgramStats.cc"
#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_NGRAMCOUNTS(float);
#endif

