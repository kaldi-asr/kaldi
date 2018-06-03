/*
 * matherr.c --
 *	Math error handling
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1996-2011 SRI International, 2012 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/matherr.c,v 1.7 2012/07/07 05:56:44 stolcke Exp $";
#endif

#include <math.h>
#include <string.h>

#if defined(SING) && !defined(WIN32)
int
#if defined(_MSC_VER)
_matherr(struct _exception *x)
#else
matherr(struct exception *x)
#endif
{
    if (x->type == SING && strcmp(x->name, "log10") == 0) {
	/*
	 * suppress warnings about log10(0.0)
	 */
	return 1;
    } else {
	return 0;
    }
}
#endif /* SING && !WIN32 */

