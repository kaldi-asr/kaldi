/*
 * testNBest --
 *	Test for NBest class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testNBest.cc,v 1.2 1999/08/01 09:22:47 stolcke Exp $";
#endif

#include <stdio.h>

#include "Vocab.h"
#include "NBest.h"

int
main(int argc, char *argv[])
{
    Vocab vocab;

    for (unsigned i = 1; argv[i] != 0; i ++) {
	NBestList nbestlist(vocab);
	File file(argv[i], "r");

	nbestlist.read(file);
	nbestlist.acousticNorm();
	nbestlist.acousticDenorm();
	// nbestlist.acousticOffset = 0.0;

	File sout(stdout);
	nbestlist.write(sout, false);
    }

    exit(0);
}
