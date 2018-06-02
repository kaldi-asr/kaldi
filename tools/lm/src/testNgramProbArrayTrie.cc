/*
 * testNgramProbArrayTrie --
 *	Rudimentary test for NgramProbArrayTrie class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testNgramProbArrayTrie.cc,v 1.3 2013/03/15 23:37:47 stolcke Exp $";
#endif

#include <stdio.h>

#include "NgramProbArrayTrie.h"

int
main (int argc, char *argv[])
{
    unsigned dim = 2;
    unsigned order = 3;

    if (argc >= 3) {
	dim = atol(argv[2]);

	if (argc >= 4) {
	    order = atol(argv[2]);
	}
    }

    Vocab vocab;
    NgramProbArrayTrie ng(vocab, order, dim);

    if (argc >= 2) {
	File file(argv[1], "r");

	if (!ng.read(file)) {
	    cerr << "bad format in " << argv[1] << endl;
	    exit(2);
	}
    }

    cerr << "*** NgramProbArrayTrie ***\n";
    {
	File file(stdout);

	ng.write(file, 0);
    }

    NgramProbArrayTrie ng_copy(ng);

    ng.clear();

    cerr << "*** NgramProbArrayTrie (copy) ***\n";
    {
	File file(stdout);

	ng_copy.write(file, 0);
    }

    cerr << "*** NgramProbArrayTrie (cleared) ***\n";
    {
	File file(stdout);

	ng.write(file, 0);
    }

    exit(0);
}
