/*
 * testNgram --
 *	Rudimentary test for Ngram LM class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testNgram.cc,v 1.2 2016/06/24 00:18:40 stolcke Exp $";
#endif

#include <stdio.h>

#include "Ngram.h"

int
main (int argc, char *argv[])
{
    Vocab vocab;
    Ngram ng(vocab, 3);

    int repeat = 1;

    if (argc >= 2) {

	if (argc >= 3) {
		repeat = atoi(argv[2]);

	}
	cerr << "reading " << argv[1] << " " << repeat << " time(s)\n";

	for (int i = 0; i < repeat; i ++) {
		File file(argv[1], "r");

		ng.read(file);
	}
    }

    if (argc >= 3) {
	exit(0);
    }

    cerr << "*** Ngram LM ***\n";
    {
	File file(stdout);

	ng.write(file);
    }

    Ngram ng_copy(ng);

    ng.clear();

    cerr << "*** Ngram LM (copy) ***\n";
    {
	File file(stdout);

	ng_copy.write(file);
    }

    cerr << "*** Ngram LM (cleared) ***\n";
    {
	File file(stderr);

	ng.write(file);
    }

    exit(0);
}
