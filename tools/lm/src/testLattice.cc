/*
 * testLattice --
 *	Test for WordLattice class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testLattice.cc,v 1.6 2010/06/02 05:49:58 stolcke Exp $";
#endif

#include <stdio.h>

#include "Vocab.h"
#include "WordLattice.h"
#include "Array.cc"

int
main (int argc, char *argv[])
{
    Vocab vocab;
    WordLattice lat(vocab);

    if (argc == 2) {
	File file(argv[1], "r");

	lat.read(file);
    }

    {
	char *line;
	File input(stdin);
	unsigned num = 0;

	while ((line = input.getline())) {
	    VocabString sentence[maxWordsPerLine + 1];
	    VocabIndex words[maxWordsPerLine + 1];

	    (void)vocab.parseWords(line, sentence, maxWordsPerLine + 1);
	    vocab.addWords(sentence, words, maxWordsPerLine);

	    if (num++ == 0) {
		lat.addWords(words, 1.0);
	    } else {
		lat.alignWords(words, 1.0);
	    }
	}
    }
    
    {
	File file(stdout);
	lat.write(file);
    }

    {
	makeArray(unsigned, sorted, lat.numNodes);
	unsigned reachable = lat.sortNodes(sorted);
	cerr << "sorted nodes: ";
	for (unsigned i = 0; i < reachable; i ++) {
	    cerr << " " << sorted[i];
	}
	cerr << endl;
    }

    exit(0);
}
