/*
 * testError --
 *	Test for WordAlign class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testError.cc,v 1.3 2010/06/02 05:49:58 stolcke Exp $";
#endif

#include <stdio.h>

#include "Vocab.h"
#include "WordAlign.h"

int
main (int argc, char *argv[])
{
    Vocab vocab;

    char *line;
    File input(stdin);
    unsigned num = 0;

    VocabIndex ref[maxWordsPerLine + 1];

    while ((line = input.getline())) {
	VocabString sentence[maxWordsPerLine + 1];
	VocabIndex hyp[maxWordsPerLine + 1];

	(void)vocab.parseWords(line, sentence, maxWordsPerLine + 1);

	if (num++ == 0) {
	    /*
	     * First line -- use a ref string
	     */
	    vocab.addWords(sentence, ref, maxWordsPerLine);
	} else {
	    /*
	     * Subsequent lines -- compute errors against ref
	     */
	    vocab.addWords(sentence, hyp, maxWordsPerLine);

	    unsigned total, sub, ins, del;
	    total = wordError(ref, hyp, sub, ins, del);

	    cout << "sub " << sub 
		 << " ins " << ins
		 << " del " << del
		 << " wer " << total
		 << endl;
	}
    }

    exit(0);
}
