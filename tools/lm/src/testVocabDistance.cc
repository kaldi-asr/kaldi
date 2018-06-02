/*
 * test vocabulary distance class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2000 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testVocabDistance.cc,v 1.2 2000/08/06 02:39:35 stolcke Exp $";
#endif

#include "File.h"
#include "Vocab.h"
#include "VocabMultiMap.h"
#include "VocabDistance.h"

int
main(int argc, char **argv)
{
    Vocab vocab;
    Vocab pronVocab;
    VocabMultiMap dictionary(vocab, pronVocab);

    if (argc > 1) {
	File file(argv[1], "r");
	dictionary.read(file);
    }

    DictionaryDistance d(vocab, dictionary);
    DictionaryAbsDistance dabs(vocab, dictionary);

    while (1) {
	char word1[100],  word2[100];
    	if (scanf("%s %s", word1, word2) != 2) {
	    break;
	}

	VocabIndex w1 = vocab.addWord(word1);
	VocabIndex w2 = vocab.addWord(word2);

	cout << "rel distance = " << d.distance(w1, w2) << endl;
	cout << "rel penalty = " << d.penalty(w1) << endl;
	cout << "abs distance = " << dabs.distance(w1, w2) << endl;
	cout << "abs penalty = " << dabs.penalty(w1) << endl;
    }

    exit(0);
}

