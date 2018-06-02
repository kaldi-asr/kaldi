//
// Test the hash function for VocabIndex strings
//

#ifndef lint
static char Copyright[] = "Copyright (c) 2004-2006 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testHash.cc,v 1.6 2012-04-22 04:18:16 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>

#include "File.h"
#include "Vocab.h"
#include "NgramStats.h"
#include "Array.cc"

static inline size_t
myLHash_hashKey(const VocabIndex *key, unsigned maxBits)
{
    unsigned long i = 0;

    for (; *key != Vocab_None; key ++) {
	i += (i << 12) + *key;
    }
    return LHash_hashKey(i, maxBits);
}

int
main(int argc, char *argv[])
{
    if (argc < 3) {
	cerr << "usage: testHash numBits ngram-file\n";
	exit(2);
    }
    unsigned numBits = atoi(argv[1]);

/* Test for integer hashing
    for (int i = -2; i <= 2; i ++) {
	    unsigned key = (unsigned)i;
	    unsigned long hash = LHash_hashKey(key, numBits);

	    printf("key = %d \t hash = 0x%08lx\n", key, hash);
    }
*/

    Vocab vocab;
    NgramStats ngrams(vocab, 10);

    ngrams.debugme(1);

    {
	File f(argv[2], "r");
	ngrams.read(f);
    }

    vocab.use();

    for (unsigned n = 1; n <= 3; n ++) {
	makeArray(VocabIndex, key, n+1);
	NgramsIter iter(ngrams, key, n);
      
	while (iter.next()) {
	    unsigned long hash = myLHash_hashKey(key, numBits);
	    cout << key << " " << hash << endl;
	}
    }

    exit(0);
}

