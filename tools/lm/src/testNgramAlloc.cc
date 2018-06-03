
#include <iostream>
#include <assert.h>

#include "Vocab.h"
#include "Ngram.h"
#include "File.h"

int 
main(int argc, char *argv[])
{
    if (argc < 3) {
	cerr << "usage: testNgramAlloc Num LMfile\n";
	exit(2);
    }

    int num = atoi(argv[1]);
    const char *lmName = argv[2];

    for (int i = 0; i < num; i++) {
	Vocab vcb;
	File lmfile(lmName, "r", 1);

	Ngram *lm = new Ngram(vcb, 4);
	assert(lm);
	
	lm->read(lmfile, false);
	delete lm;
    }

    exit(0);
}
