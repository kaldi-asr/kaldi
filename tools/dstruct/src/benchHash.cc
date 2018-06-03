//
// Benchmarking for Hash and Trie datastructures
//
// $Header: /home/srilm/CVS/srilm/dstruct/src/benchHash.cc,v 1.7 2012-07-09 18:59:49 stolcke Exp $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif

//#define USE_SARRAY_TRIE

#include "option.h"
#include "LHash.cc"
#include "Trie.cc"

unsigned hashint = 0;
unsigned hashstruct = 0;
unsigned hashclass = 0;
unsigned trieint = 0;
unsigned hashsize = 1;

static Option options[] = {
    { OPT_INT, "hashint", &hashint, "allocate hash of integers" },
    { OPT_INT, "hashstruct", &hashstruct, "allocate hash of structures" },
    { OPT_INT, "hashclass", &hashclass, "allocate hash of objects" },
    { OPT_INT, "trieint", &trieint, "allocate trie of ints" },
    { OPT_INT, "hashsize", &hashsize, "num of entries per table" }
};

class myclass {
public:
	int w1;
	int w2;
};

typedef struct {
	int w1,w2;
    } mystruct;

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_LHASH(int,int);
INSTANTIATE_LHASH(int,mystruct);
INSTANTIATE_LHASH(int,myclass);
INSTANTIATE_TRIE(int,int);
#endif

int
main(int argc, char **argv)
{
    Opt_Parse(argc, argv, options, Opt_Number(options), 0);

    if (trieint) {
	cerr << "allocating " << trieint << "trie integer tables\n";

	Trie<int,int> *mytries;

	MemStats memuse;

	mytries = new Trie<int,int>[trieint];
	assert(mytries != 0);

	cerr << "BEFORE INSERTION\n";
	memuse.clear();
	unsigned i;
	for (i = 0; i < trieint; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();

	for (i = 0; i < trieint; i ++) {
		for (unsigned j = 0; j < hashsize; j ++) {
		    int keys[2];

		    keys[0] = i + j;
		    Map_noKey(keys[1]);
		    *mytries[i].insert(keys) = -(int)i;
		}
	}

	cerr << "AFTER INSERTION\n";
	memuse.clear();
	for (i = 0; i < trieint; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();
    }
	

    if (hashint) {
	cerr << "allocating " << hashint << " hash integer tables\n";

	LHash<int,int> *mytries;

	MemStats memuse;

	mytries = new LHash<int,int>[hashint];
	assert(mytries != 0);

	cerr << "BEFORE INSERTION\n";
	memuse.clear();
	unsigned i;
	for (i = 0; i < hashint; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();

	for (i = 0; i < hashint; i ++) {
		for (unsigned j = 0; j < hashsize; j ++) {
		    *mytries[i].insert(i+j) = -(int)i;
		}
	}

	cerr << "AFTER INSERTION\n";
	memuse.clear();
	for (i = 0; i < hashint; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();
    }
	
    if (hashstruct) {
	cerr << "allocating " << hashstruct << " hash struct tables\n";

	LHash<int,mystruct> *mytries;

	MemStats memuse;

	mytries = new LHash<int,mystruct>[hashstruct];
	assert(mytries != 0);

	cerr << "BEFORE INSERTION\n";
	memuse.clear();
	unsigned i;
	for (i = 0; i < hashstruct; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();

	for (i = 0; i < hashstruct; i ++) {
		for (unsigned j = 0; j < hashsize; j ++) {
		    mytries[i].insert(i+j)->w1 = -(int)i;
		}
	}

	cerr << "AFTER INSERTION\n";
	memuse.clear();
	for (i = 0; i < hashstruct; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();
    }
    if (hashclass) {
	cerr << "allocating " << hashclass << " hash object tables\n";

	LHash<int,myclass> *mytries;

	MemStats memuse;

	mytries = new LHash<int,myclass>[hashclass];
	assert(mytries != 0);

	cerr << "BEFORE INSERTION\n";
	memuse.clear();
	unsigned i;
	for (i = 0; i < hashclass; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();

	for (i = 0; i < hashclass; i ++) {
		for (unsigned j = 0; j < hashsize; j ++) {
		    mytries[i].insert(i+j)->w1 = -(int)i;
		}
	}

	cerr << "AFTER INSERTION\n";
	memuse.clear();
	for (i = 0; i < hashclass; i ++) {
		mytries[i].memStats(memuse);
	}
	memuse.print();
    }
    if (system("TERM=dumb top -d1 4") < 0) perror("system");

    exit(0);
}

