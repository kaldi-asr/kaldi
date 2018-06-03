
#ifdef PRE_ISO_CXX
# include <new.h>
# include <iostream.h>
#else
# include <new>
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifndef DEBUG
#define DEBUG
#endif

#include "LHash.cc"

typedef	unsigned int key_type;

int sortKeys(key_type k1, key_type k2)
{
	return k2 - k1;
}

int
main(int argc, char **argv)
{
    key_type key;
    key_type maxKey = 1000;

    if (argc > 1) {
	maxKey = atoi(argv[1]);
    }

    unsigned maxBits = 0;
    while (hashSize(maxBits) < maxKey) {
	assert(maxBits < LHash_maxBitLimit);
	maxBits++;
    }

    cerr << "maxBits = " << maxBits
         << " maxSize = " << hashSize(maxBits)
         << " maxKey = " << maxKey
	 << endl;

    LHash<key_type,int>::collisionCount = 0;

    LHash<key_type,int> ht(0);

    cerr << "=== Filling table 1\n";

    for (key = 0; key < maxKey; key ++) {
	*ht.insert(key) = (int)key;
    }

    cerr << "total collisions: " << LHash<key_type,int>::collisionCount << endl
	 << "avg collisions: "
	 << (double)LHash<key_type,int>::collisionCount / maxKey << endl;

    LHashIter<key_type,int> htIter(ht, sortKeys);
    LHash<key_type,int> ht2(0);

    LHash<key_type,int>::collisionCount = 0;

    cerr << "=== Copying table 1 to table 2\n";

    while (htIter.next(key)) {
	*ht2.insert(key) = (int)key;
    }

    cerr << "total collisions: " << LHash<key_type,int>::collisionCount << endl
	 << "avg collisions: "
	 << (double)LHash<key_type,int>::collisionCount / maxKey << endl;

    cerr << "=== Checking table 2\n";

    LHash<key_type,int>::collisionCount = 0;

    for (key = 0; key < maxKey; key ++) {
    	assert(ht2.find(key));
    }

    cerr << "total collisions: " << LHash<key_type,int>::collisionCount << endl
	 << "avg collisions: "
	 << (double)LHash<key_type,int>::collisionCount / maxKey << endl;

    cerr << "==== Deleting data from table 1\n";
    htIter.init();
    unsigned numdeleted = 0;
    while (htIter.next(key)) {
	assert(ht.remove(key));
	numdeleted ++;
    }
    cerr << "elements removed: " << numdeleted << endl;

    exit(0);
}

