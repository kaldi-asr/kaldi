/*
 * LHash.cc --
 *	Linear search hash table implementation
 *
 */

#ifndef _LHash_cc_
#define _LHash_cc_

#ifndef lint
static char LHash_Copyright[] = "Copyright (c) 1995-2012 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char LHash_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/LHash.cc,v 1.61 2013/10/03 03:35:03 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <new.h>
# include <iostream.h>
#else
# include <new>
# include <iostream>
using namespace std;
#endif
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <algorithm>

#include "LHash.h"
#include "BlockMalloc.h"

#undef INSTANTIATE_LHASH
#define INSTANTIATE_LHASH(KeyT, DataT) \
	template class LHash< KeyT, DataT >; \
	template class LHashIter< KeyT, DataT >

template <class KeyT, class DataT> KeyT LHash< KeyT, DataT >::zeroKey = KeyT();

#ifdef DEBUG
template <class KeyT, class DataT>
unsigned long LHash<KeyT,DataT>::collisionCount = 0;
#endif

const unsigned minHashBits = 3;		/* minimum no. bits for hashing
					 * tables smaller than this use linear
					 * search to save space */
const float fillRatio = 0.8f;		/* fill ration at which the table is
					 * expanded and rehashed */

#define BODY(b)	((LHashBody<KeyT,DataT> *)b)

#define BODY_SIZE(b, n) \
		(sizeof(*BODY(b)) + ((n) - 1) * sizeof(BODY(b)->data[0]))

/*
 * Dump the entire hash array to cerr.  Unused slots are printed as "FREE".
 */
template <class KeyT, class DataT>
void
LHash<KeyT,DataT>::dump() const
{
    if (body) {
	unsigned maxEntries = hashSize(BODY(body)->maxBits);
	unsigned i;

	for (i = 0; i < maxEntries; i++) {
	    cerr << " " << i << ": ";
	    if (Map_noKeyP(BODY(body)->data[i].key)) {
		cerr << "FREE";
	    } else {
		cerr << BODY(body)->data[i].key
#ifdef DUMP_VALUES
			/* 
			 * Only certain types can be printed.
			 */
		     << "->" << BODY(body)->data[i].value
#endif /* DUMP_VALUES */
		     ;
	    }
	}
    } else {
	cerr << "EMPTY";
    }
    cerr << endl;
}

template <class KeyT, class DataT>
void
LHash<KeyT,DataT>::memStats(MemStats &stats) const
{
    stats.total += sizeof(*this);
    if (body) {
        unsigned maxEntries = hashSize(BODY(body)->maxBits);
	size_t mySize = sizeof(*BODY(body)) +
			    sizeof(BODY(body)->data[0]) *
				(maxEntries - 1);
	stats.total += mySize;
	stats.wasted += sizeof(BODY(body)->data[0]) *
				(maxEntries - BODY(body)->nEntries);

	stats.allocStats[mySize > MAX_ALLOC_STATS ?
			    MAX_ALLOC_STATS : mySize] += 1;
    }
}

/*
 * Compute the actual minimum size required for a given number of entries
 */
static inline unsigned
roundSize(unsigned size)
{
    if (size < hashSize(minHashBits)) {
	return size;
    } else {
	return (unsigned)((size + 1)/ fillRatio);
    }
}

template <class KeyT, class DataT>
void
LHash<KeyT,DataT>::alloc(unsigned size)
{
    unsigned maxBits, maxEntries;
    unsigned i;

    /*
     * round up to power of two
     */
    maxBits = 0;
    while (hashSize(maxBits) < size) {
	assert(maxBits < LHash_maxBitLimit);
	maxBits++;
    }

    maxEntries = hashSize(maxBits);

    //cerr << "LHash::alloc: current " << (body ? BODY(body)->nEntries : 0)
    //	 << ", requested " << size 
    //	 << ", allocating " << maxEntries << " (" << maxBits << ")\n";

    body = (LHashBody<KeyT,DataT> *)BM_malloc(BODY_SIZE(body, maxEntries));
    assert(body != 0);

    BODY(body)->maxBits = maxBits;
    BODY(body)->nEntries = 0;

    for (i = 0; i < maxEntries; i++) {
        new (&(BODY(body)->data[i].key)) KeyT;	  // initialize the key object
	Map_noKey(BODY(body)->data[i].key);
        new (&(BODY(body)->data[i].value)) DataT; // initialize the key object
    }
    //cerr << "memory for header = " <<
    //		((void *)&(BODY(body)->data[0]) - (void *)BODY(body)) << endl;
    //cerr << "memory per entry = " <<
    //		((void *)&(BODY(body)->data[3]) - (void *)&(BODY(body)->data[2])) << endl;
}

template <class KeyT, class DataT>
LHash<KeyT,DataT>::LHash(unsigned size)
    : body(0)
{
    if (size != 0) {
	/*
	 * determine actual initial size
	 */
	alloc(roundSize(size));
    }
}

template <class KeyT, class DataT>
void
LHash<KeyT,DataT>::clear(unsigned size)
{
    if (body) {
	unsigned maxEntries = hashSize(BODY(body)->maxBits);
	unsigned i;

	for (i = 0; i < maxEntries; i++) {
	    if (! Map_noKeyP(BODY(body)->data[i].key)) {
		Map_freeKey(BODY(body)->data[i].key);
	    }
	}
	BM_free(body, BODY_SIZE(body, maxEntries));
	body = 0;
    }
    if (size != 0) {
	alloc(roundSize(size));
    }
}

template <class KeyT, class DataT>
void
LHash<KeyT,DataT>::setsize(unsigned size)
{
    if (body == 0 && size != 0) {
	alloc(roundSize(size));
    }
}

template <class KeyT, class DataT>
LHash<KeyT,DataT>::~LHash()
{
    clear(0);
}

template <class KeyT, class DataT>
LHash<KeyT,DataT>::LHash(const LHash<KeyT,DataT> &source)
    : body(0)
{
#ifdef DEBUG
    cerr << "warning: LHash copy constructor called\n";
#endif
    *this = source;
}

template <class KeyT, class DataT>
LHash<KeyT,DataT> &
LHash<KeyT,DataT>::operator= (const LHash<KeyT,DataT> &other)
{
#ifdef DEBUG
    cerr << "warning: LHash::operator= called\n";
#endif

    if (&other == this) {
	return *this;
    }

    /*
     * copy hash entries from old to new 
     */
    if (other.body) {
	unsigned maxEntries = hashSize(BODY(other.body)->maxBits);
	/*
	 * ensure we have exactly the same size as source table
	 */
	clear(0);
	alloc(maxEntries);

	for (unsigned i = 0; i < maxEntries; i++) {
	    KeyT thisKey = BODY(other.body)->data[i].key;

	    if (!Map_noKeyP(thisKey)) {
		/*
		 * Copy key
		 */
		BODY(body)->data[i].key = Map_copyKey(thisKey);

		/*
		 * Initialize data, required for = operator
		 */
		new (&(BODY(body)->data[i].value)) DataT;

		/*
		 * Copy data
		 */
		BODY(body)->data[i].value = BODY(other.body)->data[i].value;
	    }
	}
	BODY(body)->nEntries = BODY(other.body)->nEntries;
    } else {
	clear(0);
    }

    return *this;
}

/*
 * Returns index into data array that has the key which is either
 * equal to key, or indicates the place where key would be placed if it
 * occurred in the array.
 */
template <class KeyT, class DataT>
Boolean
LHash<KeyT,DataT>::locate(KeyT key, unsigned &index) const
{
    assert(!Map_noKeyP(key));

    if (body) {
	unsigned maxBits = BODY(body)->maxBits;
    	register MapEntry<KeyT,DataT> *data = BODY(body)->data;

	if (maxBits < minHashBits) {
	    /*
	     * Do a linear search
	     */
	    unsigned nEntries = BODY(body)->nEntries;
	    register unsigned i;

	    for (i = 0; i < nEntries; i++) {
		if (LHash_equalKey(data[i].key, key)) {
		    index = i;
		    return true;
		}
	    }
	    index = i;
	    return false;
	} else {
	    /*
	     * Do a hashed search
	     */
	    size_t hash = LHash_hashKey(key, maxBits);
	    size_t i;

	    for (i = hash; ; i = (i + 1) & hashMask(maxBits))
	    {
		if (Map_noKeyP(data[i].key)) {
		    index = i;
		    return false;
		} else if (LHash_equalKey(data[i].key, key)) {
		    index = i;
		    return true;
		}
#ifdef DEBUG
		collisionCount += 1;
#endif
	    }
	}
    } else {
	return false;
    }
}


template <class KeyT, class DataT>
DataT *
LHash<KeyT,DataT>::find(KeyT key, Boolean &foundP) const
{
    unsigned index;

    if ((foundP = locate(key, index))) {
	return &(BODY(body)->data[index].value);
    } else {
	return 0;
    }
}

template <class KeyT, class DataT>
KeyT
LHash<KeyT,DataT>::getInternalKey(KeyT key, Boolean &foundP) const
{
    unsigned index;

    if ((foundP = locate(key, index))) {
	return BODY(body)->data[index].key;
    } else {
	return zeroKey;
    }
}

template <class KeyT, class DataT>
DataT *
LHash<KeyT,DataT>::insert(KeyT key, Boolean &foundP)
{
    unsigned index;

    assert(!(Map_noKeyP(key)));

    /*
     * Make sure there is room for at least one entry
     */
    if (body == 0) {
	alloc(1);
    }

    if ((foundP = locate(key, index))) {
	return &(BODY(body)->data[index].value);
    } else {
	unsigned maxEntries = hashSize(BODY(body)->maxBits);
	unsigned nEntries = BODY(body)->nEntries;

	/*
	 * Rehash table if necessary
	 */
	unsigned minSize = roundSize(nEntries + 1);

	if (minSize > maxEntries) {
	    LHashBody<KeyT,DataT> *oldBody = BODY(body);
	    unsigned i;

	    /*
	     * Since LHash_maxEntriesLimit is a power of two minus 1
	     * we need to check this only when the array is enlarged
	     */
	    assert(nEntries < LHash_maxEntriesLimit);

	    alloc(minSize);
	    BODY(body)->nEntries = nEntries;

	    if (BODY(body)->maxBits < minHashBits) {
		/*
		 * Just copy old entries to new storage, no reindexing
		 * required.
		 */
		memcpy(BODY(body)->data, oldBody->data,
			sizeof(oldBody->data[0]) * nEntries);
	    } else {
		/*
		 * Rehash
		 */
		for (i = 0; i < maxEntries; i++) {
		    KeyT key = oldBody->data[i].key;

		    if (! Map_noKeyP(key)) {
			(void)locate(key, index);
			memcpy(&(BODY(body)->data[index]), &(oldBody->data[i]),
							sizeof(oldBody->data[0]));
		    }
		}
	    }
	    BM_free(oldBody, BODY_SIZE(oldBody, maxEntries));

	    /*
	     * Entries have been moved, so have to re-locate key
	     */
	    (void)locate(key, index);
	}

	BODY(body)->data[index].key = Map_copyKey(key);

	/*
	 * Initialize data to zero, but also call constructors, if any
	 */
	memset(&(BODY(body)->data[index].value), 0,
			sizeof(BODY(body)->data[index].value));
	new (&(BODY(body)->data[index].value)) DataT;

	BODY(body)->nEntries++;

	return &(BODY(body)->data[index].value);
    }
}

template <class KeyT, class DataT>
Boolean 
LHash<KeyT,DataT>::remove(KeyT key, DataT *removedData)
{
    unsigned index;

    if ((locate(key, index))) {
	Map_freeKey(BODY(body)->data[index].key);
	Map_noKey(BODY(body)->data[index].key);

//cerr << "REMOVE key = " << key << " index = " << index << endl;
        if (removedData != 0)
	    memcpy(removedData, &BODY(body)->data[index].value, sizeof(DataT));

	if (BODY(body)->maxBits < minHashBits) {
	    /*
	     * Linear search mode -- Just move all entries above the
	     * the one removed to fill the gap.
	     */
	    unsigned nEntries = BODY(body)->nEntries;

	    memmove(&BODY(body)->data[index],
		    &BODY(body)->data[index+1],
		    (nEntries - index - 1) * sizeof(BODY(body)->data[0]));
	    Map_noKey(BODY(body)->data[nEntries - 1].key);
	} else {
	    /*
	     * The entry after the one being deleted could actually
	     * belong to a prior spot in the table, but was bounced forward due
	     * to a collision.   The invariant used in lookup is that
	     * all locations between the hash index and the actual index are
	     * filled.  Hence we examine all entries between the deleted
	     * position and the next free position as whether they need to
	     * be moved backward.
	     */
	    while (1) {
		unsigned newIndex;

		index = (index + 1) & hashMask(BODY(body)->maxBits);

		if (Map_noKeyP(BODY(body)->data[index].key)) {
		    break;
		}

		/* 
		 * If find returns false that means the deletion has
		 * introduced a hole in the table that would prevent
		 * us from finding the next entry. Luckily, find 
		 * also tells us where the hole is.  We move the 
		 * entry to its rightful place and continue filling
		 * the hole created by this move.
		 */
		if (!locate(BODY(body)->data[index].key, newIndex)) {
//cerr << "MOVING key = " << (BODY(body)->data[index].key) << " from " << index << " to " << newIndex << endl;
		    memcpy(&(BODY(body)->data[newIndex]),
			   &(BODY(body)->data[index]),
			   sizeof(BODY(body)->data[0]));
		    Map_noKey(BODY(body)->data[index].key);
		}
	    }
	}
	BODY(body)->nEntries--;
	return true;
    } else {
	return false;
    }
}

template <class KeyT, class DataT>
void
LHashIter<KeyT,DataT>::sortKeys()
{
    /*
     * Store keys away and sort them to the user's orders.
     */
    unsigned maxEntries = hashSize(myLHashBody->maxBits);

    unsigned *sortedIndex = new unsigned[numEntries];
    assert(sortedIndex != 0);

    unsigned i;
    unsigned j = 0;
    for (i = 0; i < maxEntries; i++) {
	if (!Map_noKeyP(myLHashBody->data[i].key)) {
            sortedIndex[j++] = i;
	}
    }
    assert(j == numEntries);

    if (sortFunction != 0) {
        sort(sortedIndex, sortedIndex + numEntries, *this);
    }

    /*
     * Save the keys for enumeration.  The reason we save the keys,
     * not the indices, is that indices may change during enumeration
     * due to deletions.
     */
    sortedKeys = new KeyT[numEntries];
    assert(sortedKeys != 0);

    for (i = 0; i < numEntries; i++) {
        sortedKeys[i] = myLHashBody->data[sortedIndex[i]].key;
    }

    delete [] sortedIndex;
}

template <class KeyT, class DataT>
LHashIter<KeyT,DataT>::LHashIter(const LHash<KeyT,DataT> &lhash,
				    int (*keyCompare)(KeyT, KeyT))
    : myLHashBody(BODY(lhash.body)), current(0),
      numEntries(lhash.numEntries()), sortFunction(keyCompare), sortedKeys(0)
{
    /*
     * Note: we access the underlying LHash through the body pointer,
     * not the top-level LHash itself.  This allows the top-level object
     * to be moved without affecting an ongoing iteration.
     * XXX: This only works because
     * - it is illegal to insert while iterating
     * - deletions don't cause reallocation of the data
     */
    if (myLHashBody) {
	sortKeys();
    }
}


/*
 * Copy an existing iterator including its current position.
 * Thus, iteration in the new iterator will start at the position
 * following the current one in the old iterator.
 */
template <class KeyT, class DataT>
LHashIter<KeyT,DataT>::LHashIter(const LHashIter<KeyT,DataT> &iter)
    : myLHashBody(iter.myLHashBody), current(iter.current),
      numEntries(iter.numEntries), sortFunction(iter.sortFunction),
      sortedKeys(0)
{
    if (iter.sortedKeys) {
	sortedKeys = new KeyT[numEntries];
	assert(sortedKeys != 0);

	for (unsigned i = 0; i < numEntries; i++) {
	    sortedKeys[i] = iter.sortedKeys[i];
	}
    }
}


/*
 * This is the callback function passed to sort for comparing array
 * entries. It is passed the indices into the data array, which are
 * compared according to the user-supplied function applied to the 
 * keys found at those locations.
 */
template <class KeyT, class DataT>
bool
LHashIter<KeyT,DataT>::operator()(const unsigned idx1, const unsigned idx2)
{
    return (*(compFnType)sortFunction)
                  (BODY(myLHashBody)->data[idx1].key,
                   BODY(myLHashBody)->data[idx2].key) < 0;
}

template <class KeyT, class DataT>
LHashIter<KeyT,DataT>::~LHashIter()
{
    delete [] sortedKeys;
    sortedKeys = 0;
}


template <class KeyT, class DataT>
void 
LHashIter<KeyT,DataT>::init()
{
    delete [] sortedKeys;
    sortedKeys = 0;

    current = 0;

    {
	/*
	 * XXX: fake LHash object to access numEntries()
	 */
	LHash<KeyT,DataT> myLHash(0);

	myLHash.body = myLHashBody;
	numEntries = myLHash.numEntries();
	myLHash.body = 0;
    }

    if (myLHashBody) {
	sortKeys();
    }
}

template <class KeyT, class DataT>
DataT *
LHashIter<KeyT,DataT>::next(KeyT &key)
{

    if (myLHashBody == 0) {
	return 0;
    } else {
	unsigned index;

	assert(sortedKeys != 0);

	if (current == numEntries) {
	    return 0;
	}

	/*
	 * XXX: fake LHash object to access locate()
	 */
	LHash<KeyT,DataT> myLHash(0);

	myLHash.body = myLHashBody;
	myLHash.locate(sortedKeys[current++], index);
	myLHash.body = 0;

	key = myLHashBody->data[index].key;
	return &(myLHashBody->data[index].value);
    }
}

#undef BODY

#endif /* _LHash_cc_ */
