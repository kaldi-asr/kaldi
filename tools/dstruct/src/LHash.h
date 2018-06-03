/*
 * LHash.h --
 *	Hash table with linear search.
 * 
 * LHash<KeyT,DataT> implements Map<KeyT,DataT> as a hash table with
 * open addressing, i.e., entries are stored at the computed hash index,
 * or in the next free entry, which is located with linear search.
 * Lookups, insertions, and deletions are all done in linear expected
 * time, with a constant factor that depends on the occupancy rate of
 * the table.  No space is wasted with pointers or bucket structures,
 * but the size of the table is at least the number of entries rounded to
 * the next power of 2.
 *
 * LHashIter<KeyT,DataT> performs iterations (in random key order) over
 * the entries in a hash table.
 *
 * Copyright (c) 1995-2012 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/LHash.h,v 1.41 2012/10/18 20:55:19 mcintyre Exp $
 *
 */

#ifndef _LHash_h_
#define _LHash_h_

#include "Map.h"

/* 
 * These constants are used below to pack the number of entries 
 * and the total allocated size into a single word.
 */
#define LHASH_MAXBIT_NBITS		5
#define LHASH_MAXENTRY_NBITS		27

const unsigned LHash_maxBitLimit = ((1U<<LHASH_MAXBIT_NBITS)-1);
const unsigned LHash_maxEntriesLimit = ((1U<<LHASH_MAXENTRY_NBITS)-1);

template <class KeyT, class DataT> class LHash;		// forward declaration
template <class KeyT, class DataT> class LHashIter;	// forward declaration

template <class KeyT, class DataT>
class LHashBody
{
    friend class LHash<KeyT,DataT>;
    friend class LHashIter<KeyT,DataT>;

    unsigned maxBits:LHASH_MAXBIT_NBITS;	/* number of bits in hash code
				     	 	 *  = log2 (maxEntries) */
    unsigned nEntries:LHASH_MAXENTRY_NBITS;	/* number of entries */

    MapEntry<KeyT,DataT> data[1];	/* hashed array of key-value pairs */
};

/*
 * If we define the class as a subclass of Map<KeyT,DataT> we
 * incur an extra word of memory per instance for the virtual function
 * table pointer.
 */
template <class KeyT, class DataT>
class LHash
#ifdef USE_VIRTUAL
	 : public Map<KeyT,DataT>
#endif
{
    friend class LHashIter<KeyT,DataT>;

public:
    LHash(unsigned size = 0);
    ~LHash();

    LHash(const LHash<KeyT,DataT> &source);
    LHash<KeyT,DataT> &operator= (const LHash<KeyT,DataT> &source);

    DataT *find(KeyT key, Boolean &foundP) const;
    inline DataT *find(KeyT key) const
        { Boolean found; return find(key, found); }
    KeyT getInternalKey(KeyT key, Boolean &foundP) const;
    inline KeyT getInternalKey(KeyT key) const
        { Boolean found; return getInternalKey(key, found); }
    DataT *insert(KeyT key, Boolean &foundP);
    inline DataT *insert(KeyT key) 
        { Boolean found; return insert(key, found); }
    Boolean remove(KeyT key, DataT *removedData = 0);
    void clear(unsigned size = 0);
    void setsize(unsigned size = 0);
    unsigned numEntries() const
	{ return body ? ((LHashBody<KeyT,DataT> *)body)->nEntries : 0; } ;
    void dump() const; 			/* debugging: dump contents to cerr */

    void memStats(MemStats &stats) const;

#ifdef DEBUG
    static unsigned long collisionCount;/* for performance diagnostics */
#endif

private:
    static KeyT zeroKey;
    void *body;				/* handle to the above -- this keeps
					 * the size of an empty table to
					 * one word */
    Boolean locate(KeyT key, unsigned &index) const;
					/* locate key in *data */
    void alloc(unsigned size);		/* initialization */
};

template <class KeyT, class DataT>
class LHashIter
{
    typedef int (*compFnType)(KeyT, KeyT);

public:
    LHashIter(const LHash<KeyT,DataT> &lhash, int (*sort)(KeyT, KeyT) = 0);
    LHashIter(const LHashIter<KeyT,DataT> &iter);
    ~LHashIter();

    void init();
    DataT *next(KeyT &key);
    bool operator()(const unsigned idx1, const unsigned idx2);
					/* callback function for sort() */

private:
    LHashBody<KeyT,DataT> *myLHashBody;	/* data being iterated over */
    unsigned current;			/* current index into data
					 * or sortedKeys */
    unsigned numEntries;		/* number of entries */
    int (*sortFunction)(KeyT, KeyT);	/* key sorting function,
					 * or 0 for random order */
    KeyT *sortedKeys;			/* array of sorted keys,
					   or 0 for random order */
    void sortKeys();			/* initialize sortedKeys */
};

/*
 * Key Comparison functions
 */
#define hashSize(nbits) (1U<<(nbits))	/* allocated size of data array */
#define hashMask(nbits) (~((~(size_t)0L)<<(nbits)))
					/* bitmask used to compute hash
					 * code modulo maxEntries */

template <class KeyT>
inline Boolean
LHash_equalKey(KeyT key1, KeyT key2)
{
    return (key1 == key2);
}

inline Boolean
LHash_equalKey(const char *key1, const char *key2)
{
    return (strcmp(key1, key2) == 0);
}

/*
 * Hashing functions
 * (We provide versions for integral types and char strings;
 * user has to add more specialized definitions.)
 */
inline size_t
LHash_hashKey(unsigned long key, unsigned maxBits)
{
    return (((key * 1103515245 + 12345) >> (30-maxBits)) & hashMask(maxBits));
}

inline size_t
LHash_hashKey(const char *key, unsigned maxBits)
{
    /*
     * The string hash function was borrowed from the Tcl libary, by 
     * John Ousterhout.  This is what he wrote:
     * 
     * I tried a zillion different hash functions and asked many other
     * people for advice.  Many people had their own favorite functions,
     * all different, but no-one had much idea why they were good ones.
     * I chose the one below (multiply by 9 and add new character)
     * because of the following reasons:
     *
     * 1. Multiplying by 10 is perfect for keys that are decimal strings,
     *    and multiplying by 9 is just about as good.
     * 2. Times-9 is (shift-left-3) plus (old).  This means that each
     *    character's bits hang around in the low-order bits of the
     *    hash value for ever, plus they spread fairly rapidly up to
     *    the high-order bits to fill out the hash value.  This seems
     *    works well both for decimal and non-decimal strings.
     */

    unsigned long i = 0;

    for (; *key; key++) {
	i += (i << 3) + *key;
    }
    return LHash_hashKey(i, maxBits);
}

#endif /* _LHash_h_ */

