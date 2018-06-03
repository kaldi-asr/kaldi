/*
 * Trie.cc --
 *	The trie class implementation.
 *
 */

#ifndef _Trie_cc_
#define _Trie_cc_

#ifndef lint
static char Trie_Copyright[] = "Copyright (c) 1995-2012 SRI International, 2012-2015 Microsoft Corp.  All Rights Reserved.";
static char Trie_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Trie.cc,v 1.28 2016-03-05 01:23:28 stolcke Exp $";
#endif

#ifdef PRE_ISO_CXX
# include <new.h>
# include <iostream.h>
#else
# include <new>
# include <iostream>
using namespace std;
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "Trie.h"

#undef INSTANTIATE_TRIE

#ifdef USE_SARRAY_TRIE

# include "SArray.cc"

# define INSTANTIATE_TRIE(KeyT,DataT) \
	typedef Trie< KeyT,DataT > TrieType; \
	INSTANTIATE_SARRAY(KeyT, TrieType ); \
	template class Trie< KeyT,DataT >; \
	template class TrieIter< KeyT,DataT >; \
	template class TrieIter2< KeyT,DataT >

#else /* ! USE_SARRAY_TRIE */

# include "LHash.cc"

# define INSTANTIATE_TRIE(KeyT,DataT) \
	typedef Trie< KeyT,DataT > TrieType; \
	INSTANTIATE_LHASH(KeyT, TrieType ); \
	template class Trie< KeyT,DataT >; \
	template class TrieIter< KeyT,DataT >; \
	template class TrieIter2< KeyT,DataT >

#endif /* USE_SARRAY_TRIE */
   
template <class KeyT, class DataT>
Trie<KeyT,DataT>::Trie(unsigned size)
    : sub(size)
{
    /*
     * Data starts out zero-initialized for convenience
     */
    memset(&data, 0, sizeof(data));
}

template <class KeyT, class DataT>
Trie<KeyT,DataT>::~Trie()
{
    TrieIter<KeyT,DataT> iter(*this);
    KeyT key;
    Trie<KeyT,DataT> *node;

    /*
     * destroy all subtries recursively
     */
    while ((node = iter.next(key))) {
	node->~Trie();
    }
}

/*
 * Dump contents of Trie to cerr, using indentation to reflect node depth
 */
template <class KeyT, class DataT>
void
Trie<KeyT,DataT>::dump(unsigned indent) const
{
    TrieIter<KeyT,DataT> iter(*this);
    KeyT key;
    Trie<KeyT,DataT> *node;

#ifdef DUMP_VALUES
    cerr << "Value = " << data << endl;
#endif
    while ((node = iter.next(key))) {
	for (unsigned i = 0; i < indent; i ++) {
	    cerr << " ";
	}
	cerr << "Key = " << key << endl;
	node->dump(indent + 5);
    }
}

/*
 * Compute memory stats for Trie, including all children and grand-children
 */
template <class KeyT, class DataT>
void
Trie<KeyT,DataT>::memStats(MemStats &stats) const
{
    stats.total += sizeof(*this) - sizeof(sub);
    sub.memStats(stats);

    TrieIter<KeyT,DataT> iter(*this);
    KeyT key;
    Trie<KeyT,DataT> *node;

    while ((node = iter.next(key))) {
	stats.total -= sizeof(*node);
	node->memStats(stats);
    }
}

/* 
 * The following methods implement find(), insert(), remove()
 * by straightforward recursive decent.  Recoding this to use iteration
 * should give marginal speedups.
 */

template <class KeyT, class DataT>
Trie<KeyT,DataT> *
Trie<KeyT,DataT>::findTrie(const KeyT *keys, Boolean &foundP) const
{
    if (keys == 0 || Map_noKeyP(keys[0])) {
	foundP = true;
	return (Trie<KeyT,DataT> *)this;	// discard const
    } else {
	Trie<KeyT,DataT> *subtrie = sub.find(keys[0]);

	if (subtrie == 0) {
	    foundP = false;
	    return 0;
	} else {
	    return subtrie->findTrie(keys + 1, foundP);
	}
    }
}

template <class KeyT, class DataT>
Trie<KeyT,DataT> *
Trie<KeyT,DataT>::findPrefixTrie(const KeyT *keys, unsigned &depth) const
{
    if (keys == 0 || Map_noKeyP(keys[0])) {
	depth = 0;
	return (Trie<KeyT,DataT> *)this;	// discard const
    } else {
	Trie<KeyT,DataT> *subtrie = sub.find(keys[0]);

	if (subtrie == 0) {
	    depth = 0;
	    return (Trie<KeyT,DataT> *)this;	// discard const
	} else {
	    unsigned subDepth;
	    Trie<KeyT,DataT> *node = 
	    	subtrie->findPrefixTrie(keys + 1, subDepth);
	    depth = subDepth + 1;
	    return node;
	}
    }
}

template <class KeyT, class DataT>
Trie<KeyT,DataT> *
Trie<KeyT,DataT>::insertTrie(const KeyT *keys, Boolean &foundP)
{
    if (keys == 0 || Map_noKeyP(keys[0])) {
	foundP = true;
	return this;
    } else {
	Trie<KeyT,DataT> *subtrie = sub.insert(keys[0], foundP);

	/*
	 * Set foundP = false if a new entry was created at any level
	 */
	if (foundP) {
	    return subtrie->insertTrie(keys + 1, foundP);
	} else {
	    Boolean subFoundP;	// ignored
	    return subtrie->insertTrie(keys + 1, subFoundP);
	}
    }
}

template <class KeyT, class DataT>
Boolean
Trie<KeyT,DataT>::removeTrie(const KeyT *keys, Trie<KeyT,DataT> *removedData)
{
    if (keys == 0 || Map_noKeyP(keys[0])) {
	return false;
    } else if (Map_noKeyP(keys[1])) {
	if (removedData == 0) {
	    Trie<KeyT,DataT> node;
	    if (sub.remove(keys[0], &node)) {
#if !defined(__GNUC__) || !(__GNUC__ >= 4 && __GNUC_MINOR__ >= 9 || __GNUC__ > 4) 
		/*
		 * XXX: Call subtrie destructor explicitly since we're not
		 * passing the removed node to the caller.
		 * !!! Triggers bug with gcc >= 4.9 optimization !!!
		 */
	        node.~Trie();
#endif
		return true;
	    } else {
		return false;
	    }
	} else {
	    return sub.remove(keys[0], removedData);
	}
    } else {
        Boolean foundP;
	Trie<KeyT,DataT> *subtrie = sub.find(keys[0], foundP);

        if (!foundP) {
	    return false;
	} else {
	    return subtrie->removeTrie(keys + 1, removedData);
	}
    }
}

template <class KeyT, class DataT>
unsigned int
Trie<KeyT,DataT>:: numEntries(const KeyT *keys) const
{
    if (keys == 0 || Map_noKeyP(keys[0])) {
	return sub.numEntries();
    } else {
	Boolean foundP;
	Trie<KeyT,DataT> *subtrie = sub.find(keys[0], foundP);

        if (!foundP) {
	    return 0;
	} else {
	    return subtrie->numEntries(keys + 1);
	}
    }
}

/*
 * Iteration over all nodes at a given level in the trie
 */

template <class KeyT, class DataT>
TrieIter2<KeyT,DataT>::TrieIter2(const Trie<KeyT,DataT> &trie,
			         KeyT *keys, unsigned int level,
				 int (*sort)(KeyT,KeyT))
    : myTrie(trie), keys(keys), level(level), sort(sort),
      myIter(trie.sub, sort), subIter(0), done(false)
{
    /*
     * Set the next keys[] entry now to terminate the sequence.
     * keys[0] gets set later by next()
     */
    if (level == 0) {
	Map_noKey(keys[0]);
    } else if (level == 1) {
	Map_noKey(keys[1]);
    }
}

template <class KeyT, class DataT>
TrieIter2<KeyT,DataT>::~TrieIter2()
{
    delete subIter;
}

template <class KeyT, class DataT>
void
TrieIter2<KeyT,DataT>::init()
{
    delete subIter;
    subIter = 0;
    myIter.init();
    done = false;
}

template <class KeyT, class DataT>
Trie<KeyT,DataT> *
TrieIter2<KeyT,DataT>::next()
{
    if (level == 0) {
	/*
	 * Level enumerators exactly one node -- the trie root.
	 * NOTE: This recursion could be simplified to deal only
	 * with level == 0 as a termination case, but then
	 * a level 1 iteration would be rather expensive (creating
	 * a new iterator for each child).
	 */
	if (done) {
	    return 0;
	} else {
	    done = true;
	    return (Trie<KeyT,DataT> *)&myTrie;		// discard const
	}
    } else if (level == 1) {
	/*
	 * Just enumerate all children.
	 */
	return myIter.next(keys[0]);
    } else {
	/*
	 * Iterate over children until one of its children
	 * is found, via a recursive iteration.
	 */
	while (1) {
	    if (subIter == 0) {
		/*
		 * Previous children's children iteration exhausted.
		 * Advance to next child of our own.
		 */
		Trie<KeyT,DataT> *subTrie = myIter.next(keys[0]);
		if (subTrie == 0) {
		    /*
		     * No more children -- done.
		     */
		    return 0;
		} else {
		    /*
		     * Create the recursive iterator when 
		     * starting with a child ....
		     */
		    subIter = new TrieIter2<KeyT,DataT>(*subTrie, keys + 1,
							level - 1, sort);
		    assert(subIter != 0);
		}
	    }
	    Trie<KeyT,DataT> *next = subIter->next();
	    if (next == 0) {
		/*
		 * ... and destroy it when we're done with it.
		 */
		delete subIter;
		subIter = 0;
	    } else {
		return next;
	    }
	}
    }
}

#endif /* _Trie_cc_ */
