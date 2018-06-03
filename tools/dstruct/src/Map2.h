/*
 * Map2.h --
 *	Two-dimensional maps
 *
 * Map2<Key1T, Key2T, DataT> is a template container class that implements a
 * mapping from pairs of _keys_ (types Key1T, Key2T) to data items or _values_
 * (type DataT).  It is built as an extension of Map and has a similar
 * interface.
 *
 * DataT *find(Key1T key1, Key2T key2, Boolean &foundP)
 *	Returns a pointer to the data item found under (key1, key2), or null if
 *	the keys are not in the Map.
 *	With this and the other functions, the foundP argument is optional
 *	and returns whether the key was found.
 *
 * DataT *insert(Key1T key1, Key2T key2, Boolean &foundP)
 *	Returns a pointer to the data item for (key1, key2), creating a new
 *	entry if necessary (indicated by foundP == false).
 *	New data items are zero-initialized.
 *
 * Boolean remove(Key1T key1, Key2T key2, DataT *removedData = 0)
 *	Deletes the entry associated with (key1, key2) from the Map, returning
 *	true iff the entry was found.  If removedData != 0 the removed value is
 *	returned at *removedData.
 *
 * Boolean remove(Key1T key1)
 *	Deletes all entries with the first key key1, returning true iff any
 *	entries were deleted.
 *
 * void clear()
 *	Delete all entries.
 *
 * unsigned numEntries(Key1T key1)
 *	Returns the current number of keys (i.e., entries) stored under key1.
 *
 * MAP2_ITER_T *find1(Key1T key1, Boolean &foundP)
 *	Returns the row vector indexed by key1, or NULL if none exists.
 *
 * The DataT * pointers returned by find(), insert() and remove() are
 * valid only until the next operation on the Map2 object.  It is left
 * to the user to assign actual values by dereferencing the pointers 
 * returned.  The main benefit is that only one key lookup is needed
 * for a find-and-change operation.
 *
 * Map2Iter<Key1T, Key2T, DataT> provids iterators over the first-level
 *	entries in Map2.
 *
 * Map2Iter2<Key1T, Key2T, DataT> provids iterators over the second-level
 *	entries in Map2.
 *
 * Map2Iter2(Map2<Key1T,Key2T,DataT> &map2, Key1T key1)
 *	Creates and initializes an iteration over the entries under key1 in
 *	map2.
 *
 * void init()
 *	Reset an iteration to the first element.
 *
 * DataT *next(Key2T &key)
 *	Steps the iteration and returns a pointer to the next entry,
 *	or null if the iteration is finished.  key is set to the associated
 *	Key value.
 *
 * Copyright (c) 1999,2002 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Map2.h,v 1.9 2014-05-25 20:01:43 stolcke Exp $
 *
 */

#ifndef _Map2_h_
#define _Map2_h_

#undef MAP2_INDEX_T
#undef MAP2_ITER_T

#ifdef USE_SARRAY_MAP2

# define MAP2_INDEX_T	SArray
# define MAP2_ITER_T	SArrayIter
# include "SArray.h"

#else /* ! USE_SARRAY_MAP2 */

# define MAP2_INDEX_T	LHash
# define MAP2_ITER_T	LHashIter
# include "LHash.h"

#endif /* USE_SARRAY_MAP2 */

template <class Key1T, class Key2T, class DataT> class Map2Iter;
template <class Key1T, class Key2T, class DataT> class Map2Iter2;
						// forward declaration

template <class Key1T, class Key2T, class DataT>
class Map2
{
    friend class Map2Iter<Key1T,Key2T,DataT>;
    friend class Map2Iter2<Key1T,Key2T,DataT>;
public:
    Map2();
    ~Map2();

    DataT *find(Key1T key1, Key2T key2, Boolean &foundP) const;
    DataT *find(Key1T key1, Key2T key2) const
	{ Boolean found; return find(key1, key2, found); };
    DataT *insert(Key1T key1, Key2T key2, Boolean &foundP);
    DataT *insert(Key1T key1, Key2T key2)
	{ Boolean found; return insert(key1, key2, found); };
    Boolean remove(Key1T key1, Key2T key2, DataT *removedData = 0);
    Boolean remove(Key1T key1);

    MAP2_INDEX_T<Key2T,DataT> *find1(Key1T key1, Boolean &foundP) const
      { return sub.find(key1, foundP); };
    MAP2_INDEX_T<Key2T,DataT> *find1(Key1T key) const
      { Boolean found; return find1(key, found); };
    unsigned numEntries(Key1T key1) const
      { MAP2_INDEX_T<Key2T,DataT> *value = sub.find(key1);
	return value ? value->numEntries() : 0; };

    void clear();

    void dump() const;				/* debugging: dump contents */
    void memStats(MemStats &stats) const;	/* compute memory stats */

private:
    MAP2_INDEX_T< Key1T, MAP2_INDEX_T<Key2T,DataT> > sub;
						/* index of indices */
};

/*
 * Iteration over first-level entries 
 */
template <class Key1T, class Key2T, class DataT>
class Map2Iter
{
public:
    Map2Iter(Map2<Key1T,Key2T,DataT> &map2, int (*sort)(Key1T,Key1T) = 0)
	// XXX: use insert() in initializer to ensure entry exists
	: myIter(map2.sub, sort) {};	

    void init() { myIter.init(); } ;
    MAP2_INDEX_T<Key2T,DataT> *next(Key1T &key) { return myIter.next(key); };

private:
    MAP2_ITER_T<Key1T, MAP2_INDEX_T<Key2T,DataT> > myIter;
};

/*
 * Iteration over second-level entries 
 */
template <class Key1T, class Key2T, class DataT>
class Map2Iter2
{
public:
    Map2Iter2(Map2<Key1T,Key2T,DataT> &map2, Key1T key1,
						int (*sort)(Key2T,Key2T) = 0)
	// XXX: use insert() in initializer to ensure entry exists
	: myIter(*map2.sub.insert(key1), sort) {};	

    void init() { myIter.init(); } ;
    DataT *next(Key2T &key) { return myIter.next(key); };

private:
    MAP2_ITER_T<Key2T,DataT> myIter;
};

#endif /* _Map2_h_ */
