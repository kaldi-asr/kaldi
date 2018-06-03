/*
 * Map.h --
 *	Keyed map.
 *
 * Map<KeyT,DataT> is template a container class that implement a mapping
 * from a * set of _keys_ (type KeyT) to data items or _values_ (type DataT).
 * This is an abstract class -- it is implemented only in terms of 
 * more specific subclasses, such as lists, arrays, hash tables, etc.
 * All of these present the followig interface.
 *
 * DataT *find(KeyT key, Boolean &foundP)
 *	Returns a pointer to the data item found under key, or null of
 *	the key is not in the Map.
 *	With this and the other functions, the foundP argument is optional
 *	and returns whether the key was found.
 *
 * KeyT getInternalKey(KeyT key, Boolean &foundP)
 *	Returns the internal memory allocated for key, if it exists
 *	(or else zero data).  This can be used to share key memory
 *	between several data structures, but the keys (in case they are
 *	referenced by pointers) must not be modified.  Also note that
 *	keys returned invalid as soon as the corresponding entry in
 *	the Map (or the entire Map) is deleted.
 *
 * DataT *insert(KeyT key, Boolean &foundP)
 *	Returns a pointer to the data item for key, creating a new
 *	entry if necessary (indicated by foundP == false).
 *	New data items are zero-initialized.
 *
 * DataT *remove(KeyT key, Boolean &foundP)
 *	Deletes the entry associated with key from the Map, returning
 *	a pointer to the previously stored value, if any.
 *
 * void clear(unsigned int size)
 *	Removes all entries from table.  The optional size argument
 *	resets the allocated number of entries in the Map.
 *
 * unsigned int numEntries()
 *	Returns the current number of keys (i.e., entries) in the Map.
 *
 * The DataT * pointers returned by find(), insert() and remove() are
 * valid only until the next operation on the Map object.  It is left
 * to the user to assign actual values by dereferencing the pointers 
 * returned.  The main benefit is that only one key lookup is needed
 * for a find-and-change operation.
 *
 * Copyright (c) 1995-2006 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Map.h,v 1.25 2014-05-27 03:04:56 stolcke Exp $
 *
 */

#ifndef _Map_h_
#define _Map_h_

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if !defined(_MSC_VER) && !defined(WIN32)
#include <sys/param.h>
#endif

#include "Boolean.h"
#include "MemStats.h"

#ifndef NBBY
#define NBBY	8
#endif

/*
 * _Map is a non-template parent class to all classes Map<KeyT,DataT>.
 * It serves only to collect certain global variables shared by all
 * Map objects.
 */
class _Map
{
public:
    static unsigned int initialSize;	/* default initial size */
    static float growSize;		/* factor used in enlarging
					 * a Map on demand */
};

template <class KeyT, class DataT>
class MapEntry
{
public:
    DataT value;
    KeyT key;
};

template <class KeyT, class DataT>
class Map : public _Map
{
public:
    virtual DataT *find(KeyT key, Boolean &foundP) const = 0;
					/* key lookup -- returns a
					 * zero object and sets foundP =
					 * false if key is not found */
    virtual KeyT getInternalKey(KeyT key, Boolean &foundP) const = 0;
					/* return the internalized key
					 * for an entry -- use with caution */
    virtual DataT *insert(KeyT key, Boolean &foundP) = 0;
					/* create or change a value */
    virtual DataT *remove(KeyT key, Boolean &foundP) = 0;
					/* delete an entry ("delete" is
					 * a reserved word) */
    virtual void clear(unsigned int size) = 0;
					/* remove all entries */
    virtual unsigned int numEntries() const = 0;
					/* number of entries in Map */

    virtual void memStats(MemStats &stats) const = 0;
					/* compute memory stats */
};

/*
 * Functions to manage key memory
 *
 * When Map entries are created a private copy of the key is created
 * to save it from user modification.  This memory is returned when
 * the entry is removed.
 * An implementation uses Map_copyKey() and Map_freeKey() to save and
 * free key memory, respectively.  The general template for these
 * assumes that the key is passed by value (i.e., a scalar or struct). 
 * There are also specialized implementations for string (char *)
 * keys. For other pointer types the user is supposed to define 
 * additional specializations before instatiating the Map class.
 */
template <class KeyT>
   inline KeyT Map_copyKey(KeyT key) { return key; }
template <class KeyT>
  inline void Map_freeKey(KeyT key) {};

/* 
 * String keys need to be copied
 */
inline const char *Map_copyKey(const char *key) { return strdup(key); }
inline void Map_freeKey(const char *key) { free((void *)key); }

/*
 * Non-key values
 *
 * Map implementations may make use of a distinguished value of KeyT
 * for their own purposes (the 'non-key').  This value cannot be used by
 * clients of the Map class.  Implementations are supposed to catch attempts
 * to use this value.
 *
 * void Map_noKey(KeyT &key) sets key to the distinguished value.
 * Boolean Map_noKeyP(KeyT key) checks for it.
 *
 * The template and specializations below cover the most common cases.
 * Others need to be defined by the user.
 */
template <class KeyT>
  inline void Map_noKey(const KeyT *&key) { key = 0; }
template <class KeyT>
  inline Boolean Map_noKeyP(const KeyT *key) { return key == 0; }

/*
 * Signed integers use the smallest negative value as the non-key
 */
const short ShortNokeyValue = (short)(1u<<(sizeof(short)*NBBY-1));
const int IntNokeyValue = (int)(1u<<(sizeof(int)*NBBY-1));
const long LongNokeyValue = (long)(1uL<<(sizeof(long)*NBBY-1));

inline void Map_noKey(int &key) { key = IntNokeyValue; }
inline Boolean Map_noKeyP(int key) { return key == IntNokeyValue; }
inline void Map_noKey(short int &key) { key = ShortNokeyValue; }
inline Boolean Map_noKeyP(short int key) { return key == ShortNokeyValue; }
inline void Map_noKey(long int &key) { key = LongNokeyValue; }
inline Boolean Map_noKeyP(long int key) { return key == LongNokeyValue; }

/*
 * Unsigned integers use the largest value as the non-key
 */
const short unsigned UShortNokeyValue = ~(short unsigned)0;
const unsigned UIntNokeyValue = ~(unsigned)0;
const long unsigned ULongNokeyValue = ~(long unsigned)0;

inline void Map_noKey(unsigned &key) { key = UIntNokeyValue; }
inline Boolean Map_noKeyP(unsigned key) { return key == UIntNokeyValue; }
inline void Map_noKey(short unsigned &key) { key = UShortNokeyValue; }
inline Boolean Map_noKeyP(short unsigned key) { return key == UShortNokeyValue; }
inline void Map_noKey(long unsigned &key) { key = ULongNokeyValue; }
inline Boolean Map_noKeyP(long unsigned key) { return key == ULongNokeyValue; }

/*
 * Floats use HUGE_VAL as the non-key
 */
inline void Map_noKey(float &key) { key = HUGE_VAL; }
inline Boolean Map_noKeyP(float key) { return key == HUGE_VAL; }
inline void Map_noKey(double &key) { key = HUGE_VAL; }
inline Boolean Map_noKeyP(double key) { return key == HUGE_VAL; }

#endif /* _Map_h_ */

