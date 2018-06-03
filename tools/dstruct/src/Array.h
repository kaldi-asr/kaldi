/*
 * Array.h --
 *	Extensible array class
 *
 * Copyright (c) 1995-2010 SRI International, 2013 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Array.h,v 1.30 2014-04-07 18:19:24 frandsen Exp $
 *
 */

#ifndef _Array_h_
#define _Array_h_

#include <assert.h>

#include "Boolean.h"
#include "MemStats.h"

template <class DataT>
class Array
{
public:
    Array(int base = 0, unsigned size = 0)
	: _base(base), _size(size), _data(0), alloc_size(0)
	{ if (size > 0) { alloc(size-1); } }
    Array(Array<DataT> &source)
	: _base(source._base), _size(0), _data(0), alloc_size(0)
    	{ *this = source; }

    // We could make this virtual but it would mean extra memory overhead for the vtable
    // and none of the derived classes below allocate memory of their own.
    ~Array() { delete [] _data; }

    void clear() { delete [] _data; _data = 0; _size = alloc_size = 0; }

    DataT &operator[](long index)
    	{ unsigned long offset = index - _base; assert((long)offset >= 0);
	  if (offset >= _size) {
	    _size = offset + 1;
	    if (offset >= alloc_size) { alloc(offset); }
	  }
	  return _data[offset];
	}
    /* these are redundant, but work around problems in MS Visual C++ */
    DataT &operator[] (unsigned long long index) { return (*this)[(long) index]; }
    DataT &operator[] (long long index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned long index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned index) { return (*this)[(long) index]; }
    DataT &operator[] (int index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned short index) { return (*this)[(long) index]; }
    DataT &operator[] (short index) { return (*this)[(long) index]; }

    Array<DataT> & operator= (const Array<DataT> &other);

    operator DataT* () { return data(); }
    operator const DataT* () const { return data(); }

    DataT *data() const { return _data - _base; }
    int base() const { return _base; }
    unsigned int size() const { return _size; }


    void memStats(MemStats &stats) const;

protected:
    int _base;
    unsigned int _size;		/* used size */
    DataT *_data;
    unsigned int alloc_size;	/* allocated size */

    // Note that size is not a number of elements but highest
    // desired array offset.
    void alloc(unsigned size, Boolean zero = false);
};

/*
 * Zero-initialized array
 */
template <class DataT>
class ZeroArray: public Array<DataT>
{
public:
    ZeroArray(int base = 0, unsigned size = 0)
	: Array<DataT>(base, 0)
	{ if (size > 0) { Array<DataT>::_size = size;
	  Array<DataT>::alloc(size-1, true); } }

    ZeroArray(ZeroArray<DataT> &source)
        : Array<DataT>(source._base, 0)
        { *this = source; }

    DataT &operator[](long index)
    	{ unsigned long offset = index - Array<DataT>::_base;
	  assert((long)offset >= 0);
	  if (offset >= Array<DataT>::_size) {
	    Array<DataT>::_size = offset + 1;
	    if (offset >= Array<DataT>::alloc_size)
	      { Array<DataT>::alloc(offset, true); }
	  }
	  return Array<DataT>::_data[offset];
	}
    /* these are redundant, but work around problems in MS Visual C++ */
    DataT &operator[] (unsigned long long index) { return (*this)[(long) index]; }
    DataT &operator[] (long long index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned long index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned index) { return (*this)[(long) index]; }
    DataT &operator[] (int index) { return (*this)[(long) index]; }
    DataT &operator[] (unsigned short index) { return (*this)[(long) index]; }
    DataT &operator[] (short index) { return (*this)[(long) index]; }

    ZeroArray<DataT> & operator= (const ZeroArray<DataT> &other);
};

/*
 * An optimized version of Array for when the size never changes
 */
template <class DataT>
class StaticArray: public Array<DataT>
{
public:
    StaticArray(unsigned size)
	: Array<DataT>(0, size) { }
    StaticArray(int base, unsigned size)
	: Array<DataT>(base, size) { }

    DataT &operator[](int index)	// dispense with index range check
    	{ return Array<DataT>::_data[index - Array<DataT>::_base]; }
    /* these are redundant, but work around problems in MS Visual C++ */
    DataT &operator[](unsigned index) { return (*this)[(int)index]; }
    DataT &operator[](unsigned short index) { return (*this)[(int)index]; }
    DataT &operator[](short index) { return (*this) [(int)index]; }

private:
    // Not supported (else get defaults)
    StaticArray<DataT> & operator= (const StaticArray<DataT> &other);
    StaticArray(StaticArray<DataT> &source);
};

/*
 * Macro defining a linear array sized at run-time:
 * gcc and icc allow this as part of the language, but many other C++ compilers
 * don't (Sun Studio, MS Visual C).
 * If the 2nd case is used, SRILM_NEED_ARRAY_CC_FOR_MAKE_ARRAY will get set
 * and your code should either simply explicitly include Array.cc directly
 * or paste the following after including Array.h for selective inclusion:
 * #ifdef SRILM_NEED_ARRAY_CC_FOR_MAKE_ARRAY
 * #include "Array.cc"
 * #endif
 */
#if !defined(DEBUG) && defined(__GNUC__) && !defined(__clang__) && (!defined(__INTEL_COMPILER) || __INTEL_COMPILER >=900)
# define makeArray(T, A, n)		T A[n]
#else
# define makeArray(T, A, n)		StaticArray<T> A(n)
# define SRILM_NEED_ARRAY_CC_FOR_MAKE_ARRAY 1
#endif

#endif /* _Array_h_ */
