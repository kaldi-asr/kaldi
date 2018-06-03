/*
 * Array.cc --
 *	Extensible array implementation
 *
 */

#ifndef _Array_cc_
#define _Array_cc_

#ifndef lint
static char Array_Copyright[] = "Copyright (c) 1995-2005 SRI International, 2013 Microsoft Corp.  All Rights Reserved.";
static char Array_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Array.cc,v 1.15 2013/05/24 04:50:05 frandsen Exp $";
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "Array.h"

#undef INSTANTIATE_ARRAY
#define INSTANTIATE_ARRAY(DataT) \
	template class Array<DataT>

/*
 * extend the size of an Array to accomodate size elements
 * 	Note we want to zero-initialize the data elements by default,
 *	so we call their initializers with argument 0.  This means
 *	that all data type used as arguments to the Array template
 *	need to provide an initializer that accepts 0 as a single argument.
 */
template <class DataT>
void
Array<DataT>::alloc(unsigned size, Boolean zero)
{
    // size is highest index needed so size + 1 is number
    // of elements, and pad by half current size for growth. 
    unsigned int newSize = size + 1 + alloc_size/2;
    DataT *newData = new DataT[newSize];
    assert(newData != 0);

    if (zero) {
        memset(newData, 0, newSize * sizeof(DataT));
    }

    for (unsigned i = 0; i < alloc_size; i++) {
	newData[i] = _data[i];
    }

    delete [] _data;

    _data = newData;
    alloc_size = newSize;
}

template <class DataT>
void
Array<DataT>::memStats(MemStats &stats) const
{
    size_t mySize = alloc_size * sizeof(_data[0]);

    stats.total += mySize;
    stats.wasted += (alloc_size - _size) * sizeof(_data[0]);

    stats.allocStats[mySize > MAX_ALLOC_STATS ?
			    MAX_ALLOC_STATS : mySize] += 1;
}

template <class DataT>
Array<DataT> &
Array<DataT>::operator= (const Array<DataT> &other)
{
#ifdef DEBUG
    cerr << "warning: Array::operator= called\n";
#endif

    if (&other == this) {
	return *this;
    }

    delete [] _data;

    _base = other._base;
    _size = other._size;

    // make new array only as large as needed
    alloc_size = other._size;

    _data = new DataT[alloc_size];
    assert(_data != 0);

    for (unsigned i = 0; i < _size; i++) {
	_data[i] = other._data[i];
    }

    return *this;
}

template <class DataT>
ZeroArray<DataT> &
ZeroArray<DataT>::operator= (const ZeroArray<DataT> &other)
{
#ifdef DEBUG
    cerr << "warning: ZeroArray::operator= called\n";
#endif

    if (&other == this) {
        return *this;
    }

    delete [] Array<DataT>::_data;

    Array<DataT>::_base = other._base;
    Array<DataT>::_size = other._size;

    // make new array only as large as needed
    Array<DataT>::alloc_size = other._size;

    Array<DataT>::_data = new DataT[Array<DataT>::alloc_size];
    assert(Array<DataT>::_data != 0);

    for (unsigned i = 0; i < Array<DataT>::_size; i++) {
        Array<DataT>::_data[i] = other._data[i];
    }

    return *this;
}

#endif /* _Array_cc_ */
