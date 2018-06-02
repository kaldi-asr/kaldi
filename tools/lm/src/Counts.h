/*
 * Counts.h --
 *	Utility functions for counts
 *
 * Copyright (c) 2006 SRI International, 2013-2016 Microsoft Corp.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/lm/src/Counts.h,v 1.7 2016/04/08 23:34:42 stolcke Exp $
 *
 */

#ifndef _Counts_h_
#define _Counts_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif
#include <stdio.h>

#include "Boolean.h"
#include "XCount.h"
#include "File.h"

#ifdef USE_LONGLONG_COUNTS
typedef unsigned long long Count;	/* a count of something */
#else
typedef unsigned long Count;		/* a count of something */
#endif
typedef double FloatCount;		/* a fractional count */

extern const unsigned FloatCount_Precision;

/*
 * Type-dependent count <--> string conversions
 */
extern char ctsBuffer[100];

inline const char *
countToString(unsigned count)
{
    sprintf(ctsBuffer, "%u", count);
    return ctsBuffer;
}

inline const char *
countToString(int count)
{
    sprintf(ctsBuffer, "%d", count);
    return ctsBuffer;
}

inline const char *
countToString(long count)
{
    sprintf(ctsBuffer, "%ld", count);
    return ctsBuffer;
}

inline const char *
countToString(unsigned long count)
{
    sprintf(ctsBuffer, "%lu", count);
    return ctsBuffer;
}

inline const char *
countToString(unsigned long long count)
{
    sprintf(ctsBuffer, "%llu", count);
    return ctsBuffer;
}

inline const char *
countToString(XCount count)
{
    return countToString((XCountValue)count);
}

template <class CountT>
inline const char *
countToString(CountT count)
{
    sprintf(ctsBuffer, "%.*lg", FloatCount_Precision, (double)count);
    return ctsBuffer;
}

inline Boolean
stringToCount(const char *str, unsigned int &count)
{
    /*
     * scanf("%u") doesn't check for a positive sign, so we have to ourselves.
     */
    return (*str != '-' && sscanf(str, "%u", &count) == 1);
}

inline Boolean
stringToCount(const char *str, int &count)
{
    return (sscanf(str, "%d", &count) == 1);
}

inline Boolean
stringToCount(const char *str, unsigned short &count)
{
    /*
     * scanf("%u") doesn't check for a positive sign, so we have to ourselves.
     */
    return (*str != '-' && sscanf(str, "%hu", &count) == 1);
}

inline Boolean
stringToCount(const char *str, unsigned long &count)
{
    /*
     * scanf("%lu") doesn't check for a positive sign, so we have to ourselves.
     */
    return (*str != '-' && sscanf(str, "%lu", &count) == 1);
}

inline Boolean
stringToCount(const char *str, unsigned long long &count)
{
    /*
     * scanf("%lu") doesn't check for a positive sign, so we have to ourselves.
     */
    return (*str != '-' && sscanf(str, "%llu", &count) == 1);
}

inline Boolean
stringToCount(const char *str, long &count)
{
    return (sscanf(str, "%ld", &count) == 1);
}

inline Boolean
stringToCount(const char *str, XCount &count)
{
    XCountValue x;
    if (stringToCount(str, x)) {
    	count = x;
	return true;
    } else {
    	return false;
    }
}

template <class CountT>
static inline Boolean
stringToCount(const char *str, CountT &count)
{
    double x;
    if (sscanf(str, "%lf", &x) == 1) {
	count = x;
	return true;
    } else {
	return false;
    }
}

/*
 * Binary count I/O
 * 	Functions return 0 on failure,  number of bytes read/written otherwise
 */

unsigned writeBinaryCount(File &file, unsigned long long count,
						    unsigned minBytes = 0);
unsigned writeBinaryCount(File &file, float count);
unsigned writeBinaryCount(File &file, double count);

inline unsigned
writeBinaryCount(File &file, unsigned long count) {
    return writeBinaryCount(file, (unsigned long long)count);
}

inline unsigned
writeBinaryCount(File &file, unsigned count)
{
    return writeBinaryCount(file, (unsigned long long)count);
}

inline unsigned
writeBinaryCount(File &file, unsigned short count)
{
    return writeBinaryCount(file, (unsigned long long)count);
}

inline unsigned
writeBinaryCount(File &file, XCount count)
{
    return writeBinaryCount(file, (unsigned long long)count);
}

unsigned readBinaryCount(File &file, unsigned long long &count);
unsigned readBinaryCount(File &file, float &count);
unsigned readBinaryCount(File &file, double &count);

inline unsigned
readBinaryCount(File &file, unsigned long &count)
{
    unsigned long long lcount;
    unsigned result = readBinaryCount(file, lcount);
    if (result > 0) {
	count = (unsigned long)lcount;
    }
    return result;
}

inline unsigned
readBinaryCount(File &file, unsigned &count)
{
    unsigned long long lcount;
    unsigned result = readBinaryCount(file, lcount);
    if (result > 0) {
	count = (unsigned int)lcount;
    }
    return result;
}

inline unsigned
readBinaryCount(File &file, unsigned short &count)
{
    unsigned long long lcount;
    unsigned result = readBinaryCount(file, lcount);
    if (result > 0) {
	count = (unsigned short)lcount;
    }
    return result;
}

inline unsigned
readBinaryCount(File &file, XCount &count)
{
    unsigned long long lcount;
    unsigned result = readBinaryCount(file, lcount);
    if (result > 0) {
	count = (XCountValue)lcount;
    }
    return result;
}

#endif /* _Counts_h_ */
