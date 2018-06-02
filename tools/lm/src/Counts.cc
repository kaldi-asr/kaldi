
/*
 * Counts.cc --
 *	Functions for handling counts
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2006 SRI International, 2013-2016 Micrsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Counts.cc,v 1.9 2016/04/08 23:34:42 stolcke Exp $";
#endif

#include <stdio.h>
#include <string.h>
#include <assert.h>
#if !defined(_MSC_VER) && !defined(WIN32)
#include <sys/param.h>
#endif
#include <limits>

#include "Boolean.h"
#include "Counts.h"

const unsigned FloatCount_Precision = numeric_limits<FloatCount>::digits10 + 1;

/*
 * Byte swapping
 */

static inline
Boolean isLittleEndian()
{
    static Boolean haveEndianness = false;
    static Boolean endianIsLittle;

    if (!haveEndianness) {
	union {
		char c[2];
		short i;
	} data;

	data.i = 1;

	endianIsLittle = (data.c[0] != 0);
	haveEndianness = true;
    }

    return endianIsLittle;
}

static inline
void byteSwap(void *data, unsigned size)
{
    if (isLittleEndian() && (size > 0)) {

	if (size > sizeof(double)) {
	    // Error; unexpected (do nothing)
	    return;
	}

	char byteArray[sizeof(double)];

	for (unsigned i = 0; i < size; i ++) {
	    byteArray[i] = ((char *)data)[size - i - 1];
	}
	memcpy(data, byteArray, size);
    }
}

/*
 * Integer I/O
 */

#ifndef NBBY
#define NBBY 8
#endif

const unsigned short isLongBit = (unsigned short)1 << (sizeof(short)*NBBY-1);
const unsigned int isLongLongBit = (unsigned int)1 << (sizeof(int)*NBBY-2);
const unsigned long long isTooLongBit =
			(unsigned long long)1 << (sizeof(long long)*NBBY-2);

unsigned
writeBinaryCount(File &file, unsigned long long count, unsigned minBytes)
{
    if (minBytes <= sizeof(short) && count < isLongBit) {
	short int scount = count;

	byteSwap(&scount, sizeof(scount));
	file.fwrite(&scount, sizeof(scount), 1);
	return sizeof(scount);
    } else if (minBytes <= sizeof(int) && count < isLongLongBit) {
	unsigned int lcount = count;
	lcount |= (unsigned int)isLongBit << (sizeof(short)*NBBY);

	byteSwap(&lcount, sizeof(lcount));
	file.fwrite(&lcount, sizeof(lcount), 1);
	return sizeof(lcount);
    } else if (count < isTooLongBit) {
	unsigned int lcount = (count >> sizeof(int)*NBBY) |
			      ((unsigned int)isLongBit << (sizeof(short)*NBBY))|
			      isLongLongBit;

	byteSwap(&lcount, sizeof(lcount));
	file.fwrite(&lcount, sizeof(lcount), 1);

	lcount = (unsigned int)count;
	byteSwap(&lcount, sizeof(lcount));
	file.fwrite(&lcount, sizeof(lcount), 1);
	return 2*sizeof(lcount);
    } else {
	cerr << "writeBinaryCount: count " << count << " is too large\n";
	return 0;
    }
}

unsigned
readBinaryCount(File &file, unsigned long long &count)
{
    unsigned short scount;

    if (file.fread(&scount, sizeof(scount), 1) != 1) {
	return 0;
    } else {
	byteSwap(&scount, sizeof(scount));

	if (!(scount & isLongBit)) {
	    // short count
	    count = scount;
	    return sizeof(scount);
	} else {
	    // long or long long count
	    unsigned int lcount = (unsigned)(scount & ~isLongBit)
						<< sizeof(short)*NBBY;

	    // read second half of long count
	    if (file.fread(&scount, sizeof(scount), 1) != 1) {
		cerr << "readBinaryCount: incomplete long count\n";
		return 0;
	    } else {
		byteSwap(&scount, sizeof(scount));
		

		// assemble long count from two shorts
		lcount |= (unsigned int)scount;

		if (!(lcount & isLongLongBit)) {
		    // long count
		    count = lcount;
		    return sizeof(lcount);
		} else {
		    // long long count
		    count = (unsigned long long)(lcount & ~isLongLongBit)
						<< sizeof(unsigned)*NBBY;

		    // read second half of long count
		    if (file.fread(&lcount, sizeof(lcount), 1) != 1) {
			cerr << "readBinaryCount: incomplete long long count\n";
			return 0;
		    } else {
			byteSwap(&lcount, sizeof(lcount));

			// assemble long long count from two longs
			count |= (unsigned long long)lcount;
			return 2*sizeof(lcount);
		    }
		}
	    }
	}
    }
}


/*
 * Floating point I/O
 */

unsigned
writeBinaryCount(File &file, float count)
{
    byteSwap(&count, sizeof(count));

    if (file.fwrite(&count, sizeof(count), 1) == 1) {
	return sizeof(count);
    } else {
	return 0;
    }
}

unsigned
writeBinaryCount(File &file, double count)
{
    byteSwap(&count, sizeof(count));

    if (file.fwrite(&count, sizeof(count), 1) == 1) {
	return sizeof(count);
    } else {
	return 0;
    }
}


unsigned
readBinaryCount(File &file, float &count)
{
    if (file.fread(&count, sizeof(count), 1) == 1) {
	byteSwap(&count, sizeof(count));
	return sizeof(count);
    } else {
	return 0;
    }
}

unsigned
readBinaryCount(File &file, double &count)
{
    if (file.fread(&count, sizeof(count), 1) == 1) {
	byteSwap(&count, sizeof(count));
	return sizeof(count);
    } else {
	return 0;
    }
}

