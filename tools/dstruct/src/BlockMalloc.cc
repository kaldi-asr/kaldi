/*
 * BlockMalloc.c --
 *      A caching, blocked memory allocator
 *
 * Copyright 2011, Andreas Stolcke.  
 * Permission to use, copy, modify, and distribute this
 * software and its documentation for any purpose and without
 * fee is hereby granted, provided that the above copyright
 * notice appear in all copies.  The author
 * makes no representations about the suitability of this
 * software for any purpose.  It is provided "as is" without
 * express or implied warranty.
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2011 A. Stolcke";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/BlockMalloc.cc,v 1.4 2015-04-16 23:30:19 frandsen Exp $";
#endif

#include <stdlib.h>
#include <stdio.h>

#include "BlockMalloc.h"
#include "TLSWrapper.h"

#define BLOCK_MALLOC_BLKSIZE	(128 * 1024)		/* block size in which memory is allocated (in words) */
#define BLOCK_MALLOC_MAXWORDS	16			/* max chunk size managed in blocks (in words) */

#define WORD_SIZE		sizeof(void *)

struct BMchunk {
    struct BMchunk *nextfree;		/* pointer to next free chunk */
};

static TLSW_ARRAY(BMchunk*, freeListsTLS, BLOCK_MALLOC_MAXWORDS);	/* Note: initialized to null pointers */
static TLSW_ARRAY(BMchunk*, mallocListTLS, BLOCK_MALLOC_MAXWORDS);
static TLSW_ARRAY(unsigned, allocCountsTLS, BLOCK_MALLOC_MAXWORDS);

#ifndef NO_BLOCK_MALLOC

void *
BM_malloc(size_t size)
{
   BMchunk **mallocList  = TLSW_GET_ARRAY(mallocListTLS);
   BMchunk **freeLists   = TLSW_GET_ARRAY(freeListsTLS);
   unsigned *allocCounts = TLSW_GET_ARRAY(allocCountsTLS);

   size_t nwords = ((size-1) / WORD_SIZE + 1);

   if (nwords >= BLOCK_MALLOC_MAXWORDS) {
	/*
	 * We don't manage chunks this big
	 */
	return malloc(size);
   } else {
	void *chunk;

	if (freeLists[nwords] == NULL) {
	    /*
	     * Allocate a new block of chunks 
	     */
	    size_t nchunks = BLOCK_MALLOC_BLKSIZE / nwords;
	    size_t blockSize = nchunks * nwords * WORD_SIZE;

	    // Track linked list of allocated memory.
            size_t allocTrackerSize = sizeof(struct BMchunk);

	    void *fullChunks = malloc(allocTrackerSize + blockSize);

	    if (fullChunks == NULL) {
		return NULL;
	    } else {
	        // Block memory begins after header for tracking
	        // allocated blocks.
	        void *newChunks = (char*)fullChunks + allocTrackerSize;
                struct BMchunk* hdr = (struct BMchunk *)fullChunks;
		// This is just a linked list. In this case, it
		// is the next block to free (or NULL) when freeing.
		hdr->nextfree = mallocList[nwords];
                mallocList[nwords] = hdr;
		/*
		 * Link new chunks into the free list
		 */
		char *nextChunk;

		for (nextChunk = (char *)newChunks + blockSize - nwords * WORD_SIZE;
		     nextChunk >= (char *)newChunks;
		     nextChunk -= nwords * WORD_SIZE)
		{
		    ((struct BMchunk *)nextChunk)->nextfree =
				freeLists[nwords];
		    freeLists[nwords] = (struct BMchunk *)nextChunk;
		}
	    }

	    allocCounts[nwords]++;
	} 

	/*
	 * We have allocated memory of the right size
	 */
	chunk = freeLists[nwords];
	freeLists[nwords] = freeLists[nwords]->nextfree;

	return chunk;
   }
}

void
BM_free(void *chunk, size_t size)
{
   BMchunk **freeLists = TLSW_GET_ARRAY(freeListsTLS);

   size_t nwords = ((size-1) / WORD_SIZE + 1);

   if (nwords >= BLOCK_MALLOC_MAXWORDS) {
	/*
	 * We don't manage chunks this big
	 */
	free(chunk);
   } else {
	/*
	 * Add this chunk to the front of the free list
	 */
	((struct BMchunk *)chunk)->nextfree = 
				freeLists[nwords];
	freeLists[nwords] = (struct BMchunk *)chunk;
   }
}

#endif /* NO_BLOCK_MALLOC */

void
BM_freeThread()
{
    BMchunk **mallocList = TLSW_GET_ARRAY(mallocListTLS);

    if (mallocList != NULL) {
        unsigned i;
        for (i = 0; i < BLOCK_MALLOC_MAXWORDS; i++) {
	    BMchunk *next = mallocList[i];
	    while (next != NULL) {
		BMchunk *tmp = next->nextfree;
		free(next);
		next = tmp;
	    }
	}
    }

    TLSW_FREE(mallocListTLS);
    TLSW_FREE(allocCountsTLS);
    TLSW_FREE(freeListsTLS);
}

void
BM_printstats()
{
    unsigned *allocCounts = TLSW_GET_ARRAY(allocCountsTLS);
    unsigned i;

    for (i = 0; i < BLOCK_MALLOC_MAXWORDS; i++) {
	if (allocCounts[i] > 0) {
	    fprintf(stderr, "%u blocks of %u-word chunks\n", allocCounts[i], i);
	}
    }
}
