/*
 * BlockMalloc --
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

#ifndef _BlockMalloc_h_
#define _BlockMalloc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

#ifdef NO_BLOCK_MALLOC

#include <stdlib.h>

#define BM_malloc(size)	malloc(size)
#define BM_free(p, size) free(p)

#else /* NO_BLOCK_MALLOC */

void *BM_malloc(size_t size);
void BM_free(void *chunk, size_t size);		/* Note: must supply chunk size */

#endif /* NO_BLOCK_MALLOC */

void BM_freeThread();
void BM_printstats();

#ifdef __cplusplus
}
#endif

#endif /* _BlockMalloc_h_ */

