/*
 * DStructThreads.cc
 *
 * Provide mechanisms for freeing thread-specific resources when a thread
 * terminates long before the process.
 *
 * Copyright (c) 2012, SRI International.  All Rights Reserved.
 */

#include "DStructThreads.h"

#include "BlockMalloc.h"
#include "tserror.h"

void 
DStructThreads::freeThread() {
    BM_freeThread();

#ifndef NO_TLS
    // This is a special case that completely doesn't
    // exist unless using TLS.
    srilm_tserror_freeThread();
#endif // NO_TLS
}
